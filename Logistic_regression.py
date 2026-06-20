import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.column import Column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from Evaluation import evaluation

# Helper 
import numpy as np
import matplotlib.pyplot as plt

def plot_cvr_boxplot_from_arrays(
    y_true_bin: np.ndarray,
    p_pred: np.ndarray,
    *,
    title: str = "Test-set CVR predictions by true outcome",
    showfliers: bool = False,
):

    y_true_bin = np.asarray(y_true_bin).astype(int).ravel()
    p_pred = np.asarray(p_pred).astype(float).ravel()

    p0 = p_pred[y_true_bin == 0]
    p1 = p_pred[y_true_bin == 1]

    fig, ax = plt.subplots()
    ax.boxplot([p0, p1], labels=["No conversion (y=0)", "Conversion (y=1)"], showfliers=showfliers)
    ax.set_ylabel("Predicted conversion probability")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_cvr_boxplot_from_dataframe(
    test_pdf,
    *,
    target_col: str = "conversion",
    prob_col: str = "p_pred",
    title: str = "Test-set CVR predictions by true outcome",
    showfliers: bool = False,
):
    y = test_pdf[target_col].astype(np.int32).values
    p = test_pdf[prob_col].astype(np.float64).values
    plot_cvr_boxplot_from_arrays(y, p, title=title, showfliers=showfliers)


def as_ts(x):
    if isinstance(x, Column):
        return x
    return F.to_timestamp(F.lit(str(x)))

def lr(
    df,
    train_end,
    test_end,
    NUM_COLS,
    CAT_INT_COLS,
    #CAT_COUNTRY,
    TARGET="conversion",
    START_TS="timestamp_dt",
    EVENT_TS="conversion_timestamp_dt",
    MASK_TRAIN_COL="mask_train",
    MASK_EVAL_COL="mask_eval",
    drop_cols=("bucket", "bucket_vec", "mask_train", "mask_eval"),
    max_delay_days: int = 12,
    # MASK_COL="mask",   
    # drop_cols=("bucket", "bucket_vec", "mask"),     
    **kwargs
):
    reg_C = 0.001
    max_iter = 2000

    all_cols = set(df.columns)

    needed = {TARGET, START_TS}
    missing = [c for c in needed if c not in all_cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Cast timestamps & target
    d = (
        df
        .withColumn(TARGET, F.col(TARGET).cast("int"))
        .withColumn(START_TS, F.to_timestamp(F.col(START_TS)))
    )

    ASOF_TRAIN = as_ts(train_end)
    ASOF_TEST = as_ts(test_end) #added
    train_sdf = (d.where(F.col(START_TS) < ASOF_TRAIN).where(F.element_at(F.col(MASK_TRAIN_COL), -1) == 1))
    test_sdf = ( d.where((F.col(START_TS) >= ASOF_TRAIN) & (F.col(START_TS) < ASOF_TEST)).where(F.element_at(F.col(MASK_EVAL_COL), -1) == 1))

    print("TEST COUNT:", test_sdf.count())
    print("TEST base_rate spark:", test_sdf.select(F.avg(F.col(TARGET)).alias("br")).first()["br"])
    print("TEST y counts:", test_sdf.groupBy(TARGET).count().orderBy(TARGET).collect())

    # Optionally drop user-provided columns (bucket vectors etc.)
    for c in drop_cols:
        if c in train_sdf.columns:
            train_sdf = train_sdf.drop(c)
        if c in test_sdf.columns:
            test_sdf = test_sdf.drop(c)

    train_pdf = train_sdf.toPandas()
    test_pdf  = test_sdf.toPandas()

    print("TEST base_rate pandas:", float(test_pdf[TARGET].mean()))
    print(test_pdf[TARGET].value_counts(dropna=False))

    if len(train_pdf) == 0 or len(test_pdf) == 0:
        raise ValueError(f"Empty train/test after maturity filtering. Train={len(train_pdf)}, Test={len(test_pdf)}")

  
    # Build feature lists
    # Everything not numeric/target/timestamps/id/mask is categorical
    reserved = {TARGET, START_TS, EVENT_TS}

    NUM_COLS = [c for c in NUM_COLS if c in train_pdf.columns]
    cat_cols = [c for c in (CAT_INT_COLS) if c in train_pdf.columns]

    # Split X/y
    y_train = train_pdf[TARGET].astype(np.int32).values
    y_test  = test_pdf[TARGET].astype(np.int32).values

    X_train = train_pdf[NUM_COLS + cat_cols].copy()
    X_test  = test_pdf[NUM_COLS + cat_cols].copy()

  
    # Numeric transforms: log/log1p, standardize using TRAIN ONLY
    # + missing flags
    # Determine which cols are "strictly positive" based on TRAIN observed values
    pos_cols = []
    nonpos_or_zero_cols = []

    for c in NUM_COLS:
        tr = pd.to_numeric(X_train[c], errors="coerce")
        tr_obs = tr.dropna()
        if len(tr_obs) == 0:
            # if entirely missing in train, treat as nonpos/log1p for safety
            nonpos_or_zero_cols.append(c)
            continue
        if tr_obs.min() > 0:
            pos_cols.append(c)
        else:
            nonpos_or_zero_cols.append(c)

    # Create missing flags first
    for c in NUM_COLS:
        X_train[f"{c}__miss"] = pd.isna(pd.to_numeric(X_train[c], errors="coerce")).astype(np.int32)
        X_test[f"{c}__miss"]  = pd.isna(pd.to_numeric(X_test[c],  errors="coerce")).astype(np.int32)

    # Apply transforms
    def safe_log(x):
        return np.log(x)

    def safe_log1p_nonneg(x):
        # clamp negatives to 0 to avoid log1p issues
        return np.log1p(np.clip(x, 0, None))

    # Compute train-only mean/std on transformed values (observed only)
    num_stats = {}  # col -> (mu, sd, transform_kind)

    for c in NUM_COLS:
        tr_raw = pd.to_numeric(X_train[c], errors="coerce").astype(float)

        if c in pos_cols:
            tr_t = safe_log(tr_raw)
            kind = "log"
        else:
            tr_t = safe_log1p_nonneg(tr_raw)
            kind = "log1p"

        # stats on observed
        tr_obs = tr_t[~np.isnan(tr_t)]
        if len(tr_obs) == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = float(tr_obs.mean())
            sd = float(tr_obs.std(ddof=0))
            if sd == 0.0:
                sd = 1.0

        num_stats[c] = (mu, sd, kind)

        # transform+standardize train
        X_train[c] = (tr_t - mu) / sd

        # transform+standardize test using TRAIN stats
        te_raw = pd.to_numeric(X_test[c], errors="coerce").astype(float)
        if kind == "log":
            te_t = safe_log(te_raw)
        else:
            te_t = safe_log1p_nonneg(te_raw)
        X_test[c] = (te_t - mu) / sd

        # impute missing with 0 AFTER standardization
        X_train[c] = X_train[c].fillna(0.0)
        X_test[c]  = X_test[c].fillna(0.0)

    # Update feature lists to include missing flags
    num_and_flags = NUM_COLS + [f"{c}__miss" for c in NUM_COLS]


    # sklearn pipeline: OneHot(cat) + passthrough numeric, then logistic regression
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", "passthrough", num_and_flags),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        # penalty="l2",
        # C=reg_C,
        solver="lbfgs",      
        max_iter=max_iter,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Predict probabilities on matured test
    p_test = pipe.predict_proba(X_test)[:, 1].astype(np.float64)

    plot_cvr_boxplot_from_arrays(y_test, p_test)
    # Evaluate + return
    eval_metric_array = evaluation(y_test, p_test)
    return eval_metric_array
    
