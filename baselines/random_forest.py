import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.column import Column

from scipy.special import expit as _sigmoid_scipy, logit as _logit_scipy
from scipy.optimize import minimize as _minimize_scipy

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from Evaluation import evaluation


# Platt calibration utilities
def _fit_platt_rf(
    probs:      np.ndarray,
    y_true:     np.ndarray,
    *,
    fit_slope:  bool = False,
) -> tuple:

    probs  = np.asarray(probs,  dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    if probs.size == 0:
        return 1.0, 0.0

    y_sum = float(y_true.sum())
    if y_sum == 0.0 or y_sum == float(y_true.size):
        print("  [Platt] WARNING: no positives or no negatives in calibration set — skipping.")
        return 1.0, 0.0

    logits = _logit_scipy(np.clip(probs, 1e-7, 1.0 - 1e-7))

    def _bce(params: np.ndarray) -> float:
        a = float(params[0]) if fit_slope else 1.0
        b = float(params[1]) if fit_slope else float(params[0])
        z = a * logits + b
        p = np.clip(_sigmoid_scipy(z), 1e-12, 1.0 - 1e-12)
        return -float(np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    x0     = [1.0, 0.0] if fit_slope else [0.0]
    result = _minimize_scipy(_bce, x0, method="BFGS", options={"maxiter": 500})

    if fit_slope:
        a, b = float(result.x[0]), float(result.x[1])
    else:
        a, b = 1.0, float(result.x[0])

    print(
        f"  [Platt] a={a:.4f}  b={b:.4f}  "
        f"({'slope+intercept' if fit_slope else 'intercept only'})"
    )
    return a, b

# Apply a fitted Platt transform to an array of raw probabilities
def _apply_platt_rf(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    logits = _logit_scipy(np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7))
    return _sigmoid_scipy(a * logits + b).astype(np.float64)



# Calibration diagnostics
def _reliability_curve_rf(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple:

    y_true = np.asarray(y_true, dtype=float).ravel()
    p_pred = np.asarray(p_pred, dtype=float).ravel()

    bin_edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return np.array([p_pred.mean()]), np.array([y_true.mean()])

    bin_ids = np.digitize(p_pred, bin_edges[1:-1], right=True)

    conf, acc = [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        conf.append(float(p_pred[mask].mean()))
        acc.append(float(y_true[mask].mean()))

    return np.array(conf), np.array(acc)


def _expected_calibration_error_rf(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> float:

    y_true = np.asarray(y_true, dtype=float).ravel()
    p_pred = np.asarray(p_pred, dtype=float).ravel()

    bin_edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return abs(float(y_true.mean()) - float(p_pred.mean()))

    bin_ids = np.digitize(p_pred, bin_edges[1:-1], right=True)

    ece = 0.0
    n   = float(len(y_true))
    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(p_pred[mask].mean()))

    return float(ece)


def _plot_reliability_rf(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    title: str = "Reliability diagram — RF",
) -> None:

    conf, acc = _reliability_curve_rf(y_true, p_pred)
    r = max(float(np.max(p_pred)), 0.1)

    plt.figure()
    plt.plot([0.0, r], [0.0, r], "k--", label="Perfect calibration")
    plt.scatter(conf, acc, zorder=3, label="RF")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cvr_boxplot_from_arrays(
    y_true_bin: np.ndarray,
    p_pred:     np.ndarray,
    *,
    title:       str  = "Test-set CVR predictions by true outcome",
    showfliers:  bool = False,
) -> None:
    y_true_bin = np.asarray(y_true_bin).astype(int).ravel()
    p_pred     = np.asarray(p_pred).astype(float).ravel()
    p0 = p_pred[y_true_bin == 0]
    p1 = p_pred[y_true_bin == 1]

    fig, ax = plt.subplots()
    ax.boxplot(
        [p0, p1],
        labels=["No conversion (y=0)", "Conversion (y=1)"],
        showfliers=showfliers,
    )
    ax.set_ylabel("Predicted conversion probability")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_cvr_boxplot_from_dataframe(
    test_pdf,
    *,
    target_col: str  = "conversion",
    prob_col:   str  = "p_pred",
    title:      str  = "Test-set CVR predictions by true outcome",
    showfliers: bool = False,
) -> None:
    y = test_pdf[target_col].astype(np.int32).values
    p = test_pdf[prob_col].astype(np.float64).values
    plot_cvr_boxplot_from_arrays(y, p, title=title, showfliers=showfliers)



# Timestamp helper
def as_ts(x):
    if isinstance(x, Column):
        return x
    return F.to_timestamp(F.lit(str(x)))


def rf(
    df,
    train_end,
    test_end,
    NUM_COLS,
    CAT_INT_COLS,
    TARGET          = "conversion",
    START_TS        = "timestamp_dt",
    EVENT_TS        = "conversion_timestamp_dt",
    MASK_TRAIN_COL  = "mask_train",
    MASK_EVAL_COL   = "mask_eval",
    drop_cols       = ("bucket", "bucket_vec", "mask_train", "mask_eval"),
    max_delay_days  = 12,
    # --- Random Forest hyperparameters ---
    n_estimators      = 500,
    max_depth         = None,
    min_samples_split = 2,
    min_samples_leaf  = 250,
    max_features      = "sqrt",
    max_samples       = 0.7,
    seed              = 42,
    hparams           = None,
    # Platt calibration
    calibrate       = False,
    platt_fit_slope = True,   # False: intercept-only; True: slope + intercept
    n_cal_days      = 1,       # days to carve from training end as calibration set
    outer           = False,
    **kwargs,
):
  
    if hparams:
        n_estimators       = int(hparams.get("n_estimators", n_estimators))
        max_depth          = hparams.get("max_depth", max_depth)
        min_samples_split  = int(hparams.get("min_samples_split", min_samples_split))
        min_samples_leaf   = int(hparams.get("min_samples_leaf", min_samples_leaf))
        max_features       = hparams.get("max_features", max_features)
        max_samples        = hparams.get("max_samples", max_samples)
        #print(f"  [RF] Applied tuned hparams: {hparams}")

    all_cols = set(df.columns)
    missing  = [c for c in (TARGET, START_TS) if c not in all_cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = (
        df
        .withColumn(TARGET,   F.col(TARGET).cast("int"))
        .withColumn(START_TS, F.to_timestamp(F.col(START_TS)))
    )

    ASOF_TRAIN = as_ts(train_end)
    ASOF_TEST  = as_ts(test_end)

    if calibrate:

        ASOF_CAL_START = F.to_timestamp(
            F.date_sub(F.to_date(ASOF_TRAIN), int(n_cal_days))
        )
        train_sdf = (
            d
            .where(F.col(START_TS) < ASOF_CAL_START)
            .where(F.element_at(F.col(MASK_TRAIN_COL), -1) == 1)
        )
        cal_sdf = (
            d
            .where(F.col(START_TS) >= ASOF_CAL_START)
            .where(F.col(START_TS) <  ASOF_TRAIN)
            #.where(F.element_at(F.col(MASK_TRAIN_COL), -1) == 1)
        )
    else:
        # Without calibration: train only on the stale window.
        train_sdf = (
            d
            .where(F.col(START_TS) < ASOF_TRAIN)
            .where(F.element_at(F.col(MASK_TRAIN_COL), -1) == 1)
        )

    test_sdf = (
        d
        .where((F.col(START_TS) >= ASOF_TRAIN) & (F.col(START_TS) < ASOF_TEST))
        .where(F.element_at(F.col(MASK_EVAL_COL), -1) == 1)
    )

    print("TEST COUNT:", test_sdf.count())
    print(
        "TEST base_rate (Spark):",
        test_sdf.select(F.avg(F.col(TARGET)).alias("br")).first()["br"],
    )

    def _drop(sdf):
        return sdf.drop(*[c for c in drop_cols if c in sdf.columns])

    train_sdf = _drop(train_sdf)
    test_sdf  = _drop(test_sdf)
    if calibrate:
        cal_sdf = _drop(cal_sdf)

    train_pdf = train_sdf.toPandas()
    test_pdf  = test_sdf.toPandas()
    if calibrate:
        cal_pdf = cal_sdf.toPandas()
        print(
            f"Sizes — train: {len(train_pdf)}, "
            f"cal: {len(cal_pdf)}, "
            f"test: {len(test_pdf)}"
        )
    else:
        print(f"Sizes — train: {len(train_pdf)}, test: {len(test_pdf)}")

    print("TEST base_rate (pandas):", float(test_pdf[TARGET].mean()))
    print(test_pdf[TARGET].value_counts(dropna=False))

    if len(train_pdf) == 0 or len(test_pdf) == 0:
        raise ValueError(
            f"Empty train/test after maturity filtering. "
            f"train={len(train_pdf)}, test={len(test_pdf)}"
        )

    NUM_COLS = [c for c in NUM_COLS     if c in train_pdf.columns]
    cat_cols = [c for c in CAT_INT_COLS if c in train_pdf.columns]

    y_train = train_pdf[TARGET].astype(np.int32).values
    y_test  = test_pdf[TARGET].astype(np.int32).values
    if calibrate:
        y_cal = cal_pdf[TARGET].astype(np.int32).values

    X_train = train_pdf[NUM_COLS + cat_cols].copy()
    X_test  = test_pdf[NUM_COLS + cat_cols].copy()
    if calibrate:
        X_cal = cal_pdf[NUM_COLS + cat_cols].copy()

    # Determine transform kind from training data
    pos_cols = []
    for c in NUM_COLS:
        tr_obs = pd.to_numeric(X_train[c], errors="coerce").dropna()
        if len(tr_obs) > 0 and float(tr_obs.min()) > 0:
            pos_cols.append(c)

    pos_set = set(pos_cols)

    # Add missingness flags (before any value transforms so NaN detection is clean)
    for X in ([X_train, X_test] + ([X_cal] if calibrate else [])):
        for c in NUM_COLS:
            X[f"{c}__miss"] = (
                pd.isna(pd.to_numeric(X[c], errors="coerce")).astype(np.int32)
            )

    # Fit transform statistics on training data, then apply to all splits
    num_stats: dict = {}

    for c in NUM_COLS:
        tr_raw = pd.to_numeric(X_train[c], errors="coerce").astype(float).values

        if c in pos_set:
            tr_t = np.log(tr_raw)          # NaN where tr_raw <= 0 (should not occur)
        else:
            tr_t = np.log1p(np.clip(tr_raw, 0.0, None))

        obs = ~np.isnan(tr_t)
        mu  = float(tr_t[obs].mean()) if obs.any() else 0.0
        sd  = float(tr_t[obs].std(ddof=0)) if obs.any() else 1.0
        if sd == 0.0:
            sd = 1.0

        num_stats[c] = (mu, sd, c in pos_set)

        def _transform(X_df: pd.DataFrame, col: str, mu_: float, sd_: float, is_pos: bool):
            raw  = pd.to_numeric(X_df[col], errors="coerce").astype(float).values
            t    = np.log(raw) if is_pos else np.log1p(np.clip(raw, 0.0, None))
            X_df[col] = pd.Series(
                np.where(np.isnan(t), 0.0, (t - mu_) / sd_), index=X_df.index
            )

        _transform(X_train, c, mu, sd, c in pos_set)
        _transform(X_test,  c, mu, sd, c in pos_set)
        if calibrate:
            _transform(X_cal, c, mu, sd, c in pos_set)

    num_and_flags = NUM_COLS + [f"{c}__miss" for c in NUM_COLS]


    # sklearn pipeline: OHE(cat) + passthrough(num+flags)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", "passthrough", num_and_flags),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf  = min_samples_leaf,
        max_features      = max_features,
        max_samples       = max_samples,
        random_state      = int(seed),
        n_jobs            = -1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)


    # Predict raw probabilities on test
    p_test = pipe.predict_proba(X_test)[:, 1].astype(np.float64)
    if calibrate:
        p_cal = pipe.predict_proba(X_cal)[:, 1].astype(np.float64)


    # Platt calibration
    platt_a, platt_b = 1.0, 0.0

    if calibrate:
        print(
            f"  Calibration set: {len(cal_pdf)} rows, "
            f"raw conversion rate: {float(y_cal.mean()):.4f}"
        )
        platt_a, platt_b = _fit_platt_rf(
            p_cal,
            y_cal.astype(np.float64),
            fit_slope=platt_fit_slope,
        )
        p_test_raw = p_test.copy()           # preserved for before/after diagnostics
        p_test     = _apply_platt_rf(p_test, platt_a, platt_b)

        print(
            f"  Test set: {len(y_test)} rows, "
            f"positive rate: {float(y_test.mean()):.4f}, "
            f"mean P (raw): {float(p_test_raw.mean()):.4f}, "
            f"mean P (calibrated): {float(p_test.mean()):.4f}"
        )

  
    # Outer diagnostic plots and ECE
    if outer:
        plot_cvr_boxplot_from_arrays(
            y_test,
            p_test,
            title=f"RF CVR predictions ({'calibrated' if calibrate else 'uncalibrated'})",
        )

        if calibrate:
            _plot_reliability_rf(
                y_test,
                p_test_raw,
                title="Reliability diagram — RF (before Platt)",
            )
            _plot_reliability_rf(
                y_test,
                p_test,
                title="Reliability diagram — RF (after Platt)",
            )
            ece_before = _expected_calibration_error_rf(y_test, p_test_raw)
            ece_after  = _expected_calibration_error_rf(y_test, p_test)
            print(f"  ECE before Platt: {ece_before:.5f}")
            print(f"  ECE after  Platt: {ece_after:.5f}")
        else:
            _plot_reliability_rf(
                y_test,
                p_test,
                title="Reliability diagram — RF",
            )
            ece = _expected_calibration_error_rf(y_test, p_test)
            print(f"  ECE: {ece:.5f}")

        print(
            f"  Eval set: {len(y_test)} rows, "
            f"positive rate: {float(y_test.mean()):.4f}, "
            f"mean predicted P: {float(p_test.mean()):.4f}"
        )

    return evaluation(y_test, p_test)
