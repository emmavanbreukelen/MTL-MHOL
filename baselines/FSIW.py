# Feedback Shift Importance Weighting (FSIW)
# Yasui, Morishita, Fujita & Shibata (2020), "A Feedback Shift Correction in Predicting Conversion Rates under Delayed Feedback"

import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from pyspark.sql import functions as F

from Flag_models import (
    _as_ts,
    Model_Agnostic,
    DS_Agnostic,
    masked_bce_with_logits,
    _collect_logits,
    _calibrate_single_head,
    _sigmoid,
    _encode_categoricals_train_test,
    _prepare_numeric,
    plot_cvr_boxplot,
    PLATT_FIT_SLOPE,
)
from Evaluation import evaluation

# estimates P(S=1 | C=1, X, e)
M1_LGBM_PARAMS = dict(          
    learning_rate=0.01,
    num_leaves=64,
    max_depth=6,
    objective="binary",
)
# estimates P(S=1 | Y=0, X, e)
M0_LGBM_PARAMS = dict(          
    learning_rate=0.01,
    num_leaves=63,
    max_depth=6,
    objective="binary",
)

LGBM_N_ESTIMATORS_MAX  = 1000   # not specified in paper - default
LGBM_EARLY_STOP_ROUNDS = 50     # not specified in paper - default
IW_VALID_FRACTION      = 0.10   # not specified in paper - default
IW_PROB_EPS            = 1e-6   # numerical safety clip, not in paper

DEFAULT_TAU_DAYS = 2         



def _materialize_ts(spark, ts_col_or_val):
    col = _as_ts(ts_col_or_val)
    row = spark.range(1).select(col.alias("t")).first()
    return row["t"]


def _fit_iw_estimator(X, y, *, lgbm_params, seed, cat_feature_idx):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=IW_VALID_FRACTION,
        random_state=seed,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS_MAX,
        random_state=seed,
        verbosity=-1,
        **lgbm_params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        categorical_feature=cat_feature_idx,
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOP_ROUNDS, verbose=False)],
    )
    return model


def _build_fsiw_tables(train_pdf, *, START_TS, EVENT_TS, TARGET, train_end_ts, tau_days):
    start_ts = pd.to_datetime(train_pdf[START_TS])
    event_ts = pd.to_datetime(train_pdf[EVENT_TS])
    target   = train_pdf[TARGET].astype(int).values

    T_prime = train_end_ts - pd.Timedelta(days=tau_days)

    # y_i: label observed as of real T (identical formula to
    # event_known_train in Time_Specific_Data_Processing.py)
    y_obs = (
        (target == 1)
        & event_ts.notna()
        & (event_ts <= train_end_ts)
    ).values.astype(int)

    # e_i: real elapsed time at real T (seconds)
    e_real = (train_end_ts - start_ts).dt.total_seconds().values

    # e_i - tau: elapsed time relative to the counterfactual deadline T'
    e_prime = (T_prime - start_ts).dt.total_seconds().values

    before_Tprime = (start_ts < T_prime).values

    # --- D1_iw: t^s_i < T' and y_i == 1 (real) ---
    d1_mask = before_Tprime & (y_obs == 1)
    s1 = (event_ts.values[d1_mask] < np.datetime64(T_prime)).astype(int)

    # --- D0_iw: t^s_i < T' and (y_i == 0 or event_ts >= T') ---
    d0_mask = before_Tprime & (
        (y_obs == 0) | (event_ts.values >= np.datetime64(T_prime))
    )
    s0 = np.where(y_obs[d0_mask] == 0, 1, 0)

    return y_obs, e_real, e_prime, d1_mask, s1, d0_mask, s0


def fsiw(
    *,
    df,
    train_end,
    test_end=None,
    args=None,
    NUM_COLS,
    CAT_INT_COLS,
    TARGET="conversion",
    START_TS="timestamp_dt",
    EVENT_TS="conversion_timestamp_dt",
    seed: int = 0,
    hparams: dict | None = None,
    spark=None,
    outer: bool = False,
    calibrate: bool = True,
    platt_fit_slope: bool = PLATT_FIT_SLOPE,
    tau_days: int = DEFAULT_TAU_DAYS,
    max_delay_days=None,
    **kwargs,
):
    if args is None or getattr(args, "n_buckets", 1) != 1:
        raise ValueError(
            "FSIW is a single-head method (Yasui et al., 2020). "
            "Run with --n_buckets 1."
        )
    if getattr(args, "aux_target", None) is not None:
        raise ValueError(
            "FSIW has no auxiliary task in its original specification. "
            "Run with --aux_target unset (None)."
        )
    if max_delay_days is not None and tau_days >= max_delay_days:
        raise ValueError(
            f"tau_days={tau_days} must be strictly less than the dataset's "
            f"max delay window H ({max_delay_days} days) for the "
            f"counterfactual deadline to leave any usable elapsed-time "
            f"range. tau_days=5 is validated for Criteo only -- "
            f"reconsider before running on a shorter-horizon dataset."
        )
    if spark is None:
        raise ValueError(
            "fsiw() requires `spark` to materialize train_end/test_end "
            "(passed as Spark Column expressions in this pipeline) into "
            "concrete Python timestamps."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    g = torch.Generator()
    g.manual_seed(int(seed))

    hp = dict(
        batch_size      = 2048,
        learning_rate   = 1e-3,
        l2_weight_decay = 1e-5,
        deep_hidden     = (256, 128, 128),
        dropout_rate    = 0.1,
        epochs          = 15,
        emb_dim         = 4,
    )
    if hparams:
        hp.update(hparams)
    if isinstance(hp["deep_hidden"], str):
        hp["deep_hidden"] = tuple(
            int(x.strip()) for x in hp["deep_hidden"].strip("()").split(",") if x.strip()
        )

    # Materialize train_end/test_end to concrete timestamps
    ASOF_TRAIN = _as_ts(train_end)
    ASOF_TEST  = _as_ts(test_end)
    train_end_py = _materialize_ts(spark, ASOF_TRAIN)
    train_end_ts = pd.Timestamp(train_end_py)

    drop_cols = [c for c in ("bucket", "bucket_vec", "mask_train", "mask_eval", "aux_mask")
                 if c in df.columns]
    df = df.drop(*drop_cols)

    train_sdf = df.where(F.col(START_TS) < ASOF_TRAIN)

    if calibrate:
        ASOF_CAL = F.to_timestamp(F.date_add(F.date_trunc("day", ASOF_TRAIN), 1))
        cal_sdf  = df.where((F.col(START_TS) >= ASOF_TRAIN) & (F.col(START_TS) < ASOF_CAL))
        test_sdf = df.where((F.col(START_TS) >= ASOF_CAL) & (F.col(START_TS) < ASOF_TEST))
    else:
        test_sdf = df.where((F.col(START_TS) >= ASOF_TRAIN) & (F.col(START_TS) < ASOF_TEST))

    train_pdf = train_sdf.toPandas().reset_index(drop=True)
    test_pdf  = test_sdf.toPandas().reset_index(drop=True)
    if calibrate:
        cal_pdf = cal_sdf.toPandas().reset_index(drop=True)

    # build y_obs, e_real, e_prime, D1_iw, D0_iw
    y_obs, e_real, e_prime, d1_mask, s1, d0_mask, s0 = _build_fsiw_tables(
        train_pdf,
        START_TS=START_TS,
        EVENT_TS=EVENT_TS,
        TARGET=TARGET,
        train_end_ts=train_end_ts,
        tau_days=tau_days,
    )


    # Feature encoding
    if calibrate:
        *Xc_splits, cat_sizes = _encode_categoricals_train_test(
            train_pdf, cal_pdf, test_pdf, CAT_INT_COLS=CAT_INT_COLS
        )
        Xc_train, Xc_cal, Xc_test = Xc_splits
        Xn_train, Xn_cal, Xn_test = _prepare_numeric(
            train_pdf, cal_pdf, test_pdf, NUM_COLS=NUM_COLS
        )
    else:
        *Xc_splits, cat_sizes = _encode_categoricals_train_test(
            train_pdf, test_pdf, CAT_INT_COLS=CAT_INT_COLS
        )
        Xc_train, Xc_test = Xc_splits
        Xn_train, Xn_test = _prepare_numeric(
            train_pdf, test_pdf, NUM_COLS=NUM_COLS
        )

    X_base_train    = np.concatenate([Xc_train.astype(np.float64), Xn_train.astype(np.float64)], axis=1)
    cat_feature_idx = list(range(Xc_train.shape[1]))

    X_d1 = np.concatenate([X_base_train[d1_mask], e_prime[d1_mask].reshape(-1, 1)], axis=1)
    X_d0 = np.concatenate([X_base_train[d0_mask], e_prime[d0_mask].reshape(-1, 1)], axis=1)

    m1 = _fit_iw_estimator(X_d1, s1, lgbm_params=M1_LGBM_PARAMS, seed=seed, cat_feature_idx=cat_feature_idx)
    m0 = _fit_iw_estimator(X_d0, s0, lgbm_params=M0_LGBM_PARAMS, seed=seed, cat_feature_idx=cat_feature_idx)

    pos_idx = np.where(y_obs == 1)[0]
    neg_idx = np.where(y_obs == 0)[0]

    X_score_pos = np.concatenate([X_base_train[pos_idx], e_real[pos_idx].reshape(-1, 1)], axis=1)
    X_score_neg = np.concatenate([X_base_train[neg_idx], e_real[neg_idx].reshape(-1, 1)], axis=1)

    assert X_score_pos.shape[1] == X_d1.shape[1], (
        f"FSIW scoring/training feature width mismatch: "
        f"{X_score_pos.shape[1]} vs {X_d1.shape[1]}"
    )
    p1 = np.clip(m1.predict_proba(X_score_pos)[:, 1], IW_PROB_EPS, 1.0 - IW_PROB_EPS)
    p0 = np.clip(m0.predict_proba(X_score_neg)[:, 1], IW_PROB_EPS, 1.0 - IW_PROB_EPS)

    w_all = np.empty_like(e_real, dtype=np.float64)
    w_all[pos_idx] = 1.0 / p1
    w_all[neg_idx] = p0

    model = Model_Agnostic(
        cat_sizes    = cat_sizes,
        num_dim      = Xn_train.shape[1],
        emb_dim      = int(hp["emb_dim"]),
        deep_hidden  = tuple(hp["deep_hidden"]),
        dropout      = float(hp["dropout_rate"]),
        n_heads      = 1,
        use_aux      = False,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["l2_weight_decay"]),
    )

    train_loader = DataLoader(
        DS_Agnostic(Xc_train, Xn_train, y_obs.astype(np.float32), mask=w_all.astype(np.float32)),
        batch_size=int(hp["batch_size"]), shuffle=True, generator=g,
    )

    for _epoch in range(int(hp["epochs"])):
        model.train()
        for Xc, Xn, y_batch, w_batch in train_loader:
            Xc, Xn, y_batch, w_batch = (
                Xc.to(device), Xn.to(device), y_batch.to(device), w_batch.to(device)
            )
            logits = model(Xc, Xn)
            loss   = masked_bce_with_logits(logits, y_batch, w_batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    y_test = test_pdf[TARGET].astype(np.float32).values
    test_loader = DataLoader(
        DS_Agnostic(Xc_test, Xn_test, y_test),
        batch_size=int(hp["batch_size"]), shuffle=False,
    )

    if calibrate:
        y_cal = cal_pdf[TARGET].astype(np.float32).values
        cal_loader = DataLoader(
            DS_Agnostic(Xc_cal, Xn_cal, y_cal),
            batch_size=int(hp["batch_size"]), shuffle=False,
        )
        logits_cal, y_cal_labels   = _collect_logits(model, cal_loader,  device, mh_enabled=False, aux_enabled=False)
        logits_eval, y_eval_labels = _collect_logits(model, test_loader, device, mh_enabled=False, aux_enabled=False)
        p_all, y_all, platt_a, platt_b = _calibrate_single_head(
            logits_cal, y_cal_labels, logits_eval, y_eval_labels,
            fit_slope=platt_fit_slope,
        )
        print(f"  [FSIW] Platt a={platt_a:.4f}, b={platt_b:.4f}")
    else:
        logits_eval, y_eval_labels = _collect_logits(model, test_loader, device, mh_enabled=False, aux_enabled=False)
        y_all = y_eval_labels
        p_all = _sigmoid(logits_eval)

    y_all = y_all.astype(np.int32)
    p_all = p_all.astype(np.float64)

    if outer:
        plot_cvr_boxplot(
            y_all, p_all,
            title=f"FSIW CVR predictions ({'calibrated' if calibrate else 'uncalibrated'})",
        )
        print(f"  [FSIW] Eval set: {len(y_all)} rows, "
              f"positive rate: {float(y_all.mean()):.4f}, "
              f"mean predicted P: {float(p_all.mean()):.4f}, "
              f"D1_iw n={int(d1_mask.sum())}, D0_iw n={int(d0_mask.sum())}, "
              f"tau_days={tau_days}")

    gc.collect()
    return evaluation(y_all, p_all)
