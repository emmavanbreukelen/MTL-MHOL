from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyspark.sql import functions as F

from Time_Specific_Data_Processing import time_spec_data_preprocessing



# Works in a similar manner as HPTuning.py, but adjusted to work for RF

def day_floor(col_ts):
    return F.date_trunc("day", col_ts)


def add_days(col_date, n: int):
    return F.date_add(col_date, int(n))


# ============================================================
# Default search space
# ============================================================

DEFAULT_MIN_SAMPLES_LEAF_GRID = [10, 50, 100, 150, 200, 250]
DEFAULT_RANDOM_LOW  = 50
DEFAULT_RANDOM_HIGH = 300


# ============================================================
# Lightweight RF tuner
# ============================================================

def tune_hyperparameters_rf(
    df,
    *,
    outer_train_start_py,
    outer_train_end_py,
    model,
    seed,
    args,
    ifolds: Tuple[int, int, int],          # (itrain_days, val_days, istep_days)
    enqueue_params: Optional[Dict[str, Any]] = None,
    prev_best_score: Optional[float] = None,
    START_TS: str = "timestamp_dt",
    spark=None,
    NUM_COLS=None,
    CAT_INT_COLS=None,
    cuts=None,
    search_mode: str = "grid",             # "grid" or "random"
    min_samples_leaf_grid: Optional[List[int]] = None,
    n_random_trials: int = 8,
    random_low: int = DEFAULT_RANDOM_LOW,
    random_high: int = DEFAULT_RANDOM_HIGH,
    agg: str = "mean",                     # "mean" or "median"
):

    itrain_days, val_days, istep_days = map(int, ifolds)

    df = df.withColumn(START_TS, F.to_timestamp(F.col(START_TS)))
    spark = df.sparkSession

    outer_end_col = F.to_timestamp(F.lit(str(outer_train_end_py)))

    # Build candidate list
    if search_mode == "grid":
        candidates = list(min_samples_leaf_grid or DEFAULT_MIN_SAMPLES_LEAF_GRID)
    elif search_mode == "random":
        rng = np.random.RandomState(seed)
        candidates = sorted(
            int(v) for v in rng.randint(random_low, random_high + 1, size=n_random_trials)
        )
    else:
        raise ValueError(f"Unknown search_mode '{search_mode}'. Choose 'grid' or 'random'.")

    # Always (re-)evaluate the previously-best value too, for warm-start continuity across outer folds
    if enqueue_params and "min_samples_leaf" in enqueue_params:
        warm_val = int(enqueue_params["min_samples_leaf"])
        if warm_val not in candidates:
            candidates = [warm_val] + candidates

    # Fold boundary helpers (mirrors HPTuning.py)

    def _advance_cursor_py(cursor_py, step_days: int):
        row = spark.range(1).select(
            add_days(F.to_timestamp(F.lit(str(cursor_py))), step_days).alias("nx")
        ).first()
        return row["nx"]

    def _inner_fold_boundaries(cursor_py):
        icursor      = F.to_timestamp(F.lit(str(cursor_py)))
        itrain_start = day_floor(icursor)
        itrain_end   = add_days(itrain_start, itrain_days)
        ival_start   = itrain_end
        ival_end     = add_days(ival_start, val_days)
        return itrain_start, itrain_end, ival_start, ival_end

    def _should_stop_inner(ival_end_col) -> bool:
        return df.select((ival_end_col > outer_end_col).alias("stop")).first()["stop"]

    def _aggregate(scores: List[float]) -> float:
        if not scores:
            return float("-inf")
        return float(np.median(scores) if agg == "median" else np.mean(scores))

    # Evaluate a single candidate across all inner folds

    def _evaluate_candidate(min_samples_leaf: int) -> Tuple[float, List[float]]:
        hparams = {"min_samples_leaf": int(min_samples_leaf)}

        scores    = []
        cursor_py = outer_train_start_py
        ifold_id  = 0

        while True:
            ifold_id += 1
            itrain_start, itrain_end, ival_start, ival_end = _inner_fold_boundaries(cursor_py)

            if _should_stop_inner(ival_end):
                break

            df_ifold = df.filter(
                (F.col(START_TS) >= itrain_start) &
                (F.col(START_TS) <  ival_end)
            )
            df_ifold = time_spec_data_preprocessing(
                df=df_ifold,
                train_end=itrain_end,
                test_end=ival_end,
                eval_end=outer_train_end_py,
                aux_target=args.aux_target,
                cuts=cuts,
                CAT_INT_COLS=CAT_INT_COLS,
            )

            metrics = model(
                df=df_ifold,
                hparams=hparams,
                train_end=itrain_end,
                test_end=ival_end,
                seed=seed,
                args=args,
                spark=spark,
                NUM_COLS=NUM_COLS,
                CAT_INT_COLS=CAT_INT_COLS,
                calibrate=args.calibrate,
            )

            if "rce" not in metrics:
                raise KeyError(
                    f"model must return a dict containing 'rce'. Got: {list(metrics.keys())}"
                )

            scores.append(float(metrics["rce"]))

            cursor_py = _advance_cursor_py(cursor_py, istep_days)

            del df_ifold, metrics
            gc.collect()

        return _aggregate(scores), scores

    # Run the search
    results: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] = {}
    best_score  = float("-inf")

    for cand in candidates:
        agg_score, fold_scores = _evaluate_candidate(cand)
        results.append({
            "min_samples_leaf": int(cand),
            "agg_score": agg_score,
            "fold_scores": fold_scores,
        })

        if agg_score > best_score:
            best_score  = agg_score
            best_params = {"min_samples_leaf": int(cand)}

    if not results:
        raise RuntimeError(
            "[HPTuner_rf] No inner folds were evaluated — check the --ifolds settings "
            "relative to the outer training window."
        )

    return best_params, best_score, results
