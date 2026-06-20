from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import gc
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

from pyspark.sql import functions as F

from Time_Specific_Data_Processing import time_spec_data_preprocessing


# ============================================================
# Fold helpers (Spark)
# ============================================================

def day_floor(col_ts):
    return F.date_trunc("day", col_ts)

def add_days(col_date, n: int):
    return F.date_add(col_date, int(n))


# ============================================================
# HPO config
# ============================================================

@dataclass(frozen=True)
class HPTConfig:
    n_trials:            int  = 32
    # n_jobs is intentionally 1: Spark already parallelises internally,
    # and running Optuna trials in parallel would create competing Spark
    # jobs that corrupt each other's timing and resource allocation.
    n_jobs:              int  = 1
    timeout_s: Optional[int]  = None
    seed:                int  = 3
    direction:           str  = "maximize"   # maximise RCE

    enable_pruning:      bool = False
    pruner_warmup_steps: int  = 1

    # aggregation across inner folds
    agg:                 str  = "mean"       # "mean" or "median"


# ============================================================
# Search space
#
# Rules applied here:
#   1. Continuous/ordinal params (lr, wd, dropout, epochs) use
#      suggest_float / suggest_int so TPE can exploit ordering.
#   2. Truly discrete unordered params (batch_size, emb_dim,
#      deep_hidden) use suggest_categorical.
#   3. Every range includes the working default so TPE can
#      always rediscover it.
#   4. deep_hidden is a list of (width, width) tuples — both
#      layers are kept equal for simplicity; add asymmetric
#      options here if desired.
# ============================================================

BATCH_VALUES = [256, 512, 1024, 2048]          # default 2048 included
EMB_DIM_VALUES = [4, 8, 16, 32]                # default 8 included

# deep_hidden: list of tuples — stored/retrieved as tuples by suggest_categorical
# Optuna serialises these as strings internally; they are cast back to tuple
# in the objective before being passed to the model.
DEEP_HIDDEN_VALUES = [
    (256, 256),   
    (512, 512),
    (256, 128),
    (512, 256),
    (128, 128, 128),
    (512, 256, 128),
    (256, 128, 64),
]

# Continuous ranges — used with suggest_float / suggest_int
LR_LOW,      LR_HIGH      = 1e-4,  5e-3    
WD_LOW,      WD_HIGH      = 1e-7,  1e-3    
DROPOUT_LOW, DROPOUT_HIGH = 0.0,   0.5    
EPOCHS_LOW,  EPOCHS_HIGH  = 10,    25      

# Aux weight range (only used when aux task is enabled)
AUX_WEIGHT_LOW,  AUX_WEIGHT_HIGH = 0.05, 1.5   

# Warm-start bandwidth: if the warm-started trial's score is within
# BANDWIDTH_ABS of prev_best_score, run only FAST_EXTRA_TRIALS instead
# of the full budget (assumes the previous best params transfer well).
BANDWIDTH_ABS    = 1.0
FAST_EXTRA_TRIALS = 12


# ============================================================
# Main tuning function
# ============================================================

def tune_hyperparameters(
    df,
    *,
    outer_train_start_py,
    outer_train_end_py,
    model,
    seed,
    args,
    ifolds: Tuple[int, int, int],   # (itrain_days, val_days, istep_days)
    enqueue_params: Optional[Dict[str, Any]] = None,
    START_TS: str = "timestamp_dt",
    spark=None,
    NUM_COLS=None,
    CAT_INT_COLS=None,
    prev_best_score=None,
    cuts=None,
):
    cfg = HPTConfig()


    # Unpack fold config — all three values are in days
    itrain_days, val_days, istep_days = map(int, ifolds)

    df    = df.withColumn(START_TS, F.to_timestamp(F.col(START_TS)))
    spark = df.sparkSession

    outer_end_col = F.to_timestamp(F.lit(str(outer_train_end_py)))

    aux_enabled = getattr(args, "aux_target", None) is not None

    # Optuna study
    sampler = TPESampler(seed=cfg.seed, multivariate=True, group=True)
    pruner  = MedianPruner(n_warmup_steps=cfg.pruner_warmup_steps) if cfg.enable_pruning else None

    study = optuna.create_study(
        direction=cfg.direction,
        sampler=sampler,
        pruner=pruner,
        study_name="inner_cv_deepfm",
    )

    # Fold boundary helpers

    def _advance_cursor_py(cursor_py, step_days: int):
        """Advance the cursor by step_days calendar days (Spark-computed)."""
        row = spark.range(1).select(
            add_days(F.to_timestamp(F.lit(str(cursor_py))), step_days).alias("nx")
        ).first()
        return row["nx"]

    def _inner_fold_boundaries(cursor_py):
        """Return Spark Column expressions for the four fold boundaries."""
        icursor     = F.to_timestamp(F.lit(str(cursor_py)))
        itrain_start = day_floor(icursor)
        itrain_end   = add_days(itrain_start, itrain_days)
        ival_start   = itrain_end
        ival_end     = add_days(ival_start, val_days)
        return itrain_start, itrain_end, ival_start, ival_end

    def _should_stop_inner(ival_end_col) -> bool:
        """Stop when the validation window would exceed the outer training end."""
        return df.select((ival_end_col > outer_end_col).alias("stop")).first()["stop"]

    def _aggregate(scores: List[float]) -> float:
        if not scores:
            return float("-inf")
        return float(np.median(scores) if cfg.agg == "median" else np.mean(scores))

    # Objective

    def objective(trial: optuna.Trial) -> float:

        # -- Search space --
        # deep_hidden: suggest_categorical over explicit tuples.
        # Optuna stores the chosen value as a string internally; cast back to
        # tuple here so the model receives the correct type.
        deep_hidden_raw = trial.suggest_categorical(
            "deep_hidden", [str(v) for v in DEEP_HIDDEN_VALUES]
        )
        # Parse "(300, 300)" → (300, 300)
        deep_hidden = tuple(
            int(x.strip()) for x in deep_hidden_raw.strip("()").split(",")
        )

        hparams: Dict[str, Any] = {
            # Continuous — TPE exploits ordering via suggest_float/suggest_int
            "learning_rate":   trial.suggest_float(
                                   "learning_rate", LR_LOW, LR_HIGH, log=True),
            "l2_weight_decay": trial.suggest_float(
                                   "l2_weight_decay", WD_LOW, WD_HIGH, log=True),
            "dropout_rate":    trial.suggest_float(
                                   "dropout_rate", DROPOUT_LOW, DROPOUT_HIGH, step=0.05),
            "epochs":          trial.suggest_int(
                                   "epochs", EPOCHS_LOW, EPOCHS_HIGH),
            # Discrete unordered
            "batch_size":      trial.suggest_categorical("batch_size", BATCH_VALUES),
            "emb_dim":         trial.suggest_categorical("emb_dim", EMB_DIM_VALUES),
            # Architecture (tuple, stored as string, cast above)
            "deep_hidden":     deep_hidden,
        }

        # Aux weight only when aux task is active
        if aux_enabled:
            hparams["aux_weight"] = trial.suggest_float(
                "aux_weight", AUX_WEIGHT_LOW, AUX_WEIGHT_HIGH, log=True
            )

        # Inner fold loop
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
                calibrate = args.calibrate,
            )

            if "rce" not in metrics:
                raise KeyError(
                    f"model must return a dict containing 'rce'. Got: {list(metrics.keys())}"
                )

            scores.append(float(metrics["rce"]))

            # Report intermediate value for optional pruning
            trial.report(float(np.mean(scores)), step=ifold_id)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at inner fold {ifold_id}: running mean rce={np.mean(scores):.6f}"
                )

            cursor_py = _advance_cursor_py(cursor_py, istep_days)

            del df_ifold, metrics
            gc.collect()

        if not scores:
            # No fold completed — treat as a failed trial
            raise optuna.TrialPruned("No inner folds completed.")

        agg_score = _aggregate(scores)

        trial.set_user_attr("inner_folds_used", len(scores))
        trial.set_user_attr("inner_scores",     [float(s) for s in scores])
        trial.set_user_attr("inner_agg",         float(agg_score))
        trial.set_user_attr("aux_enabled",       bool(aux_enabled))

        return float(agg_score)

    # ============================================================
    # Warm-start: enqueue best params from the previous outer fold
    # ============================================================
    ran_warmstart   = False
    warmstart_value = None

    if enqueue_params:
        # Build the enqueue dict using the same param names as the search space.
        # deep_hidden must be the string representation so Optuna can match it
        # against the categorical choices.
        to_enqueue: Dict[str, Any] = {
            "learning_rate":   float(enqueue_params["learning_rate"]),
            "l2_weight_decay": float(enqueue_params["l2_weight_decay"]),
            "dropout_rate":    float(enqueue_params["dropout_rate"]),
            "epochs":          int(enqueue_params["epochs"]),
            "batch_size":      int(enqueue_params["batch_size"]),
            "emb_dim":         int(enqueue_params["emb_dim"]),
            # deep_hidden stored as string to match suggest_categorical choices
            "deep_hidden":     str(tuple(enqueue_params["deep_hidden"])),
        }

        if aux_enabled and "aux_weight" in enqueue_params:
            to_enqueue["aux_weight"] = float(enqueue_params["aux_weight"])

        study.enqueue_trial(to_enqueue)

        # Run the warm-start trial alone (n_jobs=1) so we can read its value
        # before deciding the remaining budget.
        study.optimize(
            objective,
            n_trials=1,
            n_jobs=1,
            timeout=cfg.timeout_s,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        ran_warmstart = True
        # The enqueued trial is always trial number 0 — index by number,
        # not by position, to be safe regardless of any internal ordering.
        t0 = next(t for t in study.trials if t.number == 0)
        if t0.state.name == "COMPLETE" and t0.value is not None:
            warmstart_value = float(t0.value)

    # ============================================================
    # Decide remaining trial budget
    # ============================================================
    if ran_warmstart and warmstart_value is not None and prev_best_score is not None:
        close = abs(warmstart_value - float(prev_best_score)) <= BANDWIDTH_ABS
        print(f"  Warm-start score: {warmstart_value:.6f} | "
              f"prev best: {prev_best_score:.6f} | "
              f"close={close}")
        remaining = FAST_EXTRA_TRIALS if close else max(0, cfg.n_trials - 1)
    else:
        remaining = cfg.n_trials

    # ============================================================
    # Main optimisation run
    # ============================================================
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=cfg.n_jobs,        # always 1 — see HPTConfig note above
            timeout=cfg.timeout_s,
            gc_after_trial=True,
            show_progress_bar=True,
        )

    # ============================================================
    # Extract best params
    # ============================================================
    best_raw    = dict(study.best_params)
    best_value  = float(study.best_value)

    # Convert deep_hidden back from string to tuple for the model
    if "deep_hidden" in best_raw:
        best_raw["deep_hidden"] = tuple(
            int(x.strip()) for x in best_raw["deep_hidden"].strip("()").split(",")
        )

    return best_raw, best_value, study
