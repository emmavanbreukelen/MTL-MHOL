
import argparse
import random
import numpy as np
import pandas as pd
import torch                           
import time
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# Import your preprocessing
from General_Data_Processing import pre_process_data
from Time_Specific_Data_Processing import time_spec_data_preprocessing
from Logistic_regression import lr
from Flag_models import flag_model
from DeepFM import deep_fm
from HPTuning import tune_hyperparameters
from HPTuner_RF import tune_hyperparameters_rf
from random_forest import rf

# -----------------------------
# Fold utilities
# -----------------------------

def day_floor(col_ts):
    return F.date_trunc("day", col_ts)

def add_days(col_date, n):
    return F.date_add(col_date, int(n))

MODEL_REGISTRY = {
    "lr": lr,
    "rf": rf,
    "stem_model": stem_model,
    "flag_model": flag_model,
    "deep_fm": deep_fm
}

# Added eval_ends
def run_outer_fold(df, fold_id, train_start, train_end, test_start, test_end, eval_end, args, cuts, NUM_COLS, CAT_INT_COLS, seed, spark, prev_best_params=None, prev_best_score=None, max_delay_days=None):
    START_TS = args.start_ts  

    dates = (
        spark.range(1)
        .select(
            train_start.alias("train_start"),
            train_end.alias("train_end"),
            test_start.alias("test_start"),
            test_end.alias("test_end"),
        )
        .first()
    )

    print(
        f"[Fold {fold_id}] "
        f"train: [{dates.train_start}, {dates.train_end}) | "
        f"test: [{dates.test_start}, {dates.test_end})"
    )

    df_fold = df

    # Keep only observations from first train month up to (but not including) month after last test month
    df_fold = df_fold.filter(
        (F.col(START_TS) >= train_start) &
        (F.col(START_TS) <  test_end)
    )

    df_fold = time_spec_data_preprocessing(
        df=df_fold,
        train_end=train_end,
        test_end=test_end,
        eval_end=eval_end,
        aux_target=args.aux_target,
        cuts=cuts,
        CAT_INT_COLS=CAT_INT_COLS
    )   
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Choose from {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[args.model]

    best_params = {}
    best_score = None
    if args.hptuning:
        df_ifold = df_fold.filter((F.col(START_TS) <  train_end))

        train_start_py = spark.range(1).select(train_start.alias("ts")).first()["ts"]
        train_end_py   = spark.range(1).select(train_end.alias("ts")).first()["ts"]

        if args.model == "rf":
            best_params, best_score, study = tune_hyperparameters_rf(
                df=df_ifold,
                outer_train_start_py=train_start_py,
                outer_train_end_py=train_end_py,
                ifolds=args.ifolds,
                model=model,
                spark=spark,
                seed=seed,
                args=args,
                NUM_COLS=NUM_COLS,
                CAT_INT_COLS=CAT_INT_COLS,
                cuts=cuts,
                enqueue_params=prev_best_params,
                prev_best_score=prev_best_score,
            )
        else:
            best_params, best_score, study = tune_hyperparameters(
                df=df_ifold,
                outer_train_start_py=train_start_py,
                outer_train_end_py=train_end_py,
                ifolds=args.ifolds,
                model=model,
                spark=spark,
                seed=seed,
                args=args,
                NUM_COLS = NUM_COLS,
                CAT_INT_COLS = CAT_INT_COLS,
                cuts=cuts,
                enqueue_params=prev_best_params,
                prev_best_score=prev_best_score,
            )

    eval_metrics = model(df = df_fold, 
                         train_end=train_end,
                         test_end=test_end, 
                         args=args,
                         seed=seed, 
                         hparams=best_params,
                         NUM_COLS = NUM_COLS,
                         CAT_INT_COLS = CAT_INT_COLS,
                         spark=spark,
                         outer=True,
                         calibrate=args.calibrate,
                         max_delay_days=max_delay_days,
                        )
        
    return eval_metrics, best_params, best_score

def average_evaluation_metrics(eval_metrics_folds: dict) -> pd.DataFrame:
    if not eval_metrics_folds:
        return pd.DataFrame()

    # Convert dict-of-dicts to DataFrame
    rows = []
    for fold_id, metrics in eval_metrics_folds.items():
        row = {"fold_id": fold_id}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("fold_id")

    # Average numeric columns
    metric_cols = [c for c in df.columns if c != "fold_id"]
    avg_row = {"fold_id": "AVG"}
    for c in metric_cols:
        avg_row[c] = float(np.nanmean(df[c].astype(float).values))

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df
   
# -----------------------------
# Main
# -----------------------------

def main(args):
    print("START")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    spark = SparkSession.builder.getOrCreate()
    
    # c = args.customer

    df_raw = spark.table(f"DATA_PATH") # Load data file
 
    CAT_INT_COLS = [
        "campaign",
        "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9",
    ]

    NUM_COLS = [
        "time_since_last_impression",
        "time_since_last_click",
        "n_previous_conversions",
        "impression_number",
    ]

   
    if args.start_ts != "timestamp_dt":
        df_raw = df_raw.withColumnRenamed("timestamp_dt", args.start_ts) 
    else:
        pass

    if args.event_ts != "conversion_timestamp_dt":
        df_raw = df_raw.withColumnRenamed("conversion_timestamp_dt", args.event_ts)
  
    df_clean, H, CUTS = pre_process_data(
        df=df_raw,
        CAT_INT_COLS = CAT_INT_COLS,
        NUM_COLS = NUM_COLS,
        aux_target=args.aux_target,
        # n_buckets=args.n_buckets,
        HEAD_COL = ["bucket_vec"]
    )
    print(df_clean.columns)

    # Determine last_impression and a fold start anchor
    EVENT_TS = args.event_ts
    START_TS = args.start_ts

    eval_end = df_clean.agg(F.max(F.col(EVENT_TS)).alias("mx")).first()["mx"]
    if eval_end is None:
        raise ValueError("eval_end is None (no conversion timestamps found).") 
    max_delay_days = int(np.ceil(H / 86400.0))
    last_eligible_test_start = df_clean.select(
        F.date_sub(F.to_date(F.lit(eval_end)), max_delay_days).alias("last_day")
    ).first()["last_day"]    

    last_impression = df_clean.agg(F.max(F.col(START_TS)).alias("mx")).first()["mx"]
    if last_impression is None:
        raise ValueError("last_impression is None (no conversion event timestamps).")

    first_session = df_clean.agg(F.min(F.col(START_TS)).alias("mn")).first()["mn"]
    if first_session is None:
        raise ValueError("first_session is None (no start timestamps).")
    
    cursor0 = first_session
    train_days, test_days, step_days = args.ofolds  

    print("Outer fold params (days):", dict(train_days=train_days, test_days=test_days,step_days=step_days)) 
    print("Data time span (impressions):", first_session, "→", df_clean.agg(F.max(F.col(START_TS))).first()[0])
    # print("Data time span (conversions):", df_clean.agg(F.min(F.col(EVENT_TS))).first()[0], "→", last_impression)
    print("Data time span (conversions):", df_clean.agg(F.min(F.col(EVENT_TS))).first()[0], "→", eval_end)

    # Iterate folds (framework only)
    fold_id = 0
    eval_metrics_folds={}
    best_params_folds={}  
    # for first fold, there are no previous best params so TPE just starts randomly
    prev_best_params = None
    prev_best_score = None
    while True:
        fold_id += 1

        # Define fold boundaries day-aligned (not month-aligned)
        cursor = F.to_timestamp(F.lit(str(cursor0)))
        train_start = day_floor(cursor)                      
        train_end   = add_days(train_start, train_days)      
        test_start  = train_end
        test_end    = add_days(test_start, test_days)
 
        last_impression_day = F.to_date(F.lit(last_impression))

        should_stop = spark.range(1).select(
            (test_end > last_impression_day).alias("stop")
        ).first()["stop"]

        if should_stop:
            break
        
        eval_metrics, best_params, best_score = run_outer_fold(
            df=df_clean,
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            eval_end=eval_end,
            args=args,
            cuts=CUTS,
            NUM_COLS=NUM_COLS,
            CAT_INT_COLS=CAT_INT_COLS,
            seed=seed,
            spark=spark,
            prev_best_params=prev_best_params,
            prev_best_score=prev_best_score,
            max_delay_days = max_delay_days,
        )        

        eval_metrics_folds[fold_id] = eval_metrics
        best_params_folds[fold_id] = best_params

        print(f"Fold {fold_id} evaluation: {eval_metrics}")
        print(f"Fold {fold_id} best_params: {best_params}")

        # update so next fold can use the best params of previous folds
        prev_best_params = best_params
        prev_best_score = best_score

        # Advance cursor0 by step_days for the next fold
        next_cursor0 = (
            df_clean
            .select(add_days(F.to_timestamp(F.lit(str(cursor0))), step_days).alias("nx"))
            .first()["nx"]
        )
        cursor0 = next_cursor0

    df_clean.unpersist()
    summary_df = average_evaluation_metrics(eval_metrics_folds)
    print(summary_df.to_string(index=False))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux_target", type=str, default=None, help="Enable auxiliary target in preprocessing: Binary: click; No AUX: None ")
    parser.add_argument("--hptuning", action="store_true", default=True, help="Enable hyperparameter tuning")
    parser.add_argument("--calibrate", action="store_true", default=True, help="Enable calibration")
    parser.add_argument("--n_buckets", type=int, default=1, help="4: multi-head, 1: single-head")
    parser.add_argument("--model", type=str, default="rf", help="Select model: [rf: random_forest, lr: Logistic_regression, stem_model: Multitask_Stem, flag_model: Flag_models, deep_fm: DeepFM]")

    # Column names (keep configurable)
    parser.add_argument("--start_ts", type=str, default="timestamp_dt")
    parser.add_argument("--event_ts", type=str, default="conversion_timestamp_dt")

    # Outer fold params
    # [train_days, test_days, step_days]
    parser.add_argument(
        "--ofolds",
        nargs=3,
        type=int,
        default=[25, 1, 1], # CAN CHANGE THIS
        metavar=("TRAIN_D", "TEST_D", "STEP_D"),
        help="Outer folds in DAYS: TRAIN_D TEST_D STEP_D. Example: --ofolds 24 6 1"
)

    # Inner fold params
    # [train_days, test_days, step_days]
    parser.add_argument(
        "--ifolds",
        nargs=3,
        type=int,
        default=[23, 1, 1],  # CAN CHANGE THIS
        metavar=("TRAIN_D", "VAL_D", "STEP_D"),
        help="Inner folds in DAYS: TRAIN_D VAL_D STEP_D. Example: --ifolds 19 5 1"
)

    args, _ = parser.parse_known_args()

    start = time.perf_counter()
    main(args)
    end = time.perf_counter()
    elapsed = end - start
    print(f"\n===== TOTAL WALL-CLOCK RUNTIME: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds) =====")
