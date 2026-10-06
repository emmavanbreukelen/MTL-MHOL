from pyspark.sql import functions as F
import numpy as np

def pre_process_data(
    df,
    *,
    CAT_INT_COLS=None,
    NUM_COLS=None,
    aux_target=None,
    n_buckets=4,
    HEAD_COL=["bucket_vec"]
):

    START_TS   = "timestamp_dt"
    EVENT_TS   = "conversion_timestamp_dt"
    SESSION_ID = "impression_id"

    TARGET     = "conversion"

    # Define AUX_TARGET safely
    AUX_TARGET = aux_target

    # bucket cutoffs -> H
    bounds = (
        df
        .select(
            F.col("bucket.bucket_id").alias("bucket_id"),
            F.col("bucket.upper_bound").alias("upper_bound")
        )
        .where(F.col("bucket_id").isNotNull() & F.col("upper_bound").isNotNull())
        .distinct()
        .orderBy("bucket_id")
        .collect()
    )

    CUTS = np.array([float(r["upper_bound"]) for r in bounds], dtype=np.float32)

    if len(CUTS) != n_buckets:
        raise ValueError(f"Expected {n_buckets} bucket cutoffs from df.bucket, got {len(CUTS)}: {CUTS}")

    H = float(CUTS[-1])

    # required columns
    cols_present = set(df.columns)

    required_variables = [TARGET, START_TS, SESSION_ID, EVENT_TS] + HEAD_COL
    if AUX_TARGET is not None:
        required_variables += [AUX_TARGET]

    miss = [c for c in required_variables if c not in cols_present]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # Keep only those present
    CAT_INT_COLS = [c for c in CAT_INT_COLS if c in cols_present]
    NUM_COLS     = [c for c in NUM_COLS     if c in cols_present]


    # select columns
    select_cols = (
        [TARGET, START_TS, SESSION_ID, EVENT_TS]
        + ([AUX_TARGET] if AUX_TARGET is not None else [])
        + CAT_INT_COLS
        + NUM_COLS
        + HEAD_COL
    )

    df = (
        df.select(*select_cols)
          .withColumn(TARGET, F.col(TARGET).cast("int"))
          .withColumn(START_TS, F.to_timestamp(F.col(START_TS)))
          .withColumn(EVENT_TS, F.to_timestamp(F.col(EVENT_TS)))
    )

    if AUX_TARGET is not None:
        df = (
            df.withColumn(AUX_TARGET, F.col(AUX_TARGET).cast("int"))
        )

    # fill categoricals, keep numerics, fill bucket_vec with zeros if null
    zero_vec = F.array_repeat(F.lit(0), n_buckets)

    for hc in HEAD_COL:
        # keep as array; if null -> [0,0,0,0]
        df = df.withColumn(hc, F.coalesce(F.col(hc), zero_vec))

    for c in CAT_INT_COLS:
        df = df.withColumn(c, F.coalesce(F.col(c).cast("int"), F.lit(0)))

    for c in NUM_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))


    return df, H, CUTS
