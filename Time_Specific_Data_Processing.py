from pyspark.sql import functions as F

def print_mask_stats(df, START_TS, train_end, MASK_COL="mask", AUX_MASK_COL="aux_mask", aux_target=None):
    n = df.count()
    conversions = df.agg(F.sum("conversion")).first()[0]
    not_all_ones = F.expr(f"NOT aggregate({MASK_COL}, true, (acc, x) -> acc AND x = 1)")
    df = df.filter(F.col(START_TS) < train_end)
    row = (
        df.agg(
            (F.count_if(not_all_ones) / F.count(F.lit(1)) * 100).alias("pct_primary_masked"),
            (F.count(F.lit(1))).alias("n_rows"),
        )
        .first()
    )

    msg = f"Primary masked (mask not all-ones): {row['pct_primary_masked']:.2f}% (n = {n}; n_train={row['n_rows']}; n_test = {n - row['n_rows']})"

    
    
    msg += f"| Conversions: {conversions:.2f}"

    print(msg)

# Helper Functions

def unknown_int_cats_by_join(df, supervised_train_rows, cat_cols):
    for c in cat_cols:
        # allowed values seen in supervised TRAIN
        allowed_df = supervised_train_rows.select(F.col(c).alias(c)).where(F.col(c).isNotNull()).distinct()

        # left join to mark allowed rows
        df = (
            df.join(allowed_df.withColumn("_ok", F.lit(1)), on=c, how="left")
              .withColumn(c, F.when(F.col("_ok") == 1, F.col(c)).otherwise(F.lit(-1)))
              .drop("_ok")
        )
    return df

def unknown_country_by_join(df, supervised_train_rows, country_col="country", unknown_value="__MISSING__"):
    allowed_c = (
        supervised_train_rows
        .select(F.col(country_col).alias(country_col))
        .where(F.col(country_col).isNotNull())
        .distinct()
    )

    df = (
        df.join(allowed_c.withColumn("_ok", F.lit(1)), on=country_col, how="left")
          .withColumn(country_col, F.when(F.col("_ok") == 1, F.col(country_col)).otherwise(F.lit(unknown_value)))
          .drop("_ok")
    )
    return df

def time_spec_data_preprocessing(
    df,
    train_end,          
    test_end, 
    eval_end,        
    cuts,               
    aux_target=None,
    START_TS="timestamp_dt",
    EVENT_TS="conversion_timestamp_dt",
    TARGET="conversion",
    MASK_TRAIN_COL="mask_train", 
    MASK_EVAL_COL="mask_eval",
    AUX_MASK_COL="aux_mask",
    CAT_INT_COLS=None,
    n_heads=1           
):
    # sanity checks
    if cuts is None or len(cuts) == 0:
        raise ValueError("cuts must be provided as a non-empty list of bucket cutoffs (seconds).")

    CAT_INT_COLS = CAT_INT_COLS or []

    # Ensure timestamps exist as timestamps
    df = (
        df.withColumn(START_TS, F.to_timestamp(F.col(START_TS)))
          .withColumn(EVENT_TS, F.to_timestamp(F.col(EVENT_TS)))
    )

    cuts_arr = F.array(*[F.lit(float(c)) for c in cuts])

    # Train mask: what is known at train_end
    age_train_sec = (
        F.unix_timestamp(F.lit(train_end)) - F.unix_timestamp(F.col(START_TS))
    ).cast("double")

    event_known_train = (
        (F.col(TARGET) == 1) &
        F.col(EVENT_TS).isNotNull() &
        (F.col(EVENT_TS) <= F.lit(train_end))
    )

    mask_train = F.transform(
        cuts_arr,
        lambda c: F.when(event_known_train, F.lit(1))
                .otherwise((age_train_sec >= c).cast("int"))
    )
    
    mask_eval = F.array(*[F.lit(1) for _ in cuts])

    df = (
        df.withColumn(MASK_TRAIN_COL, mask_train)
        .withColumn(MASK_EVAL_COL, mask_eval)
    )
    
    if aux_target is not None:
        df = df.withColumn(AUX_MASK_COL, F.lit(1))

    # "unknown" categories not seen in *supervised train* rows are set to 0

    # Primary supervised if at least one bucket is mature
    if n_heads > 1:
        supervised_train_rows = df.filter(
            (F.col(START_TS) < train_end) &
            (F.aggregate(F.col(MASK_TRAIN_COL), F.lit(0), lambda acc, x: acc + x) > 0)
        )
    else:
        supervised_train_rows = df.filter(
            (F.col(START_TS) < train_end) &
            (F.element_at(F.col(MASK_TRAIN_COL), -1) == 1)
        )

    df = unknown_int_cats_by_join(df, supervised_train_rows, CAT_INT_COLS)
   
    print_mask_stats(
        df,
        START_TS=START_TS,
        train_end=train_end,
        MASK_COL=MASK_TRAIN_COL,
        AUX_MASK_COL=AUX_MASK_COL,
        aux_target=aux_target,
    )
    return df
