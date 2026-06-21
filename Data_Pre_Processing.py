import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import broadcast

from pyspark.sql.functions import (
    col, when, hour, dayofmonth, month, date_format
)

from pyspark.sql import functions as F
import numpy as np


# Helper functions: add delays
def add_delay_buckets(df, time_col, n_bins=5):
    values = (
        df
        .filter(
            F.col(time_col).isNotNull()
        )
        .select(time_col)
        .rdd.map(lambda x: x[0])  # convert spark rows to a list of numeric values
        .collect()
    )
    H = max(values)

    # for n_bins = 4: fixed bin edges (seconds): 0-1D, 1-4D, 4-8D, 8-12D (H=12D)
    if n_bins == 4:
        bins = [0.0, 86400, 345600, 691200, float(H)]
        n_bins = len(bins) - 1 

    # print bins to check
    print("Delay bins (in seconds):")
    for i in range(len(bins) - 1):
        right = "]" if i == len(bins) - 2 else ")"
        print(f"  Bucket {i+1}: [{bins[i]}, {bins[i+1]}{right}")

    # assign exactly one bucket per row:
    # lower bound inclusive, upper bound exclusive, last bucket upper bound inclusive
    bucket_col = None

    for i in range(len(bins) - 1):
        lower = float(bins[i])
        upper = float(bins[i + 1])

        cond = (
            (F.col(time_col) >= lower) &
            (
                F.col(time_col) <= upper
                if i == len(bins) - 2
                else F.col(time_col) < upper
            )
        )

        bucket_value = F.struct(
            F.lit(i + 1).alias("bucket_id"),
            F.lit(lower).alias("lower_bound"),
            F.lit(upper).alias("upper_bound")
        )

        bucket_col = (
            F.when(cond, bucket_value)
            if bucket_col is None
            else bucket_col.when(cond, bucket_value)
        )

    # add the triple in bucket column with (bucketnr, lb, ub)
    df = df.withColumn("bucket", bucket_col)

    # one-hot vector column of length n_bins
    df = df.withColumn(
        "bucket_vec",
        F.when(
            F.col("bucket.bucket_id").isNotNull(),
            F.transform(
                F.sequence(F.lit(1), F.lit(n_bins)),
                lambda i: (i == F.col("bucket.bucket_id")).cast("int")
            )
        ).otherwise(F.array_repeat(F.lit(0), n_bins))
    )

    return df


# Main function: create dataframe and add features
df_criteo = spark.table("RAW_DATA_PATH") # Load the raw data file

# Add conversion delay
df_criteo = df_criteo.withColumn(
    "conversion_delay",
    F.when(
        F.col("conversion_timestamp") != -1,
        F.col("conversion_timestamp").cast("long") -
        F.col("timestamp").cast("long")
    ).otherwise(F.lit(-1))
)

# Convert timestamps
start_ts = "2025-01-01 00:00:00"  # simple format
start_unix = F.unix_timestamp(F.lit(start_ts), "yyyy-MM-dd HH:mm:ss")

df_criteo = (
    df_criteo
    .withColumn(
        "timestamp_dt",
        F.from_unixtime(start_unix + F.col("timestamp")).cast("timestamp")
    )
    .withColumn(
        "conversion_timestamp_dt",
        F.when(
            F.col("conversion_timestamp") != -1,
            F.from_unixtime(start_unix + F.col("conversion_timestamp")).cast("timestamp")
        ).otherwise(None)
    )
)

# Create features
w_prev = (
    Window.partitionBy("uid")
    .orderBy("timestamp_dt")
    .rowsBetween(Window.unboundedPreceding, -1)
)
w_ordered = Window.partitionBy("uid").orderBy("timestamp_dt")

df_final = (
    df_criteo
    # ---- number of previous conversions ----
    .withColumn(
        "n_previous_conversions",
        F.coalesce(F.sum("conversion").over(w_prev), F.lit(0))
    )
    # ---- time since last impression ----
    .withColumn(
        "prev_timestamp",
        F.lag("timestamp_dt").over(w_ordered)
    ).withColumn(
        "time_since_last_impression",
        F.col("timestamp_dt").cast("long") - F.col("prev_timestamp").cast("long")
    ).drop("prev_timestamp")

    # ---- impression number ----
    .withColumn(
        "impression_number",
        F.row_number().over(w_ordered)
    )

    # ---- row identifier ----
    .withColumn("impression_id", F.expr("uuid()"))
)

# Write to table
df_final.write \
  .format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("INTERMEDIATE_DATA_PATH") # Intermediate step (after first preprocessing step)


# Downsampling and and implement cutoff
from pyspark.sql import functions as F

# Original written table
table_name = "INTERMEDIATE_DATA_PATH" # Load the data file you obtain after the first preprocessing steps (intermediate data path)

# Load table
df_full = spark.table(table_name)

BUCKET_COL = "bucket.bucket_id"  
CUTOFF = 1036800    #12D cutoff for conversions  
NEG_FRACTION = 0.75 # Downsample the negative class similarly
SEED = 3

# 1) Remove conversions in last bucket (everything after 12D)
df_downsampled = df_full.filter(
    ~(
        (F.col("conversion") == 1) &
        (F.col("conversion_delay") > CUTOFF)
    )
)

# 2) Downsample non-converters by 25%, keep all remaining converters
df_downsampled = df_downsampled.filter(F.col("conversion") == 1).unionByName(
    df_downsampled.filter(F.col("conversion") == 0).sample(withReplacement=False, fraction=NEG_FRACTION, seed=SEED)
)

df_downsampled = df_downsampled.sample(withReplacement=False, fraction=0.02, seed=3)
# Downsample
# df_final = df_criteo.sample(withReplacement=False, fraction=0.02, seed=3)

# Add buckets based on function in previous bloc
df_downsampled = add_delay_buckets(df=df_downsampled, time_col="conversion_delay", n_bins=4)

# Write to table
df_downsampled.write \
  .format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("PREPROCESSED_DATA_PATH") # Obtain the preprocessed data which can be used in the full model pipeline
