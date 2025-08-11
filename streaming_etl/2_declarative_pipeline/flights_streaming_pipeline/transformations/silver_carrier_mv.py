# # This file defines transformations for silver zone.
# # Keep this file small (<5 tables)
# # Use "+ Add" in the file browser to add more.

import dlt
from datetime import datetime

from pyspark.sql.functions import col, lit, expr, coalesce, row_number, when
from pyspark.sql.window import Window
import pandas as pd
import os

@dlt.table(name = "dlt_py_carrier_silver")
def carrier_silver():
    current_dir = os.getcwd()
    csv_full_path = f"{current_dir}/T_CARRIER_DECODE.csv"
    reference_data = pd.read_csv(csv_full_path, header=0)
    carrier_reference_df = spark.createDataFrame(reference_data)
    return carrier_reference_df
