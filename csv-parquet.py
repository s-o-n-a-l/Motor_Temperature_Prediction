import polars as pl
from pathlib import Path

CSV_FILES = [Path("/home/arvr-lab-16/Downloads/motor temperature /221031a4-ead8-4f43-94de-d40a7a26e3b3.csv")]#month 1,2,4,5,3,6

for csv_path in CSV_FILES:
    parquet_path = csv_path.with_suffix(".parquet")
    print(f"Processing: {csv_path.name}->{parquet_path.name}")

    (
        pl.scan_csv(csv_path, infer_schema_length = 10000)
        .sink_parquet(parquet_path, compression="snappy")
    )
print("all csv->parquet files")   