""" Interacts with BigQuery Warehouse to save stockdata"""

import os
import json
from google.cloud import bigquery
from dotenv import load_dotenv
from prefect import task

load_dotenv()

BIGQUERY_CREDENTIALS_PATH = os.getenv("BIGQUERY_CREDENTIALS_PATH")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = "stock_data"
TABLE_ID = "aapl_stock_price"

client = bigquery.Client.from_service_account_json(BIGQUERY_CREDENTIALS_PATH)

@task
def insert_into_bigquery(data : dict):
    """
    Inserts stock data into BigQuery Data Warehouse
    """

    time_series = data.get("Time Series (1min)", {})
    rows_to_insert = []

    for ts, metrics in time_series.items():
        row = {
            "timestamp": ts,
            "open": float(metrics["1. open"]),
            "high": float(metrics["2. high"]),
            "low": float(metrics["3. low"]),
            "close": float(metrics["4. close"]),
            "volume": int(metrics["5. volume"]),
        }

        rows_to_insert.append(row)

    if not rows_to_insert:
        print("No valid data to insert")
        return

    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
    table = client.get_table(table_ref)

    errors = client.insert_rows_json(table, rows_to_insert)

    if errors == []: # No errors
        print(f"Inserted {len(rows_to_insert)} rows into {DATASET_ID}.{TABLE_ID}.")
    else:
        print(f"Encountered errors while inserting rows: {errors}")

