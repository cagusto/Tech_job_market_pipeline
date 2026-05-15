"""DuckDB load module for processed job market data."""

from __future__ import annotations

import logging
import os

import duckdb
import pandas as pd


logger = logging.getLogger(__name__)

PROCESSED_PARQUET_PATH = os.path.join("data", "processed", "cleaned_jobs.parquet")
DATABASE_PATH = os.path.join("data", "processed", "tech_jobs.duckdb")
JOBS_TABLE_NAME = "jobs"


def load_data_to_duckdb() -> int:
    """Load cleaned jobs from Parquet into a local DuckDB database.

    Reads `data/processed/cleaned_jobs.parquet`, connects to (or creates)
    `data/processed/tech_jobs.duckdb`, and replaces the `jobs` table with the
  latest dataset. This full-refresh strategy keeps the load step idempotent.

    Returns:
        Number of rows loaded into the `jobs` table.

    Raises:
        FileNotFoundError: If the Parquet input file does not exist.
        duckdb.Error: If database operations fail.
        OSError: If required directories cannot be created.
    """
    logger.info("INFO - Starting DuckDB load from %s.", PROCESSED_PARQUET_PATH)

    if not os.path.isfile(PROCESSED_PARQUET_PATH):
        logger.error("ERROR - Parquet input file not found: %s", PROCESSED_PARQUET_PATH)
        raise FileNotFoundError(f"Parquet input file not found: {PROCESSED_PARQUET_PATH}")

    try:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        jobs_df = pd.read_parquet(PROCESSED_PARQUET_PATH)

        with duckdb.connect(DATABASE_PATH) as connection:
            connection.register("jobs_df", jobs_df)
            connection.execute(
                f"CREATE OR REPLACE TABLE {JOBS_TABLE_NAME} AS SELECT * FROM jobs_df"
            )
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {JOBS_TABLE_NAME}"
            ).fetchone()[0]
    except duckdb.Error as exc:
        logger.error("ERROR - DuckDB load failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("ERROR - Unexpected failure during DuckDB load: %s", exc)
        raise

    logger.info(
        "SUCCESS - Loaded %s rows into %s (%s).",
        row_count,
        JOBS_TABLE_NAME,
        DATABASE_PATH,
    )
    return int(row_count)
