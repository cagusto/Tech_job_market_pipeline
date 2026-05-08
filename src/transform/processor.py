"""Data transformation module for raw Adzuna jobs."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import pandas as pd


logger = logging.getLogger(__name__)

RAW_INPUT_PATH = os.path.join("data", "raw", "raw_adzuna_jobs.json")
PROCESSED_OUTPUT_PATH = os.path.join("data", "processed", "cleaned_jobs.parquet")
SKILL_KEYWORDS: Dict[str, str] = {
    "python": "python",
    "sql": "sql",
    "aws": "aws",
    "spark": "spark",
    "azure": "azure",
    "gcp": "gcp",
}


def _contains_keyword(text: str, keyword: str) -> bool:
    """Check if a keyword is present in a text (case-insensitive).

    Args:
        text: Input text to search.
        keyword: Keyword to match.

    Returns:
        True if keyword exists in text, otherwise False.
    """
    return keyword in text.lower()


def add_skill_columns(
    dataframe: pd.DataFrame, description_column: str, skills: Dict[str, str]
) -> pd.DataFrame:
    """Create boolean skill columns based on job description content.

    Args:
        dataframe: DataFrame containing job records.
        description_column: Column name with textual job descriptions.
        skills: Mapping of output skill suffix to keyword to search.

    Returns:
        The same DataFrame enriched with `skill_<name>` boolean columns.
    """
    descriptions = dataframe[description_column].fillna("").astype(str).str.lower()
    for skill_name, keyword in skills.items():
        dataframe[f"skill_{skill_name}"] = descriptions.apply(
            lambda text: _contains_keyword(text, keyword)
        )
    return dataframe


def _select_and_rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select relevant raw columns and rename to analytics-friendly names."""
    transformed = pd.DataFrame(
        {
            "job_title": dataframe.get("title", pd.Series(dtype="object")),
            "company_name": dataframe.get("company", pd.Series(dtype="object")).apply(
                lambda company: company.get("display_name")
                if isinstance(company, dict)
                else None
            ),
            "location": dataframe.get("location", pd.Series(dtype="object")).apply(
                lambda location: location.get("display_name")
                if isinstance(location, dict)
                else None
            ),
            "salary_min": dataframe.get("salary_min", pd.Series(dtype="float64")),
            "salary_max": dataframe.get("salary_max", pd.Series(dtype="float64")),
            "description": dataframe.get("description", pd.Series(dtype="object")),
        }
    )
    return transformed


def process_raw_jobs() -> pd.DataFrame:
    """Transform raw Adzuna JSON jobs into a cleaned parquet dataset.

    The function reads `data/raw/raw_adzuna_jobs.json`, transforms raw fields to
    a curated schema, creates boolean columns for key technology skills detected
    in job descriptions, handles nulls, and writes the result to
    `data/processed/cleaned_jobs.parquet`.

    Returns:
        Cleaned pandas DataFrame ready for downstream analytics.

    Raises:
        FileNotFoundError: If raw JSON file does not exist.
        json.JSONDecodeError: If raw JSON content is invalid.
        KeyError: If expected keys are missing in JSON structure.
        OSError: If writing parquet output fails.
        ValueError: If the raw results section is not a list.
    """
    logger.info("INFO - Starting jobs processing from raw JSON.")

    try:
        with open(RAW_INPUT_PATH, "r", encoding="utf-8") as file:
            raw_payload = json.load(file)
    except FileNotFoundError as exc:
        logger.error("ERROR - Raw input file not found: %s", RAW_INPUT_PATH)
        raise
    except json.JSONDecodeError as exc:
        logger.error("ERROR - Invalid JSON in raw input file: %s", exc)
        raise

    try:
        raw_results = raw_payload["results"]
    except KeyError:
        logger.error("ERROR - Raw payload missing required key: 'results'.")
        raise

    if not isinstance(raw_results, list):
        logger.error("ERROR - Raw payload 'results' must be a list.")
        raise ValueError("Raw payload 'results' must be a list.")

    try:
        raw_df = pd.DataFrame(raw_results)
        cleaned_df = _select_and_rename_columns(raw_df)

        # Fill missing text fields and normalize whitespace for consistency.
        for text_column in ("job_title", "company_name", "location", "description"):
            cleaned_df[text_column] = (
                cleaned_df[text_column].fillna("").astype(str).str.strip()
            )

        # Keep salary as numeric nullable columns.
        cleaned_df["salary_min"] = pd.to_numeric(
            cleaned_df["salary_min"], errors="coerce"
        )
        cleaned_df["salary_max"] = pd.to_numeric(
            cleaned_df["salary_max"], errors="coerce"
        )

        cleaned_df = add_skill_columns(
            dataframe=cleaned_df,
            description_column="description",
            skills=SKILL_KEYWORDS,
        )

        os.makedirs(os.path.dirname(PROCESSED_OUTPUT_PATH), exist_ok=True)
        cleaned_df.to_parquet(PROCESSED_OUTPUT_PATH, index=False)
    except Exception as exc:
        logger.error("ERROR - Failed during jobs processing: %s", exc)
        raise

    logger.info(
        "SUCCESS - Processing finished. %s records saved to %s.",
        len(cleaned_df),
        PROCESSED_OUTPUT_PATH,
    )
    return cleaned_df
