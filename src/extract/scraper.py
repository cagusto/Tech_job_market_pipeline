"""Adzuna extraction module for tech job market data."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import requests
import urllib3
from dotenv import load_dotenv
from requests import Response
from requests.exceptions import RequestException


logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
DEFAULT_COUNTRY = "br"
DEFAULT_RESULTS_PER_PAGE = 50
DEFAULT_TIMEOUT_SECONDS = 15
RAW_OUTPUT_PATH = os.path.join("data", "raw", "raw_adzuna_jobs.json")


def _build_query_params(
    app_id: str, app_key: str, page: int, search_term: str, location: str
) -> Dict[str, Any]:
    """Build query params for the Adzuna jobs endpoint."""
    return {
        "app_id": app_id,
        "app_key": app_key,
        "what": search_term,
        "where": location,
        "results_per_page": DEFAULT_RESULTS_PER_PAGE,
        "content-type": "application/json",
    }


def _fetch_jobs(
    country: str,
    page: int,
    app_id: str,
    app_key: str,
    search_term: str,
    location: str,
) -> List[Dict[str, Any]]:
    """Fetch jobs from Adzuna for one search location."""
    endpoint = f"{ADZUNA_BASE_URL}/{country}/search/{page}"
    params = _build_query_params(
        app_id=app_id,
        app_key=app_key,
        page=page,
        search_term=search_term,
        location=location,
    )

    response: Response = requests.get(
        endpoint,
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        verify=False,
    )
    response.raise_for_status()
    payload: Dict[str, Any] = response.json()
    return payload.get("results", [])


def extract_adzuna_jobs(page: int = 1) -> Dict[str, Any]:
    """Extract Data Engineer jobs from Adzuna and persist raw JSON.

    This function reads Adzuna credentials from environment variables loaded
    from a `.env` file, fetches Data Engineer vacancies for Brazil and remote
    searches, and stores the raw extraction output in `data/raw`.

    Args:
        page: The result page number to query in the Adzuna API.

    Returns:
        A dictionary containing metadata and extracted raw job records.

    Raises:
        ValueError: If required API credentials are missing.
        RequestException: If the HTTP request fails (e.g., timeout/connection).
        OSError: If writing the output file fails.
        json.JSONDecodeError: If API response is not valid JSON.
    """
    logger.info("INFO - Starting Adzuna extraction for Data Engineer jobs.")

    load_dotenv()
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        logger.error(
            "ERROR - Missing ADZUNA_APP_ID and/or ADZUNA_APP_KEY in environment."
        )
        raise ValueError(
            "Missing ADZUNA_APP_ID and/or ADZUNA_APP_KEY in environment variables."
        )

    try:
        brazil_jobs = _fetch_jobs(
            country=DEFAULT_COUNTRY,
            page=page,
            app_id=app_id,
            app_key=app_key,
            search_term="Data Engineer",
            location="Brasil",
        )
        remote_jobs = _fetch_jobs(
            country=DEFAULT_COUNTRY,
            page=page,
            app_id=app_id,
            app_key=app_key,
            search_term="Data Engineer",
            location="Remoto",
        )
    except RequestException as exc:
        logger.error("ERROR - Adzuna API request failed: %s", exc)
        raise

    combined_jobs = brazil_jobs + remote_jobs

    # Remove duplicates by Adzuna job id while preserving order.
    seen_ids = set()
    deduplicated_jobs: List[Dict[str, Any]] = []
    for job in combined_jobs:
        job_id = job.get("id")
        if job_id in seen_ids:
            continue
        seen_ids.add(job_id)
        deduplicated_jobs.append(job)

    extraction_payload: Dict[str, Any] = {
        "source": "adzuna",
        "country": DEFAULT_COUNTRY,
        "query": "Data Engineer",
        "locations": ["Brasil", "Remoto"],
        "page": page,
        "total_records": len(deduplicated_jobs),
        "results": deduplicated_jobs,
    }

    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(extraction_payload, file, ensure_ascii=False, indent=2)

    logger.info(
        "SUCCESS - Adzuna extraction finished. %s records saved to %s.",
        extraction_payload["total_records"],
        RAW_OUTPUT_PATH,
    )
    return extraction_payload
