"""Adzuna extraction module for tech job market data."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Set

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
MAX_PAGES_PER_SEARCH = 5
REQUEST_DELAY_SECONDS = 1
RAW_OUTPUT_PATH = os.path.join("data", "raw", "raw_adzuna_jobs.json")

JOB_TITLES: List[str] = [
    "Data Engineer",
    "Data Scientist",
    "Analytics Engineer",
]
SEARCH_LOCATIONS: List[str] = ["Brasil", "Remoto"]


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


def _fetch_jobs_page(
    country: str,
    page: int,
    app_id: str,
    app_key: str,
    job_title: str,
    location: str,
) -> List[Dict[str, Any]]:
    """Fetch one page of jobs from Adzuna for a title/location pair.

    Args:
        country: Adzuna country code (e.g. ``br``).
        page: Page number to request.
        app_id: Adzuna application ID.
        app_key: Adzuna application key.
        job_title: Job title search term.
        location: Location search term.

    Returns:
        List of job records returned for the requested page.
    """
    endpoint = f"{ADZUNA_BASE_URL}/{country}/search/{page}"
    params = _build_query_params(
        app_id=app_id,
        app_key=app_key,
        page=page,
        search_term=job_title,
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


def _fetch_paginated_jobs(
    country: str,
    app_id: str,
    app_key: str,
    job_title: str,
    location: str,
) -> List[Dict[str, Any]]:
    """Fetch up to ``MAX_PAGES_PER_SEARCH`` pages for one title/location.

    Args:
        country: Adzuna country code.
        app_id: Adzuna application ID.
        app_key: Adzuna application key.
        job_title: Job title search term.
        location: Location search term.

    Returns:
        Combined list of jobs across all fetched pages.
    """
    collected_jobs: List[Dict[str, Any]] = []

    for page in range(1, MAX_PAGES_PER_SEARCH + 1):
        logger.info(
            "INFO - Fetching jobs | title='%s' | location='%s' | page=%s/%s",
            job_title,
            location,
            page,
            MAX_PAGES_PER_SEARCH,
        )

        page_jobs = _fetch_jobs_page(
            country=country,
            page=page,
            app_id=app_id,
            app_key=app_key,
            job_title=job_title,
            location=location,
        )

        if not page_jobs:
            logger.info(
                "INFO - No results returned | title='%s' | location='%s' | page=%s. "
                "Stopping pagination for this search.",
                job_title,
                location,
                page,
            )
            break

        collected_jobs.extend(page_jobs)
        logger.info(
            "INFO - Page fetched | title='%s' | location='%s' | page=%s | "
            "records_on_page=%s | records_accumulated=%s",
            job_title,
            location,
            page,
            len(page_jobs),
            len(collected_jobs),
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    return collected_jobs


def _deduplicate_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate jobs by Adzuna ID while preserving order."""
    seen_ids: Set[Any] = set()
    deduplicated_jobs: List[Dict[str, Any]] = []

    for job in jobs:
        job_id = job.get("id")
        if job_id in seen_ids:
            continue
        seen_ids.add(job_id)
        deduplicated_jobs.append(job)

    return deduplicated_jobs


def extract_adzuna_jobs() -> Dict[str, Any]:
    """Extract multiple job titles from Adzuna and persist raw JSON.

    Iterates over predefined job titles and locations, paginating up to
    ``MAX_PAGES_PER_SEARCH`` pages per search. Applies a delay between API
    calls to reduce rate-limit errors and writes all results to a single
    ``raw_adzuna_jobs.json`` file.

    Returns:
        Dictionary containing metadata and extracted raw job records.

    Raises:
        ValueError: If required API credentials are missing.
        RequestException: If an HTTP request fails.
        OSError: If writing the output file fails.
        json.JSONDecodeError: If API response is not valid JSON.
    """
    logger.info(
        "INFO - Starting Adzuna extraction | titles=%s | locations=%s | "
        "max_pages=%s",
        JOB_TITLES,
        SEARCH_LOCATIONS,
        MAX_PAGES_PER_SEARCH,
    )

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

    all_jobs: List[Dict[str, Any]] = []

    try:
        for job_title in JOB_TITLES:
            logger.info("INFO - Processing job title='%s'.", job_title)

            for location in SEARCH_LOCATIONS:
                title_location_jobs = _fetch_paginated_jobs(
                    country=DEFAULT_COUNTRY,
                    app_id=app_id,
                    app_key=app_key,
                    job_title=job_title,
                    location=location,
                )
                all_jobs.extend(title_location_jobs)

    except RequestException as exc:
        logger.error("ERROR - Adzuna API request failed: %s", exc)
        raise

    deduplicated_jobs = _deduplicate_jobs(all_jobs)

    extraction_payload: Dict[str, Any] = {
        "source": "adzuna",
        "country": DEFAULT_COUNTRY,
        "job_titles": JOB_TITLES,
        "locations": SEARCH_LOCATIONS,
        "max_pages_per_search": MAX_PAGES_PER_SEARCH,
        "total_records": len(deduplicated_jobs),
        "results": deduplicated_jobs,
    }

    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(extraction_payload, file, ensure_ascii=False, indent=2)

    logger.info(
        "SUCCESS - Adzuna extraction finished. %s unique records saved to %s.",
        extraction_payload["total_records"],
        RAW_OUTPUT_PATH,
    )
    return extraction_payload
