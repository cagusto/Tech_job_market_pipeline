"""Data transformation module for raw Adzuna jobs."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


logger = logging.getLogger(__name__)

RAW_INPUT_PATH = os.path.join("data", "raw", "raw_adzuna_jobs.json")
PROCESSED_OUTPUT_PATH = os.path.join("data", "processed", "cleaned_jobs.parquet")

REMOTE_KEYWORDS: Tuple[str, ...] = (
    "remote",
    "remoto",
    "home office",
    "home-office",
    "trabalho remoto",
    "anywhere",
)

SENIORITY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Junior", ("jr", "junior", "júnior", "trainee", "estagiário", "estagiaria")),
    ("Pleno", ("pleno", "mid", "mid-level", "mid level", "intermediate")),
    ("Senior", ("sr", "senior", "sênior", "lead", "principal", "staff")),
)

JOB_CATEGORY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Cientista de Dados", ("scientist", "cientista", "pesquisador")),
    ("Analista de Dados", ("analyst", "analista", "analytics")),
    ("Engenheiro de Dados", ("engineer", "engenheiro", "engineering")),
)

BRAZILIAN_STATES: Dict[str, str] = {
    "acre": "Acre",
    "alagoas": "Alagoas",
    "amapa": "Amapá",
    "amapá": "Amapá",
    "amazonas": "Amazonas",
    "bahia": "Bahia",
    "ceara": "Ceará",
    "ceará": "Ceará",
    "distrito federal": "Distrito Federal",
    "espirito santo": "Espírito Santo",
    "espírito santo": "Espírito Santo",
    "goias": "Goiás",
    "goiás": "Goiás",
    "maranhao": "Maranhão",
    "maranhão": "Maranhão",
    "mato grosso do sul": "Mato Grosso do Sul",
    "mato grosso": "Mato Grosso",
    "minas gerais": "Minas Gerais",
    "para": "Pará",
    "pará": "Pará",
    "paraiba": "Paraíba",
    "paraíba": "Paraíba",
    "parana": "Paraná",
    "paraná": "Paraná",
    "pernambuco": "Pernambuco",
    "piaui": "Piauí",
    "piauí": "Piauí",
    "rio de janeiro": "Rio de Janeiro",
    "rio grande do norte": "Rio Grande do Norte",
    "rio grande do sul": "Rio Grande do Sul",
    "rondonia": "Rondônia",
    "rondônia": "Rondônia",
    "roraima": "Roraima",
    "santa catarina": "Santa Catarina",
    "sao paulo": "São Paulo",
    "são paulo": "São Paulo",
    "sergipe": "Sergipe",
    "tocantins": "Tocantins",
    "ac": "Acre",
    "al": "Alagoas",
    "ap": "Amapá",
    "am": "Amazonas",
    "ba": "Bahia",
    "ce": "Ceará",
    "df": "Distrito Federal",
    "es": "Espírito Santo",
    "go": "Goiás",
    "ma": "Maranhão",
    "ms": "Mato Grosso do Sul",
    "mt": "Mato Grosso",
    "mg": "Minas Gerais",
    "pa": "Pará",
    "pb": "Paraíba",
    "pr": "Paraná",
    "pe": "Pernambuco",
    "pi": "Piauí",
    "rj": "Rio de Janeiro",
    "rn": "Rio Grande do Norte",
    "rs": "Rio Grande do Sul",
    "ro": "Rondônia",
    "rr": "Roraima",
    "sc": "Santa Catarina",
    "sp": "São Paulo",
    "se": "Sergipe",
    "to": "Tocantins",
}

# Longer phrases first to avoid partial matches (e.g., "power bi" before "bi").
TECH_SKILLS: Tuple[str, ...] = (
    "apache airflow",
    "apache kafka",
    "apache spark",
    "amazon redshift",
    "google bigquery",
    "microsoft fabric",
    "power bi",
    "machine learning",
    "deep learning",
    "data warehouse",
    "data lake",
    "data modeling",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "kubernetes",
    "terraform",
    "postgresql",
    "mysql",
    "mongodb",
    "snowflake",
    "databricks",
    "looker",
    "tableau",
    "powerbi",
    "airflow",
    "dbt",
    "kafka",
    "spark",
    "hadoop",
    "hive",
    "presto",
    "trino",
    "flink",
    "redis",
    "elasticsearch",
    "docker",
    "jenkins",
    "gitlab",
    "github",
    "bitbucket",
    "ansible",
    "prometheus",
    "grafana",
    "excel",
    "python",
    "sql",
    "aws",
    "azure",
    "gcp",
    "java",
    "scala",
    "r",
    "go",
    "rust",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "react",
    "angular",
    "vue",
    "node",
    "django",
    "flask",
    "fastapi",
    "pandas",
    "numpy",
    "polars",
    "duckdb",
    "superset",
    "metabase",
    "lookml",
    "etl",
    "elt",
)


def _normalize_text(value: Optional[object]) -> str:
    """Convert nullable values to a clean lowercase string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _contains_any_keyword(text: str, keywords: Tuple[str, ...]) -> bool:
    """Check whether any keyword appears in text using word boundaries."""
    for keyword in keywords:
        pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def categorize_job_category(job_title: Optional[object]) -> str:
    """Map original job title into a high-level job category.

    Args:
        job_title: Original job title from the source system.

    Returns:
        Normalized job category label.
    """
    title = _normalize_text(job_title)
    if not title:
        return "Não Classificado"

    for category, keywords in JOB_CATEGORY_RULES:
        if _contains_any_keyword(title, keywords):
            return category

    return "Outros"


def extract_seniority(job_title: Optional[object]) -> str:
    """Extract seniority level from the original job title.

    Args:
        job_title: Original job title from the source system.

    Returns:
        Seniority label (Junior, Pleno, Senior, or Não Informado).
    """
    title = _normalize_text(job_title)
    if not title:
        return "Não Informado"

    for seniority, keywords in SENIORITY_RULES:
        if _contains_any_keyword(title, keywords):
            return seniority

    return "Não Informado"


def _is_remote_location(location_text: str) -> bool:
    """Return True when location text indicates remote work."""
    return any(keyword in location_text for keyword in REMOTE_KEYWORDS)


def _extract_brazilian_state(location_text: str) -> Optional[str]:
    """Try to identify a Brazilian state from a location string."""
    cleaned = re.sub(r"^estado de\s+", "", location_text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("  ", " ").strip()

    # Check full location and comma-separated segments (city, state patterns).
    segments = [segment.strip() for segment in cleaned.split(",") if segment.strip()]
    candidates = segments if segments else [cleaned]

    # Longest state names first to avoid partial false positives.
    sorted_states = sorted(BRAZILIAN_STATES.items(), key=lambda item: len(item[0]), reverse=True)

    for candidate in reversed(candidates):
        candidate_norm = candidate.lower()
        for state_key, state_name in sorted_states:
            if re.search(rf"\b{re.escape(state_key)}\b", candidate_norm, flags=re.IGNORECASE):
                return state_name
            if candidate_norm == state_key:
                return state_name

    return None


def normalize_location(location: Optional[object]) -> str:
    """Normalize location into Remoto or a Brazilian state when possible.

    Args:
        location: Raw location value from source.

    Returns:
        Normalized location label.
    """
    location_text = _normalize_text(location)
    if not location_text:
        return "Não Informado"

    if _is_remote_location(location_text):
        return "Remoto"

    state = _extract_brazilian_state(location_text)
    if state:
        return state

    # Fallback: keep a simplified free-text location without noisy prefixes.
    simplified = re.sub(r"^estado de\s+", "", location_text, flags=re.IGNORECASE).strip()
    return simplified.title() if simplified else "Não Informado"


def extract_skills_from_description(description: Optional[object]) -> str:
    """Extract technology skills from job description as a CSV string.

    Args:
        description: Job description text.

    Returns:
        Comma-separated skills found in the description (sorted, unique).
    """
    text = _normalize_text(description)
    if not text:
        return ""

    found_skills: List[str] = []
    for skill in TECH_SKILLS:
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found_skills.append(skill)

    # Preserve discovery order while removing duplicates.
    unique_skills = list(dict.fromkeys(found_skills))
    return ", ".join(unique_skills)


def _select_base_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select and rename core columns from raw Adzuna payload."""
    return pd.DataFrame(
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
                else location
            ),
            "salary_min": dataframe.get("salary_min", pd.Series(dtype="float64")),
            "salary_max": dataframe.get("salary_max", pd.Series(dtype="float64")),
            "description": dataframe.get("description", pd.Series(dtype="object")),
        }
    )


def _apply_business_rules(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply dynamic business transformations to the jobs DataFrame."""
    enriched = dataframe.copy()

    enriched["job_category"] = enriched["job_title"].apply(categorize_job_category)
    enriched["seniority"] = enriched["job_title"].apply(extract_seniority)
    enriched["location_normalized"] = enriched["location"].apply(normalize_location)
    enriched["skills"] = enriched["description"].apply(extract_skills_from_description)

    return enriched


def _handle_null_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize null handling for text and numeric columns."""
    cleaned = dataframe.copy()

    text_columns = (
        "job_title",
        "company_name",
        "location",
        "description",
        "job_category",
        "seniority",
        "location_normalized",
        "skills",
    )
    for column in text_columns:
        if column not in cleaned.columns:
            continue
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    if "salary_min" in cleaned.columns:
        cleaned["salary_min"] = pd.to_numeric(cleaned["salary_min"], errors="coerce")
    if "salary_max" in cleaned.columns:
        cleaned["salary_max"] = pd.to_numeric(cleaned["salary_max"], errors="coerce")

    return cleaned


def _select_final_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep only the curated columns used downstream."""
    final_columns = [
        "job_title",
        "company_name",
        "location",
        "location_normalized",
        "job_category",
        "seniority",
        "skills",
        "salary_min",
        "salary_max",
        "description",
    ]
    available_columns = [column for column in final_columns if column in dataframe.columns]
    return dataframe[available_columns].copy()


def process_raw_jobs() -> pd.DataFrame:
    """Transform raw Adzuna JSON jobs into a cleaned parquet dataset.

    Reads `data/raw/raw_adzuna_jobs.json`, applies business rules for category,
    seniority, location normalization and dynamic skill extraction, then writes
    the curated dataset to `data/processed/cleaned_jobs.parquet`.

    Returns:
        Cleaned pandas DataFrame ready for downstream analytics.

    Raises:
        FileNotFoundError: If raw JSON file does not exist.
        json.JSONDecodeError: If raw JSON content is invalid.
        KeyError: If expected keys are missing in JSON structure.
        OSError: If writing parquet output fails.
        ValueError: If the raw results section is not a list.
    """
    logger.info("INFO - Starting dynamic jobs processing from raw JSON.")

    try:
        with open(RAW_INPUT_PATH, "r", encoding="utf-8") as file:
            raw_payload = json.load(file)
    except FileNotFoundError:
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
        # 1) Build base frame from raw payload.
        raw_df = pd.DataFrame(raw_results)
        base_df = _select_base_columns(raw_df)

        # 2) Create all derived columns before null handling.
        enriched_df = _apply_business_rules(base_df)

        # 3) Clean nulls and types after derived columns exist.
        null_safe_df = _handle_null_values(enriched_df)

        # 4) Keep final curated schema.
        cleaned_df = _select_final_columns(null_safe_df)

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
