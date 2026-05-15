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
    (
        "Estágio/Trainee",
        ("estagio", "estágio", "intern", "trainee", "estagiario", "estagiário", "estagiaria", "estagiária"),
    ),
    ("Assistente", ("assistente", "assistant")),
    ("Junior", ("jr", "junior", "júnior")),
    ("Pleno", ("pleno", "mid", "mid-level", "mid level", "intermediate")),
    ("Sênior", ("sr", "senior", "sênior", "principal")),
    ("Especialista", ("especialista", "specialist", "lead", "tech lead")),
    ("Gestão", ("manager", "gerente", "diretor", "diretora", "head", "coordenador", "coordenadora")),
)

JOB_CATEGORY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Cientista de Dados", ("scientist", "cientista", "pesquisador")),
    ("Analista de Dados", ("analyst", "analista", "analytics")),
    ("Engenheiro de Dados", ("engineer", "engenheiro", "engineering")),
)

# Mapping dictionary: cities/capitals/aliases -> Brazilian state name.
LOCATION_TO_STATE_MAP: Dict[str, str] = {
    # Acre
    "acre": "Acre",
    "rio branco": "Acre",
    "ac": "Acre",
    # Alagoas
    "alagoas": "Alagoas",
    "maceio": "Alagoas",
    "maceió": "Alagoas",
    "al": "Alagoas",
    # Amapá
    "amapa": "Amapá",
    "amapá": "Amapá",
    "macapa": "Amapá",
    "macapá": "Amapá",
    "ap": "Amapá",
    # Amazonas
    "amazonas": "Amazonas",
    "manaus": "Amazonas",
    "am": "Amazonas",
    # Bahia
    "bahia": "Bahia",
    "salvador": "Bahia",
    "feira de santana": "Bahia",
    "ba": "Bahia",
    # Ceará
    "ceara": "Ceará",
    "ceará": "Ceará",
    "fortaleza": "Ceará",
    "ce": "Ceará",
    # Distrito Federal
    "distrito federal": "Distrito Federal",
    "brasilia": "Distrito Federal",
    "brasília": "Distrito Federal",
    "df": "Distrito Federal",
    # Espírito Santo
    "espirito santo": "Espírito Santo",
    "espírito santo": "Espírito Santo",
    "vitoria": "Espírito Santo",
    "vitória": "Espírito Santo",
    "vila velha": "Espírito Santo",
    "serra": "Espírito Santo",
    "cariacica": "Espírito Santo",
    "anchieta": "Espírito Santo",
    "es": "Espírito Santo",
    # Goiás
    "goias": "Goiás",
    "goiás": "Goiás",
    "goiania": "Goiás",
    "goiânia": "Goiás",
    "aparecida de goiania": "Goiás",
    "go": "Goiás",
    # Maranhão
    "maranhao": "Maranhão",
    "maranhão": "Maranhão",
    "sao luis": "Maranhão",
    "são luís": "Maranhão",
    "ma": "Maranhão",
    # Mato Grosso
    "mato grosso": "Mato Grosso",
    "cuiaba": "Mato Grosso",
    "cuiabá": "Mato Grosso",
    "mt": "Mato Grosso",
    # Mato Grosso do Sul
    "mato grosso do sul": "Mato Grosso do Sul",
    "campo grande": "Mato Grosso do Sul",
    "ms": "Mato Grosso do Sul",
    # Minas Gerais
    "minas gerais": "Minas Gerais",
    "belo horizonte": "Minas Gerais",
    "contagem": "Minas Gerais",
    "uberlandia": "Minas Gerais",
    "uberlândia": "Minas Gerais",
    "betim": "Minas Gerais",
    "juiz de fora": "Minas Gerais",
    "montes claros": "Minas Gerais",
    "mg": "Minas Gerais",
    # Pará
    "para": "Pará",
    "pará": "Pará",
    "belem": "Pará",
    "belém": "Pará",
    "ananindeua": "Pará",
    "pa": "Pará",
    # Paraíba
    "paraiba": "Paraíba",
    "paraíba": "Paraíba",
    "joao pessoa": "Paraíba",
    "joão pessoa": "Paraíba",
    "pb": "Paraíba",
    # Paraná
    "parana": "Paraná",
    "paraná": "Paraná",
    "curitiba": "Paraná",
    "londrina": "Paraná",
    "maringa": "Paraná",
    "maringá": "Paraná",
    "pr": "Paraná",
    # Pernambuco
    "pernambuco": "Pernambuco",
    "recife": "Pernambuco",
    "olinda": "Pernambuco",
    "pe": "Pernambuco",
    # Piauí
    "piaui": "Piauí",
    "piauí": "Piauí",
    "teresina": "Piauí",
    "pi": "Piauí",
    # Rio de Janeiro
    "rio de janeiro": "Rio de Janeiro",
    "niteroi": "Rio de Janeiro",
    "niterói": "Rio de Janeiro",
    "duque de caxias": "Rio de Janeiro",
    "nova iguacu": "Rio de Janeiro",
    "nova iguaçu": "Rio de Janeiro",
    "sao goncalo": "Rio de Janeiro",
    "são gonçalo": "Rio de Janeiro",
    "rj": "Rio de Janeiro",
    # Rio Grande do Norte
    "rio grande do norte": "Rio Grande do Norte",
    "natal": "Rio Grande do Norte",
    "rn": "Rio Grande do Norte",
    # Rio Grande do Sul
    "rio grande do sul": "Rio Grande do Sul",
    "porto alegre": "Rio Grande do Sul",
    "caxias do sul": "Rio Grande do Sul",
    "canoas": "Rio Grande do Sul",
    "rs": "Rio Grande do Sul",
    # Rondônia
    "rondonia": "Rondônia",
    "rondônia": "Rondônia",
    "porto velho": "Rondônia",
    "ro": "Rondônia",
    # Roraima
    "roraima": "Roraima",
    "boa vista": "Roraima",
    "rr": "Roraima",
    # Santa Catarina
    "santa catarina": "Santa Catarina",
    "florianopolis": "Santa Catarina",
    "florianópolis": "Santa Catarina",
    "joinville": "Santa Catarina",
    "blumenau": "Santa Catarina",
    "sc": "Santa Catarina",
    # São Paulo
    "sao paulo": "São Paulo",
    "são paulo": "São Paulo",
    "campinas": "São Paulo",
    "santos": "São Paulo",
    "guarulhos": "São Paulo",
    "sorocaba": "São Paulo",
    "ribeirao preto": "São Paulo",
    "ribeirão preto": "São Paulo",
    "sao jose dos campos": "São Paulo",
    "são josé dos campos": "São Paulo",
    "osasco": "São Paulo",
    "sp": "São Paulo",
    # Sergipe
    "sergipe": "Sergipe",
    "aracaju": "Sergipe",
    "se": "Sergipe",
    # Tocantins
    "tocantins": "Tocantins",
    "palmas": "Tocantins",
    "to": "Tocantins",
}

# Longer phrases first to avoid partial matches (e.g., "machine learning" before "r").
TECH_SKILLS: Tuple[str, ...] = (
    "apache airflow",
    "apache kafka",
    "apache spark",
    "amazon redshift",
    "google bigquery",
    "microsoft fabric",
    "machine learning",
    "deep learning",
    "data warehouse",
    "data modeling",
    "sql server",
    "power bi",
    "data lake",
    "scikit-learn",
    "ci/cd",
    "pytorch",
    "tensorflow",
    "kubernetes",
    "terraform",
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "snowflake",
    "databricks",
    "bigquery",
    "redshift",
    "cassandra",
    "metabase",
    "looker",
    "tableau",
    "qlik",
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
    "oracle",
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
    "nlp",
    "agile",
    "scrum",
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
    "lookml",
    "etl",
    "elt",
)


def _normalize_text(value: Optional[object]) -> str:
    """Convert nullable values to a clean lowercase string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _build_keyword_pattern(keyword: str) -> str:
    """Build a regex pattern that respects token boundaries around a keyword."""
    escaped_keyword = re.escape(keyword)
    start_boundary = r"\b" if re.match(r"\w", keyword[0]) else r"(?<!\w)"
    end_boundary = r"\b" if re.match(r"\w", keyword[-1]) else r"(?!\w)"
    return rf"{start_boundary}{escaped_keyword}{end_boundary}"


def _contains_any_keyword(text: str, keywords: Tuple[str, ...]) -> bool:
    """Check whether any keyword appears in text using word/token boundaries."""
    for keyword in keywords:
        pattern = _build_keyword_pattern(keyword)
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


def extract_seniority(
    job_title: Optional[object], description: Optional[object] = None
) -> str:
    """Extract seniority level from the original job title and description.

    Args:
        job_title: Original job title from the source system.
        description: Job description text.

    Returns:
        Seniority label by highest detected hierarchy, or Não Informado.
    """
    searchable_text = f"{_normalize_text(job_title)} {_normalize_text(description)}".strip()
    if not searchable_text:
        return "Não Informado"

    for seniority, keywords in reversed(SENIORITY_RULES):
        if _contains_any_keyword(searchable_text, keywords):
            return seniority

    return "Não Informado"


def _lookup_state_from_text(location_text: str) -> Optional[str]:
    """Match a location string against the mapping dictionary."""
    cleaned_text = re.sub(r"^estado de\s+", "", location_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip().lower()
    if not cleaned_text:
        return None

    # Longest keys first to prioritize specific city names over short aliases.
    sorted_keys = sorted(LOCATION_TO_STATE_MAP.keys(), key=len, reverse=True)
    for location_key in sorted_keys:
        pattern = rf"\b{re.escape(location_key)}\b"
        if re.search(pattern, cleaned_text, flags=re.IGNORECASE):
            return LOCATION_TO_STATE_MAP[location_key]

    return None


def normalize_location(location: Optional[object]) -> str:
    """Normalize location using mapping dictionary rules.

    Rules:
        1) Remote keywords -> ``Remoto``.
        2) Dictionary lookup (case-insensitive) for states/cities/aliases.
        3) Fallback to the last comma-separated token lookup.
        4) If still unmatched -> ``Outros``.

    Args:
        location: Raw location value from source.

    Returns:
        Normalized location label.
    """
    location_text = _normalize_text(location)
    if not location_text:
        return "Não Informado"

    if any(keyword in location_text for keyword in REMOTE_KEYWORDS):
        return "Remoto"

    # Try full string first, then each comma-separated segment (right to left).
    segments = [
        segment.strip()
        for segment in re.sub(r"^estado de\s+", "", location_text, flags=re.IGNORECASE).split(",")
        if segment.strip()
    ]
    search_candidates = [location_text] + list(reversed(segments))

    for candidate in search_candidates:
        mapped_state = _lookup_state_from_text(candidate)
        if mapped_state:
            return mapped_state

    # Fallback: last term after comma, including direct dictionary key match.
    if segments:
        last_segment = segments[-1].strip().lower()
        if last_segment in LOCATION_TO_STATE_MAP:
            return LOCATION_TO_STATE_MAP[last_segment]

        mapped_last_segment = _lookup_state_from_text(last_segment)
        if mapped_last_segment:
            return mapped_last_segment

    return "Outros"


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
        pattern = _build_keyword_pattern(skill)
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
    enriched["seniority"] = enriched.apply(
        lambda row: extract_seniority(row.get("job_title"), row.get("description")),
        axis=1,
    )
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
