"""Streamlit dashboard for Tech Job Market pipeline insights."""

from __future__ import annotations

import os
from typing import List, Optional, Set

import duckdb
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join("data", "processed", "tech_jobs.duckdb")
JOBS_TABLE_NAME = "jobs"
TOP_SKILLS_LIMIT = 10

SENIORITY_ORDER: List[str] = ["Junior", "Pleno", "Senior", "Não Informado"]

RAW_TABLE_COLUMNS: List[str] = [
    "job_title",
    "company_name",
    "job_category",
    "seniority",
    "location",
    "location_normalized",
    "skills",
    "salary_min",
    "salary_max",
    "description",
]


# ---------------------------------------------------------------------------
# Data access and transformations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="A carregar dados do DuckDB...")
def load_jobs_data() -> pd.DataFrame:
    """Load the jobs table from the local DuckDB database."""
    if not os.path.isfile(DATABASE_PATH):
        raise FileNotFoundError(
            f"Base de dados não encontrada em '{DATABASE_PATH}'. "
            "Executa primeiro o passo de carga (load_data_to_duckdb)."
        )

    with duckdb.connect(DATABASE_PATH, read_only=True) as connection:
        return connection.execute(f"SELECT * FROM {JOBS_TABLE_NAME}").df()


def _sorted_unique_values(series: pd.Series) -> List[str]:
    """Return sorted unique non-empty values from a pandas Series."""
    values = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(values)


def _split_skills(skills_value: object) -> List[str]:
    """Split a comma-separated skills string into normalized tokens."""
    if skills_value is None or (isinstance(skills_value, float) and pd.isna(skills_value)):
        return []

    return [
        skill.strip().lower()
        for skill in str(skills_value).split(",")
        if skill.strip()
    ]


def compute_skill_frequency(
    dataframe: pd.DataFrame, top_n: int = TOP_SKILLS_LIMIT
) -> pd.DataFrame:
    """Count skill frequency from comma-separated `skills` column.

    Args:
        dataframe: Filtered jobs dataset.
        top_n: Number of top skills to return.

    Returns:
        DataFrame with columns `skill` and `count`.
    """
    if dataframe.empty or "skills" not in dataframe.columns:
        return pd.DataFrame(columns=["skill", "count"])

    exploded_skills = (
        dataframe["skills"]
        .apply(_split_skills)
        .explode()
        .dropna()
    )

    if exploded_skills.empty:
        return pd.DataFrame(columns=["skill", "count"])

    skill_counts = exploded_skills.value_counts().reset_index()
    skill_counts.columns = ["skill", "count"]
    return skill_counts.head(top_n)


def filter_jobs_by_skills(
    dataframe: pd.DataFrame, selected_skills: List[str]
) -> pd.DataFrame:
    """Keep only rows that contain at least one selected skill."""
    if not selected_skills:
        return dataframe

    selected_set: Set[str] = {skill.lower() for skill in selected_skills}

    def _row_matches(row_skills: object) -> bool:
        row_skill_set = set(_split_skills(row_skills))
        return bool(row_skill_set.intersection(selected_set))

    mask = dataframe["skills"].apply(_row_matches)
    return dataframe[mask].copy()


def apply_sidebar_filters(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply normalized dimension filters and dynamic top-skill filter."""
    st.sidebar.header("Filtros")

    category_options = _sorted_unique_values(dataframe["job_category"])
    location_options = _sorted_unique_values(dataframe["location_normalized"])
    seniority_options = _sorted_unique_values(dataframe["seniority"])

    selected_categories = st.sidebar.multiselect(
        "Categoria da vaga",
        options=category_options,
        default=[],
        placeholder="Todas as categorias",
    )
    selected_locations = st.sidebar.multiselect(
        "Localização normalizada",
        options=location_options,
        default=[],
        placeholder="Todas as localizações",
    )
    selected_seniorities = st.sidebar.multiselect(
        "Senioridade",
        options=seniority_options,
        default=[],
        placeholder="Todas as senioridades",
    )

    filtered = dataframe.copy()

    if selected_categories:
        filtered = filtered[filtered["job_category"].isin(selected_categories)]
    if selected_locations:
        filtered = filtered[filtered["location_normalized"].isin(selected_locations)]
    if selected_seniorities:
        filtered = filtered[filtered["seniority"].isin(selected_seniorities)]

    # Top 10 skills are computed from the current core-filtered dataset.
    top_skills_df = compute_skill_frequency(filtered, top_n=TOP_SKILLS_LIMIT)
    skill_options = top_skills_df["skill"].tolist()

    st.sidebar.markdown("##### Top Skills (filtro dinâmico)")
    selected_skills = st.sidebar.multiselect(
        "Skills",
        options=skill_options,
        default=[],
        placeholder="Todas as skills do Top 10",
        help="Opções recalculadas com base nos filtros de categoria, localização e senioridade.",
    )

    return filter_jobs_by_skills(filtered, selected_skills)


def build_seniority_crosstab(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build category vs seniority job count matrix."""
    if dataframe.empty:
        return pd.DataFrame()

    crosstab = pd.crosstab(
        dataframe["job_category"],
        dataframe["seniority"],
        dropna=False,
    )

    ordered_columns = [col for col in SENIORITY_ORDER if col in crosstab.columns]
    extra_columns = [col for col in crosstab.columns if col not in ordered_columns]
    crosstab = crosstab[ordered_columns + extra_columns]

    crosstab.index.name = "Categoria da Vaga"
    return crosstab


def compute_location_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count jobs grouped by normalized location."""
    if dataframe.empty:
        return pd.DataFrame(columns=["location_normalized", "job_count"])

    distribution = (
        dataframe["location_normalized"]
        .fillna("Não Informado")
        .astype(str)
        .str.strip()
        .replace("", "Não Informado")
        .value_counts()
        .reset_index()
    )
    distribution.columns = ["location_normalized", "job_count"]
    return distribution


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_kpis(dataframe: pd.DataFrame, top_skills_df: pd.DataFrame) -> None:
    """Display primary KPI metrics."""
    st.subheader("Métricas Principais")

    unique_skills = 0
    if not top_skills_df.empty:
        unique_skills = len(top_skills_df)

    col_total, col_categories, col_locations, col_skills = st.columns(4)

    col_total.metric("Total de vagas analisadas", len(dataframe))
    col_categories.metric("Categorias distintas", dataframe["job_category"].nunique())
    col_locations.metric(
        "Localizações distintas", dataframe["location_normalized"].nunique()
    )
    col_skills.metric("Skills no Top 10", unique_skills)


def render_top_skills_section(top_skills_df: pd.DataFrame) -> None:
    """Render Top 10 skills bar chart and summary table."""
    st.subheader("Top 10 Skills mais exigidas")

    if top_skills_df.empty:
        st.info("Sem dados de skills para os filtros selecionados.")
        return

    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        chart_data = top_skills_df.set_index("skill")["count"]
        st.bar_chart(chart_data, height=420)

    with table_col:
        st.markdown("##### Ranking")
        st.dataframe(top_skills_df, use_container_width=True, hide_index=True)


def render_seniority_crosstab(crosstab_df: pd.DataFrame) -> None:
    """Render category vs seniority pivot table."""
    st.subheader("Distribuição por Categoria e Senioridade")

    if crosstab_df.empty:
        st.info("Sem dados suficientes para gerar a tabela cruzada.")
        return

    st.dataframe(crosstab_df, use_container_width=True)


def render_location_section(location_df: pd.DataFrame) -> None:
    """Render normalized location distribution."""
    st.subheader("Distribuição por Localização Normalizada")

    if location_df.empty:
        st.info("Sem dados de localização para os filtros selecionados.")
        return

    chart_data = location_df.set_index("location_normalized")["job_count"]
    st.bar_chart(chart_data, height=380)
    st.dataframe(location_df, use_container_width=True, hide_index=True)


def render_raw_data_table(dataframe: pd.DataFrame) -> None:
    """Display filtered job records for manual inspection."""
    st.subheader("Dados brutos filtrados")

    if dataframe.empty:
        st.warning("Nenhuma vaga encontrada com os filtros atuais.")
        return

    available_columns = [col for col in RAW_TABLE_COLUMNS if col in dataframe.columns]
    display_df = dataframe[available_columns].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _validate_required_columns(dataframe: pd.DataFrame) -> Optional[str]:
    """Validate that normalized schema columns exist in loaded data."""
    required_columns = {
        "job_category",
        "seniority",
        "location_normalized",
        "skills",
        "job_title",
        "company_name",
    }
    missing = required_columns - set(dataframe.columns)
    if missing:
        return (
            "O dataset carregado não contém as colunas normalizadas esperadas: "
            f"{', '.join(sorted(missing))}. "
            "Reexecuta o pipeline de transformação e carga."
        )
    return None


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Tech Job Market Insights",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Tech Job Market Insights")
    st.caption(
        "Dashboard interativo com categorias, senioridade, localização normalizada "
        "e skills dinâmicas."
    )

    try:
        jobs_df = load_jobs_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    schema_error = _validate_required_columns(jobs_df)
    if schema_error:
        st.error(schema_error)
        st.stop()

    filtered_df = apply_sidebar_filters(jobs_df)

    top_skills_df = compute_skill_frequency(filtered_df, top_n=TOP_SKILLS_LIMIT)
    seniority_crosstab_df = build_seniority_crosstab(filtered_df)
    location_df = compute_location_distribution(filtered_df)

    render_kpis(filtered_df, top_skills_df)
    st.divider()

    render_top_skills_section(top_skills_df)
    st.divider()

    render_seniority_crosstab(seniority_crosstab_df)
    st.divider()

    render_location_section(location_df)
    st.divider()

    render_raw_data_table(filtered_df)


if __name__ == "__main__":
    main()
