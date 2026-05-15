"""Streamlit dashboard for Tech Job Market pipeline insights."""

from __future__ import annotations

import os
from typing import List

import duckdb
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join("data", "processed", "tech_jobs.duckdb")
JOBS_TABLE_NAME = "jobs"
SKILL_COLUMNS: List[str] = [
    "skill_python",
    "skill_sql",
    "skill_aws",
    "skill_spark",
    "skill_azure",
    "skill_gcp",
]


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading jobs from DuckDB...")
def load_jobs_data() -> pd.DataFrame:
    """Load the jobs table from the local DuckDB database.

    Returns:
        DataFrame with all records from the `jobs` table.

    Raises:
        FileNotFoundError: If the DuckDB file does not exist.
    """
    if not os.path.isfile(DATABASE_PATH):
        raise FileNotFoundError(
            f"Database not found at '{DATABASE_PATH}'. "
            "Run the ETL load step first (load_data_to_duckdb)."
        )

    with duckdb.connect(DATABASE_PATH, read_only=True) as connection:
        return connection.execute(f"SELECT * FROM {JOBS_TABLE_NAME}").df()


def apply_sidebar_filters(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply job title and location filters from the sidebar.

    Args:
        dataframe: Full jobs dataset.

    Returns:
        Filtered copy of the input DataFrame.
    """
    st.sidebar.header("Filters")

    job_titles = sorted(
        dataframe["job_title"].dropna().astype(str).str.strip().unique().tolist()
    )
    locations = sorted(
        dataframe["location"].dropna().astype(str).str.strip().unique().tolist()
    )

    selected_titles = st.sidebar.multiselect(
        "Job title",
        options=job_titles,
        default=[],
        placeholder="All titles",
    )
    selected_locations = st.sidebar.multiselect(
        "Location",
        options=locations,
        default=[],
        placeholder="All locations",
    )

    filtered = dataframe.copy()
    if selected_titles:
        filtered = filtered[filtered["job_title"].isin(selected_titles)]
    if selected_locations:
        filtered = filtered[filtered["location"].isin(selected_locations)]

    return filtered


def compute_top_skills(
    dataframe: pd.DataFrame, skill_columns: List[str], top_n: int = 10
) -> pd.DataFrame:
    """Aggregate boolean skill columns into demand counts.

    Args:
        dataframe: Filtered jobs dataset.
        skill_columns: List of boolean skill column names.
        top_n: Maximum number of skills to return.

    Returns:
        DataFrame with columns `skill` and `count`, sorted descending.
    """
    available_skills = [col for col in skill_columns if col in dataframe.columns]
    if not available_skills:
        return pd.DataFrame(columns=["skill", "count"])

    skill_counts = dataframe[available_skills].sum().astype(int)
    skill_counts.index = skill_counts.index.str.replace("skill_", "", regex=False)

    top_skills = (
        skill_counts.reset_index()
        .rename(columns={"index": "skill", 0: "count"})
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return top_skills


def compute_location_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count jobs grouped by location.

    Args:
        dataframe: Filtered jobs dataset.

    Returns:
        DataFrame with columns `location` and `job_count`.
    """
    distribution = (
        dataframe["location"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
        .value_counts()
        .reset_index()
    )
    distribution.columns = ["location", "job_count"]
    return distribution


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_kpis(dataframe: pd.DataFrame) -> None:
    """Display primary KPI metrics."""
    st.subheader("Métricas Principais")
    col_total, col_skills, col_locations = st.columns(3)

    detected_skills = sum(
        1 for col in SKILL_COLUMNS if col in dataframe.columns and dataframe[col].any()
    )
    unique_locations = dataframe["location"].nunique()

    col_total.metric(label="Total de vagas analisadas", value=len(dataframe))
    col_skills.metric(label="Skills monitoradas", value=detected_skills)
    col_locations.metric(label="Localizações distintas", value=unique_locations)


def render_top_skills_chart(top_skills: pd.DataFrame) -> None:
    """Render bar chart for most demanded skills."""
    st.subheader("Top 10 Skills mais exigidas")
    if top_skills.empty:
        st.info("No skill data available for the current filter selection.")
        return

    chart_data = top_skills.set_index("skill")["count"]
    st.bar_chart(chart_data, height=400)


def render_location_section(location_distribution: pd.DataFrame) -> None:
    """Render location distribution chart and supporting table."""
    st.subheader("Distribuição de vagas por localização")
    if location_distribution.empty:
        st.info("No location data available for the current filter selection.")
        return

    chart_data = location_distribution.set_index("location")["job_count"]
    st.bar_chart(chart_data, height=400)
    st.dataframe(location_distribution, use_container_width=True, hide_index=True)


def render_raw_data_table(dataframe: pd.DataFrame) -> None:
    """Display filtered raw records at the bottom of the page."""
    st.subheader("Dados filtrados")
    st.dataframe(dataframe, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Tech Job Market Insights",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Tech Job Market Insights")
    st.caption("Dashboard interativo alimentado pelo pipeline ETL (DuckDB).")

    try:
        jobs_df = load_jobs_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    filtered_df = apply_sidebar_filters(jobs_df)
    top_skills_df = compute_top_skills(filtered_df, SKILL_COLUMNS, top_n=10)
    location_df = compute_location_distribution(filtered_df)

    render_kpis(filtered_df)
    st.divider()

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        render_top_skills_chart(top_skills_df)
    with table_col:
        st.markdown("##### Resumo rápido")
        if not top_skills_df.empty:
            st.table(top_skills_df.reset_index(drop=True))
        else:
            st.write("Sem dados de skills.")

    st.divider()
    render_location_section(location_df)
    st.divider()
    render_raw_data_table(filtered_df)


if __name__ == "__main__":
    main()
