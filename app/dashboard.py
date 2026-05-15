"""Streamlit dashboard for Tech Job Market pipeline insights."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Set, cast

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    layout="wide",
    page_title="Tech Job Market",
    page_icon="📊",
)

st.markdown(
    """
    <style>
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e8edf3;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(30, 41, 59, 0.08);
            padding: 1.1rem 0.8rem;
            text-align: center;
            min-height: 110px;
        }
        .kpi-label {
            color: #64748b;
            font-size: 0.92rem;
            font-weight: 500;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            color: #2E86C1;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0;
        }
        .section-title {
            margin-top: 0.2rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join("data", "processed", "tech_jobs.duckdb")
JOBS_TABLE_NAME = "jobs"
TOP_SKILLS_LIMIT = 10
PRIMARY_COLOR = "#2E86C1"
SECONDARY_COLORS = ["#2E86C1", "#3498DB", "#5DADE2", "#85C1E9", "#AED6F1"]

SENIORITY_ORDER: List[str] = [
    "Estágio/Trainee",
    "Assistente",
    "Junior",
    "Pleno",
    "Sênior",
    "Especialista",
    "Gestão",
    "Não Informado",
]

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
    dataframe: pd.DataFrame, top_n: Optional[int] = TOP_SKILLS_LIMIT
) -> pd.DataFrame:
    """Count skill frequency from comma-separated `skills` column."""
    if dataframe.empty or "skills" not in dataframe.columns:
        return pd.DataFrame(columns=["skill", "count"])

    exploded_skills = dataframe["skills"].apply(_split_skills).explode().dropna()
    if exploded_skills.empty:
        return pd.DataFrame(columns=["skill", "count"])

    skill_counts = exploded_skills.value_counts().reset_index()
    skill_counts.columns = ["skill", "count"]
    if top_n is None:
        return skill_counts
    return skill_counts.head(top_n)


def compute_seniority_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count jobs by seniority level."""
    if dataframe.empty or "seniority" not in dataframe.columns:
        return pd.DataFrame(columns=["seniority", "job_count"])

    distribution = (
        dataframe["seniority"]
        .fillna("Não Informado")
        .astype(str)
        .str.strip()
        .replace("", "Não Informado")
        .value_counts()
        .reset_index()
    )
    distribution.columns = ["seniority", "job_count"]

    order_map = {level: index for index, level in enumerate(SENIORITY_ORDER)}

    def _seniority_sort_key(value: object) -> int:
        return order_map.get(str(value), len(SENIORITY_ORDER))

    distribution["sort_key"] = distribution["seniority"].map(_seniority_sort_key)
    return distribution.sort_values("sort_key").drop(columns=["sort_key"])


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
    """Apply sidebar filters and return filtered dataset."""
    st.sidebar.header("Filtros")

    category_options = _sorted_unique_values(dataframe["job_category"])
    location_options = _sorted_unique_values(dataframe["location_normalized"])
    seniority_options = _sorted_unique_values(dataframe["seniority"])

    selected_categories = st.sidebar.multiselect(
        "Categoria",
        options=category_options,
        default=[],
        placeholder="Todas",
    )
    selected_locations = st.sidebar.multiselect(
        "Localização",
        options=location_options,
        default=[],
        placeholder="Todas",
    )
    selected_seniorities = st.sidebar.multiselect(
        "Nível de Senioridade",
        options=seniority_options,
        default=[],
        placeholder="Todos",
    )

    filtered = dataframe.copy()
    if selected_categories:
        filtered = filtered[filtered["job_category"].isin(selected_categories)]
    if selected_locations:
        filtered = filtered[filtered["location_normalized"].isin(selected_locations)]
    if selected_seniorities:
        filtered = filtered[filtered["seniority"].isin(selected_seniorities)]

    all_skills_df = compute_skill_frequency(filtered, top_n=None)
    skill_options = all_skills_df["skill"].tolist()

    selected_skills = st.sidebar.multiselect(
        "Skill Específica",
        options=skill_options,
        default=[],
        placeholder="Todas",
        help="Lista recalculada conforme os filtros acima.",
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


def _style_plotly_figure(fig: go.Figure) -> go.Figure:
    """Apply transparent background and clean layout."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13, color="#334155"),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False),
    )
    return fig


def _render_kpi_card(column: Any, label: str, value: int) -> None:
    """Render one KPI card using custom HTML/CSS."""
    column.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value">{value:,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(dataframe: pd.DataFrame, all_skills_df: pd.DataFrame) -> None:
    """Render KPI cards in a four-column layout."""
    st.markdown('<h3 class="section-title">📌 Métricas Principais</h3>', unsafe_allow_html=True)

    total_jobs = len(dataframe)
    total_categories = dataframe["job_category"].nunique() if not dataframe.empty else 0
    total_locations = (
        dataframe["location_normalized"].nunique() if not dataframe.empty else 0
    )
    monitored_skills = len(all_skills_df) if not all_skills_df.empty else 0

    col_1, col_2, col_3, col_4 = st.columns(4)
    _render_kpi_card(col_1, "Total de Vagas", total_jobs)
    _render_kpi_card(col_2, "Categorias", total_categories)
    _render_kpi_card(col_3, "Localidade", total_locations)
    _render_kpi_card(col_4, "Skills Monitoradas", monitored_skills)


def render_main_charts(
    top_skills_df: pd.DataFrame,
    all_skills_df: pd.DataFrame,
    seniority_df: pd.DataFrame,
) -> None:
    """Render top skills chart, full skills ranking and seniority donut chart."""
    st.markdown('<h3 class="section-title">📈 Panorama do Mercado</h3>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns([6, 4])

    with chart_left:
        st.markdown("##### 🚀 Top 10 Skills")
        if top_skills_df.empty:
            st.info("Nenhuma skill encontrada para os filtros atuais.")
        else:
            skills_sorted = top_skills_df.sort_values("count", ascending=True)
            skills_fig = px.bar(
                skills_sorted,
                x="count",
                y="skill",
                orientation="h",
                text="count",
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            skills_fig.update_traces(textposition="outside", cliponaxis=False)
            skills_fig.update_layout(
                title="Demanda por skill",
                showlegend=False,
                xaxis_title="Quantidade de vagas",
                yaxis_title="",
            )
            st.plotly_chart(_style_plotly_figure(skills_fig), use_container_width=True)

        st.markdown("##### Ranking Completo de Skills")
        if all_skills_df.empty:
            st.info("Nenhuma skill encontrada para montar o ranking completo.")
        else:
            skills_ranking_df = all_skills_df.rename(
                columns={"skill": "Skill", "count": "Quantidade de Vagas"}
            )
            styled_ranking = (
                skills_ranking_df.style.background_gradient(
                    cmap="Blues",
                    subset=cast(Any, pd.IndexSlice[:, ["Quantidade de Vagas"]]),
                )
                .format({"Quantidade de Vagas": "{:.0f}"})
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#2E86C1"),
                                ("color", "white"),
                                ("font-weight", "600"),
                                ("text-align", "center"),
                            ],
                        },
                        {"selector": "td", "props": [("text-align", "center")]},
                    ]
                )
            )
            st.dataframe(styled_ranking, use_container_width=True, hide_index=True)

    with chart_right:
        st.markdown("##### 🎯 Distribuição de Senioridade")
        if seniority_df.empty:
            st.info("Sem dados de senioridade para os filtros atuais.")
        else:
            donut_fig = px.pie(
                seniority_df,
                names="seniority",
                values="job_count",
                hole=0.4,
                color_discrete_sequence=SECONDARY_COLORS,
            )
            donut_fig.update_traces(textposition="inside", textinfo="percent+label")
            donut_fig.update_layout(title="Composição por nível")
            st.plotly_chart(_style_plotly_figure(donut_fig), use_container_width=True)


def render_seniority_crosstab(crosstab_df: pd.DataFrame) -> None:
    """Render styled category vs seniority matrix."""
    st.markdown(
        '<h3 class="section-title">🧩 Distribuição por Categoria e Senioridade</h3>',
        unsafe_allow_html=True,
    )

    if crosstab_df.empty:
        st.info("Sem dados suficientes para gerar a tabela cruzada.")
        return

    display_df = crosstab_df.reset_index()
    numeric_columns = [col for col in display_df.columns if col != "Categoria da Vaga"]

    styled_table = (
        display_df.style.background_gradient(
            cmap="Blues", subset=cast(Any, numeric_columns), axis=None
        )
        .format({col: "{:.0f}" for col in numeric_columns})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#2E86C1"),
                        ("color", "white"),
                        ("font-weight", "600"),
                        ("text-align", "center"),
                    ],
                },
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
    )

    st.dataframe(styled_table, use_container_width=True, hide_index=True)


def render_raw_data_expander(dataframe: pd.DataFrame) -> None:
    """Render raw jobs table inside a collapsible expander."""
    with st.expander("Consultar Dados Brutos das Vagas", expanded=False):
        if dataframe.empty:
            st.warning("Nenhuma vaga encontrada com os filtros atuais.")
            return

        available_columns = [col for col in RAW_TABLE_COLUMNS if col in dataframe.columns]
        display_df = dataframe[available_columns].copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "job_title": st.column_config.TextColumn("Título"),
                "company_name": st.column_config.TextColumn("Empresa"),
                "job_category": st.column_config.TextColumn("Categoria"),
                "seniority": st.column_config.TextColumn("Senioridade"),
                "location": st.column_config.TextColumn("Local original"),
                "location_normalized": st.column_config.TextColumn("Local"),
                "skills": st.column_config.TextColumn("Skills"),
                "salary_min": st.column_config.NumberColumn("Salário mín.", format="%.0f"),
                "salary_max": st.column_config.NumberColumn("Salário máx.", format="%.0f"),
                "description": st.column_config.TextColumn("Descrição", width="large"),
            },
        )


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
    st.title("📊 Tech Job Market Insights")
    st.caption("Painel premium de inteligência de mercado para vagas de dados.")

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

    if filtered_df.empty:
        st.warning("Nenhuma vaga encontrada com os filtros selecionados. Ajuste os filtros na barra lateral.")

    all_skills_df = compute_skill_frequency(filtered_df, top_n=None)
    top_skills_df = all_skills_df.head(TOP_SKILLS_LIMIT)
    seniority_df = compute_seniority_distribution(filtered_df)
    seniority_crosstab_df = build_seniority_crosstab(filtered_df)

    render_kpi_row(filtered_df, all_skills_df)
    st.divider()
    render_main_charts(top_skills_df, all_skills_df, seniority_df)
    st.divider()
    render_seniority_crosstab(seniority_crosstab_df)
    st.divider()
    render_raw_data_expander(filtered_df)


if __name__ == "__main__":
    main()
