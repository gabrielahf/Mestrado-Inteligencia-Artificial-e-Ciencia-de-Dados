# app.py

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from pathlib import Path
import plotly.io as pio
import json

from styles import *
# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "data" / "processed" / "eda"

# ==========================================================
# CONFIGURAÇÃO DOS GRÁFICOS
# ==========================================================

GRAPH_METADATA = {
    "scatter_gasto_atividade_raw": {
        "title": "Relação entre Gastos e Atividade Parlamentar",
        "description": (
            "Deputados acima da linha de tendência podem indicar "
            "alto custo proporcional à atividade parlamentar."
        ),
    },
    "heatmap_correlacoes": {
        "title": "Correlações entre Indicadores Parlamentares",
        "description": (
            "Mostra relações entre gastos, atividade, despesas "
            "e indicadores de desempenho."
        ),
    },
    "bar_top_partidos_gasto_total": {
        "title": "Partidos com Maior Volume de Gastos",
        "description": ("Comparação entre os partidos com maior gasto agregado."),
    },
    "boxplot_gasto_por_partido_top10": {
        "title": "Distribuição de Gastos por Partido",
        "description": ("Permite identificar dispersão, consistência e outliers."),
    },
    "bar_top_ufs_gasto_medio": {
        "title": "UFs com Maior Gasto Médio",
        "description": ("Comparação do gasto médio entre estados."),
    },
    "stacked_area_partido_tempo": {
        "title": "Evolução Temporal dos Gastos",
        "description": ("Evolução dos gastos parlamentares ao longo do tempo."),
    },
    "mapa_choropleth_gastos_uf": {
        "title": "Distribuição Geográfica dos Gastos",
        "description": ("Mapa dos gastos parlamentares agregados por estado."),
    },
    "heatmap_partido_categoria_log": {
        "title": "Composição de Gastos por Categoria",
        "description": (
            "Identifica quais categorias concentram despesas em cada partido."
        ),
    },
    "treemap_partido_categoria": {
        "title": "Hierarquia de Gastos Parlamentares",
        "description": (
            "Visualização hierárquica das despesas por partido, categoria e deputado."
        ),
    },
    "parallel_coordinates_perfis": {
        "title": "Perfis Multivariados dos Partidos",
        "description": ("Comparação simultânea entre múltiplas métricas."),
    },
    "heatmap_clusterizado_partidos": {
        "title": "Clusterização de Perfis Parlamentares",
        "description": ("Agrupa partidos com comportamento semelhante."),
    },
}

# ==========================================================
# LOAD FIGURES
# ==========================================================

json_files = sorted(EDA_DIR.glob("*.json"))

figures = {}

for json_file in json_files:
    try:
        fig_dict = json.loads(json_file.read_text())

        if "data" not in fig_dict:
            continue

        fig = pio.from_json(json.dumps(fig_dict))

        figures[json_file.stem] = fig

    except Exception:
        continue

# ==========================================================
# APP
# ==========================================================

app = dash.Dash(__name__)

# ==========================================================
# HELPERS
# ==========================================================


def build_kpi_card(icon, title, value_id, value="0"):

    return html.Div(
        [
            html.Div(
                [
                    html.Div(icon, style=KPI_ICON),
                    html.Div(
                        [
                            html.Div(title, style=KPI_LABEL),
                            html.Div(value, id=value_id, style=KPI_VALUE),
                        ],
                        style=KPI_TEXT_CONTAINER,
                    ),
                ],
                style=KPI_CARD_CONTENT,
            )
        ],
        style=KPI_CARD,
    )


def create_graph_card(graph_key):

    if graph_key not in figures:
        return html.Div()

    meta = GRAPH_METADATA.get(graph_key, {})

    return html.Div(
        [
            html.Div(
                [
                    html.H3(meta.get("title", graph_key), style=GRAPH_TITLE),
                    html.P(meta.get("description", ""), style=GRAPH_DESCRIPTION),
                ]
            ),
            dcc.Graph(
                figure=figures[graph_key],
                style={"height": "500px"},
                config={"displayModeBar": False},
            ),
        ],
        style=GRAPH_CARD,
    )


def build_section(title, description, graph_keys):

    return html.Div(
        [
            html.Div(
                [
                    html.H2(title, style=SECTION_TITLE),
                    html.P(description, style=SECTION_DESCRIPTION),
                ],
                style=SECTION_HEADER,
            ),
            html.Div([create_graph_card(g) for g in graph_keys], style=GRAPH_GRID),
        ],
        style=SECTION_CONTAINER,
    )


# ==========================================================
# SIDEBAR
# ==========================================================

sidebar = html.Div(
    [
        html.Div("Radar Parlamentar", style=SIDEBAR_TITLE),
        html.Div(
            "Plataforma de análise de gastos parlamentares.", style=SIDEBAR_SUBTITLE
        ),
        html.Div(
            [
                html.Div("NAVEGAÇÃO", style=NAV_TITLE),
                html.A(
                    "Indicadores Parlamentares",
                    href="#indicadores-parlamentares",
                    style=NAV_LINK,
                ),
                html.A("Análise por Partido", href="#analise-partido", style=NAV_LINK),
                # criar graficos em que eh possivel comparar os deputados e no que ele mais gastou
                
                html.A("Análise Regional", href="#analise-regional", style=NAV_LINK),
                html.A("Evolução Temporal", href="#evolucao-temporal", style=NAV_LINK),
                html.A(
                    "Investigação Avançada",
                    href="#investigacao-avancada",
                    style=NAV_LINK,
                ),
            ],
            style=NAV_SECTION,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# ==========================================================
# LAYOUT
# ==========================================================

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        sidebar,
        html.Div(
            [
                # ==================================================
                # INDICADORES
                # ==================================================
                html.Div(
                    build_section(
                        "Indicadores Parlamentares",
                        "Análise da eficiência e correlação entre gastos e atividade legislativa.",
                        [
                            "scatter_gasto_atividade_raw",
                            "heatmap_correlacoes",
                        ],
                    ),
                    id="indicadores-parlamentares",
                ),
                # ==================================================
                # PARTIDOS
                # ==================================================
                html.Div(
                    build_section(
                        "Análise por Partido",
                        "Comparação entre partidos políticos e padrões de despesa.",
                        [
                            "bar_top_partidos_gasto_total",
                            "boxplot_gasto_por_partido_top10",
                        ],
                    ),
                    id="analise-partido",
                ),
                # ==================================================
                # REGIONAL
                # ==================================================
                html.Div(
                    build_section(
                        "Análise Regional",
                        "Distribuição geográfica e comparação entre unidades federativas.",
                        ["bar_top_ufs_gasto_medio", "mapa_choropleth_gastos_uf"],
                    ),
                    id="analise-regional",
                ),
                # ==================================================
                # DEPUTADOS
                # ==================================================
                # html.Div(
                #     build_section(
                #         "Análise por Deputados",
                #         "TESTE.",
                #         [
                #             "scatter_performance_deputado"
                #         ]
                #     ),
                #     id="analise-regional"
                # ),
                # ==================================================
                # TEMPORAL
                # ==================================================
                html.Div(
                    build_section(
                        "Evolução Temporal",
                        "Mudanças nos padrões de gastos ao longo do tempo.",
                        ["stacked_area_partido_tempo"],
                    ),
                    id="evolucao-temporal",
                ),
                # ==================================================
                # AVANÇADO
                # ==================================================
                html.Div(
                    build_section(
                        "Investigação Avançada",
                        "Análises multivariadas, clustering e exploração de padrões complexos.",
                        [
                            "heatmap_partido_categoria_log",
                            "treemap_partido_categoria",
                            "parallel_coordinates_perfis",
                            "heatmap_clusterizado_partidos",
                        ],
                    ),
                    id="investigacao-avancada",
                ),
            ],
            style=CONTENT_STYLE,
        ),
    ],
    style=APP_STYLE,
)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
