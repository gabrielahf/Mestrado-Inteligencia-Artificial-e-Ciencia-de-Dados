from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from urllib.request import urlopen

# ==========================================================
# Configuração
# ==========================================================
# Este script gera um conjunto de análises exploratórias (EDA)
# em Plotly, para facilitar a reutilização posterior no Dash.
# Mantém os gráficos já existentes e adiciona novos gráficos
# mais expressivos inspirados em boas práticas de chart selection.

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
EDA_DIR = DATA_PROCESSED / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PARQUET = DATA_PROCESSED / "dataset_mestre_clean.parquet"
DATASET_CSV = DATA_PROCESSED / "dataset_mestre_clean.csv"
DESP_CAT_CSV = DATA_PROCESSED / "despesas_categorizadas.csv"

# Tema simples e consistente para facilitar futura integração com Dash
PLOTLY_TEMPLATE = "plotly_white"

# Cores básicas inspiradas em boas práticas de legibilidade
COLOR_PRIMARY = "#4C78A8"
COLOR_SECONDARY = "#F58518"
COLOR_TERTIARY = "#54A24B"
COLOR_ACCENT = "#E45756"


# ==========================================================
# Carregamento
# ==========================================================


def load_dataset() -> pd.DataFrame:
    """Carrega o dataset mestre limpo, priorizando o parquet."""
    if DATASET_PARQUET.exists():
        return pd.read_parquet(DATASET_PARQUET)
    if DATASET_CSV.exists():
        return pd.read_csv(DATASET_CSV)
    raise FileNotFoundError("Nenhum dataset mestre encontrado em data/processed/.")


def load_categorized_expenses() -> pd.DataFrame | None:
    """Carrega o CSV de despesas categorizadas, caso exista."""
    if DESP_CAT_CSV.exists():
        return pd.read_csv(DESP_CAT_CSV)
    return None


# ==========================================================
# Preparação básica
# ==========================================================


def prepare_master_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante tipos numéricos e cria colunas auxiliares para a EDA.
    Mantemos a lógica simples para facilitar a explicação.
    """
    df = df.copy()

    numeric_cols = [
        "gasto_total",
        "gasto_liquido",
        "qtd_despesas",
        "qtd_estornos",
        "total_proposicoes",
        "total_eventos",
        "atividade_composta",
        "gasto_total_ajustado",
        "custo_por_atividade",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Recria atividade_composta se necessário
    if "atividade_composta" not in df.columns:
        df["atividade_composta"] = pd.to_numeric(
            df.get("total_proposicoes", 0), errors="coerce"
        ).fillna(0.0) + 0.3 * pd.to_numeric(
            df.get("total_eventos", 0), errors="coerce"
        ).fillna(0.0)

    # Transformação log para gasto total (útil em distribuições assimétricas)
    if "gasto_total" in df.columns:
        df["log_gasto_total"] = np.log1p(df["gasto_total"].clip(lower=0.0))
    else:
        df["log_gasto_total"] = 0.0

    # Remove infinitos em métricas derivadas
    if "custo_por_atividade" in df.columns:
        df["custo_por_atividade"] = (
            df["custo_por_atividade"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    # Preenche metadados principais, se existirem
    for col in ["nome", "siglaPartido", "siglaUf"]:
        if col in df.columns:
            df[col] = df[col].fillna("Desconhecido")

    # Faixas simples de gasto para algumas comparações exploratórias
    if "gasto_total" in df.columns and len(df) > 0:
        try:
            df["faixa_gasto"] = pd.qcut(
                df["gasto_total"].rank(method="first"),
                q=4,
                labels=["Baixo", "Médio-baixo", "Médio-alto", "Alto"],
            )
        except Exception:
            df["faixa_gasto"] = "Única"

    return df


# ==========================================================
# Utilitários de exportação
# ==========================================================


def save_plotly_figure(fig: go.Figure, name: str) -> None:
    """
    Salva a figura em HTML interativo, JSON (para app.py) e tenta PNG estático.
    """
    html_path = EDA_DIR / f"{name}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")

    json_path = EDA_DIR / f"{name}.json"
    json_path.write_text(pio.to_json(fig), encoding="utf-8")

    png_path = EDA_DIR / f"{name}.png"
    try:
        fig.write_image(png_path, scale=2, width=1200, height=700)
    except Exception:
        # Se kaleido não estiver disponível, o HTML já é suficiente para a EDA.
        pass


def export_summary(df: pd.DataFrame) -> None:
    """Salva um resumo numérico simples para apoiar o relatório."""
    summary = {
        "n_deputados": int(len(df)),
        "gasto_total": {
            "media": float(df["gasto_total"].mean())
            if "gasto_total" in df.columns
            else 0.0,
            "mediana": float(df["gasto_total"].median())
            if "gasto_total" in df.columns
            else 0.0,
            "q1": float(df["gasto_total"].quantile(0.25))
            if "gasto_total" in df.columns
            else 0.0,
            "q3": float(df["gasto_total"].quantile(0.75))
            if "gasto_total" in df.columns
            else 0.0,
            "max": float(df["gasto_total"].max())
            if "gasto_total" in df.columns
            else 0.0,
        },
        "atividade_composta": {
            "media": float(df["atividade_composta"].mean())
            if "atividade_composta" in df.columns
            else 0.0,
            "mediana": float(df["atividade_composta"].median())
            if "atividade_composta" in df.columns
            else 0.0,
            "max": float(df["atividade_composta"].max())
            if "atividade_composta" in df.columns
            else 0.0,
        },
        "custo_por_atividade": {
            "media": float(df["custo_por_atividade"].mean())
            if "custo_por_atividade" in df.columns
            else 0.0,
            "mediana": float(df["custo_por_atividade"].median())
            if "custo_por_atividade" in df.columns
            else 0.0,
            "q3": float(df["custo_por_atividade"].quantile(0.75))
            if "custo_por_atividade" in df.columns
            else 0.0,
            "max": float(df["custo_por_atividade"].max())
            if "custo_por_atividade" in df.columns
            else 0.0,
        },
    }

    (EDA_DIR / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ==========================================================
# Helpers para as visualizações “spicy”
# ==========================================================


def _positive_value_column(df: pd.DataFrame) -> str | None:
    """Define a melhor coluna monetária positiva para análises de despesa."""
    if "valorLiquido_pos" in df.columns:
        df["valorLiquido_pos"] = pd.to_numeric(
            df["valorLiquido_pos"], errors="coerce"
        ).fillna(0.0)
        return "valorLiquido_pos"
    if "valorLiquido" in df.columns:
        df["valorLiquido"] = pd.to_numeric(df["valorLiquido"], errors="coerce").fillna(
            0.0
        )
        df["valor_eda"] = df["valorLiquido"].clip(lower=0.0)
        return "valor_eda"
    return None


def _pick_time_column(df: pd.DataFrame) -> str | None:
    """Escolhe uma coluna temporal anual disponível para análise ao longo do tempo."""
    for col in ["anoConsulta", "anoDocumento", "ano"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            return col
    return None


def _build_party_category_matrix(
    df_mestre: pd.DataFrame, df_desp_cat: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Monta a matriz partido x categoria usada no heatmap e em outras análises."""
    base = df_desp_cat.copy()
    base["idDeputado"] = pd.to_numeric(base["idDeputado"], errors="coerce")
    base = base.dropna(subset=["idDeputado"]).copy()
    base["idDeputado"] = base["idDeputado"].astype(int)

    valor_col = _positive_value_column(base)
    if valor_col is None:
        return pd.DataFrame()

    meta = df_mestre[
        [c for c in ["idDeputado", "siglaPartido"] if c in df_mestre.columns]
    ].drop_duplicates()
    base = base.merge(meta, on="idDeputado", how="left")
    base["siglaPartido"] = base["siglaPartido"].fillna("Desconhecido")

    top_parties = (
        base.groupby("siglaPartido")[valor_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    matriz = (
        base[base["siglaPartido"].isin(top_parties)]
        .groupby(["siglaPartido", "categoria_macro"])[valor_col]
        .sum()
        .unstack(fill_value=0.0)
    )
    return matriz


# ==========================================================
# 1) Distribuições e outliers (existentes)
# ==========================================================


def plot_histograms(df: pd.DataFrame) -> None:
    """Gera histogramas para gasto total e log(gasto total)."""
    if "gasto_total" not in df.columns:
        return

    valores = df["gasto_total"].clip(lower=0.0)

    fig = px.histogram(
        df.assign(gasto_total_plot=valores),
        x="gasto_total_plot",
        nbins=30,
        template=PLOTLY_TEMPLATE,
        title="Distribuição do gasto total",
        labels={"gasto_total_plot": "Gasto total (R$)", "count": "Frequência"},
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    fig.add_vline(
        x=valores.mean(),
        line_dash="dash",
        line_color=COLOR_ACCENT,
        annotation_text="Média",
    )
    fig.add_vline(
        x=valores.median(),
        line_dash="dash",
        line_color=COLOR_TERTIARY,
        annotation_text="Mediana",
    )
    fig.update_layout(bargap=0.05)
    save_plotly_figure(fig, "hist_gasto_total")


def plot_boxplots(df: pd.DataFrame) -> None:
    """Mostra outliers e dispersão em três métricas-chave."""
    metricas = [
        ("gasto_total", "Gasto total"),
        ("atividade_composta", "Atividade composta"),
        ("custo_por_atividade", "Custo por atividade"),
    ]
    metricas_validas = [c for c, _ in metricas if c in df.columns]
    if not metricas_validas:
        return

    dados = df[metricas_validas].melt(var_name="métrica", value_name="valor")
    dados["valor"] = dados["valor"].replace([np.inf, -np.inf], np.nan)
    dados = dados.dropna(subset=["valor"])

    fig = px.box(
        dados,
        x="métrica",
        y="valor",
        points="outliers",
        template=PLOTLY_TEMPLATE,
        title="Boxplots de métricas principais",
        color="métrica",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False)
    save_plotly_figure(fig, "boxplots_metricas")


# ==========================================================
# 2) Comparações por grupos (existentes)
# ==========================================================


def plot_parties(df: pd.DataFrame, top_n: int = 10) -> None:
    """Compara partidos por gasto total e distribuição de gasto."""
    if "siglaPartido" not in df.columns or "gasto_total" not in df.columns:
        return

    gasto_partido = (
        df.groupby("siglaPartido", as_index=False)["gasto_total"]
        .sum()
        .sort_values("gasto_total", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        gasto_partido,
        x="siglaPartido",
        y="gasto_total",
        template=PLOTLY_TEMPLATE,
        title=f"Top {top_n} partidos por gasto total",
        labels={"siglaPartido": "Partido", "gasto_total": "Gasto total (R$)"},
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    save_plotly_figure(fig, "bar_top_partidos_gasto_total")

    partidos_top = gasto_partido["siglaPartido"].tolist()
    df_top = df[df["siglaPartido"].isin(partidos_top)].copy()
    fig = px.box(
        df_top,
        x="siglaPartido",
        y="gasto_total",
        points="outliers",
        category_orders={"siglaPartido": partidos_top},
        template=PLOTLY_TEMPLATE,
        title=f"Distribuição do gasto total nos top {top_n} partidos",
        labels={"siglaPartido": "Partido", "gasto_total": "Gasto total (R$)"},
        color="siglaPartido",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    save_plotly_figure(fig, "boxplot_gasto_por_partido_top10")


def plot_ufs(df: pd.DataFrame, top_n: int = 15) -> None:
    """Compara UFs pelo gasto médio por deputado."""
    if "siglaUf" not in df.columns or "gasto_total" not in df.columns:
        return

    uf_media = (
        df.groupby("siglaUf", as_index=False)["gasto_total"]
        .mean()
        .rename(columns={"gasto_total": "gasto_medio"})
        .sort_values("gasto_medio", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        uf_media,
        x="siglaUf",
        y="gasto_medio",
        template=PLOTLY_TEMPLATE,
        title=f"Top {top_n} UFs por gasto médio por deputado",
        labels={"siglaUf": "UF", "gasto_medio": "Gasto médio (R$)"},
        color_discrete_sequence=[COLOR_TERTIARY],
    )
    save_plotly_figure(fig, "bar_top_ufs_gasto_medio")


# ==========================================================
# 3) Relações entre variáveis (existentes)
# ==========================================================


def plot_scatter_gasto_atividade(df: pd.DataFrame, n_rotulos: int = 8) -> None:
    """
    Scatter principal da EDA.
    Usa log_gasto_total no eixo X para reduzir assimetria.
    Destaca alguns outliers com maior custo por atividade.
    """
    if "log_gasto_total" not in df.columns or "atividade_composta" not in df.columns:
        return

    hover_cols = [
        c
        for c in [
            "nome",
            "siglaPartido",
            "siglaUf",
            "gasto_total",
            "atividade_composta",
            "custo_por_atividade",
        ]
        if c in df.columns
    ]

    fig = px.scatter(
        df,
        x="log_gasto_total",
        y="atividade_composta",
        hover_data=hover_cols,
        template=PLOTLY_TEMPLATE,
        title="Relação entre gasto total e atividade composta",
        labels={
            "log_gasto_total": "log(1 + gasto total)",
            "atividade_composta": "Atividade composta",
        },
        opacity=0.7,
    )
    fig.update_traces(
        marker=dict(color=COLOR_PRIMARY, size=8, line=dict(color="white", width=0.5))
    )

    x = df["log_gasto_total"].to_numpy(dtype=float)
    y = df["atividade_composta"].to_numpy(dtype=float)
    if len(df) >= 2 and np.std(x) > 0:
        coef = np.polyfit(x, y, deg=1)
        xp = np.linspace(x.min(), x.max(), 100)
        yp = coef[0] * xp + coef[1]
        fig.add_trace(
            go.Scatter(
                x=xp,
                y=yp,
                mode="lines",
                name="Tendência linear",
                line=dict(color=COLOR_ACCENT, dash="dash"),
            )
        )

    if "custo_por_atividade" in df.columns and "nome" in df.columns:
        candidatos = (
            df[df["atividade_composta"] > 0]
            .sort_values("custo_por_atividade", ascending=False)
            .head(n_rotulos)
        )
        fig.add_trace(
            go.Scatter(
                x=candidatos["log_gasto_total"],
                y=candidatos["atividade_composta"],
                mode="text",
                text=candidatos["nome"],
                textposition="top center",
                showlegend=False,
                textfont=dict(size=10, color="#333333"),
            )
        )

    save_plotly_figure(fig, "scatter_gasto_atividade_log")


def plot_scatter_gasto_atividade_raw(df: pd.DataFrame, n_rotulos: int = 8) -> None:
    """
    Scatter principal da EDA com gasto_total BRUTO no eixo X.
    Sem transformações log — valores diretos em R$ para interpretação imediata.
    """
    if "gasto_total" not in df.columns or "atividade_composta" not in df.columns:
        return

    hover_cols = [
        c
        for c in [
            "nome",
            "siglaPartido",
            "siglaUf",
            "gasto_total",
            "atividade_composta",
            "custo_por_atividade",
        ]
        if c in df.columns
    ]

    # Filtrar valores válidos
    df_plot = df[df["gasto_total"] >= 0].copy()
    if df_plot.empty:
        return

    fig = px.scatter(
        df_plot,
        x="gasto_total",
        y="atividade_composta",
        hover_data=hover_cols,
        template=PLOTLY_TEMPLATE,
        title="Relação entre gasto total (R$) e atividade composta",
        labels={
            "gasto_total": "Gasto total (R$)",
            "atividade_composta": "Atividade composta",
        },
        opacity=0.7,
    )

    # Formatar eixo X para moeda brasileira
    fig.update_xaxes(tickformat=",.0f", title="Gasto total (R$)")

    # Formatar hover para mostrar valores legíveis
    fig.update_traces(
        marker=dict(color=COLOR_PRIMARY, size=8, line=dict(color="white", width=0.5)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Partido: %{customdata[1]} (%{customdata[2]})<br>"
            "Gasto total: <b>R$ %{x:,.2f}</b><br>"
            "Atividade composta: %{y:.2f}<br>"
            "Custo/atividade: R$ %{customdata[5]:,.2f}<br>"
            "<extra></extra>"
        ),
    )

    # Linha de tendência (opcional: calcular em log para estabilidade, plotar em raw)
    x_raw = df_plot["gasto_total"].replace(0, 1e-3).to_numpy(dtype=float)
    y = df_plot["atividade_composta"].to_numpy(dtype=float)

    if len(df_plot) >= 2 and np.std(x_raw) > 0:
        # Fit em escala log para evitar que outliers dominem a regressão
        x_log = np.log(x_raw)
        coef = np.polyfit(x_log, y, deg=1)

        # Gerar pontos para plotar em escala raw
        xp_raw = np.logspace(np.log10(x_raw.min()), np.log10(x_raw.max()), 100)
        xp_log = np.log(xp_raw)
        yp = coef[0] * xp_log + coef[1]

        fig.add_trace(
            go.Scatter(
                x=xp_raw,
                y=yp,
                mode="lines",
                name="Tendência",
                line=dict(color=COLOR_ACCENT, dash="dash"),
                hoverinfo="skip",
            )
        )

    # Destacar deputados com maior custo por atividade
    if "custo_por_atividade" in df.columns and "nome" in df.columns:
        candidatos = (
            df_plot[df_plot["atividade_composta"] > 0]
            .sort_values("custo_por_atividade", ascending=False)
            .head(n_rotulos)
        )
        fig.add_trace(
            go.Scatter(
                x=candidatos["gasto_total"],
                y=candidatos["atividade_composta"],
                mode="text",
                text=candidatos["nome"],
                textposition="top center",
                showlegend=False,
                textfont=dict(size=10, color="#333333"),
                hoverinfo="skip",
            )
        )

    save_plotly_figure(fig, "scatter_gasto_atividade_raw")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap simples de correlação entre variáveis numéricas principais."""
    cols = [
        c
        for c in [
            "gasto_total",
            "gasto_liquido",
            "qtd_despesas",
            "qtd_estornos",
            "total_proposicoes",
            "total_eventos",
            "atividade_composta",
            "custo_por_atividade",
        ]
        if c in df.columns
    ]
    if len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        title="Correlação entre métricas principais",
    )
    save_plotly_figure(fig, "heatmap_correlacoes")


# ==========================================================
# 4) Composição por partido (existente e também conta no lineup)
# ==========================================================


def plot_party_category_heatmap(
    df_mestre: pd.DataFrame, df_desp_cat: pd.DataFrame | None, top_n: int = 10
) -> None:
    if df_desp_cat is None:
        return
    if (
        "idDeputado" not in df_desp_cat.columns
        or "categoria_macro" not in df_desp_cat.columns
    ):
        return
    if "siglaPartido" not in df_mestre.columns:
        return

    matriz = _build_party_category_matrix(df_mestre, df_desp_cat, top_n=top_n)
    if matriz.empty:
        return

    # Apply log1p transformation for color scaling
    matriz_log = np.log1p(matriz)

    # Create figure with log-transformed values for coloring
    fig = px.imshow(
        matriz_log,
        color_continuous_scale="YlGnBu",
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        title="Heatmap: log(1 + gasto) por partido e macro-categoria",
        labels=dict(color="log(1 + gasto) (R$)"),
        # x/y labels will auto-use matriz columns/index
    )

    # ✅ Set customdata and hovertemplate via update_traces
    fig.update_traces(
        customdata=matriz,  # Original absolute values
        hovertemplate=(
            "<b>%{y}</b> → %{x}<br>"
            "Gasto: <b>R$ %{customdata:,.2f}</b><br>"
            "<extra></extra>"
        ),
    )

    # Optional: Add custom tick labels on colorbar to show original scale
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickvals=np.log1p([0, 1e5, 1e6, 1e7, 3.5e7]),
            ticktext=["0", "100k", "1M", "10M", "35M"],
            title="Gasto (R$)",
        )
    )

    save_plotly_figure(fig, "heatmap_partido_categoria_log")


# ==========================================================
# 5) Anomalias / rankings (existentes)
# ==========================================================


def export_anomaly_tables(df: pd.DataFrame, top_n: int = 15) -> None:
    """Exporta tabelas simples para discutir outliers e casos extremos no relatório."""
    cols_base = [
        c for c in ["idDeputado", "nome", "siglaPartido", "siglaUf"] if c in df.columns
    ]

    if "gasto_total" in df.columns:
        cols = cols_base + [
            c
            for c in ["gasto_total", "atividade_composta", "custo_por_atividade"]
            if c in df.columns
        ]
        df[cols].sort_values("gasto_total", ascending=False).head(top_n).to_csv(
            EDA_DIR / "anomalias_top_gasto_total.csv", index=False
        )

    if "custo_por_atividade" in df.columns:
        cols = cols_base + [
            c
            for c in ["custo_por_atividade", "gasto_total", "atividade_composta"]
            if c in df.columns
        ]
        (
            df[df.get("atividade_composta", 0) > 0][cols]
            .sort_values("custo_por_atividade", ascending=False)
            .head(top_n)
            .to_csv(EDA_DIR / "anomalias_top_custo_por_atividade.csv", index=False)
        )

    if "qtd_estornos" in df.columns:
        cols = cols_base + [
            c
            for c in ["qtd_estornos", "gasto_liquido", "gasto_total"]
            if c in df.columns
        ]
        df[cols].sort_values("qtd_estornos", ascending=False).head(top_n).to_csv(
            EDA_DIR / "anomalias_top_estornos.csv", index=False
        )


# ==========================================================
# 6 plots “spicy” adicionados
# ==========================================================


def plot_treemap_party_uf_deputado(df: pd.DataFrame, top_n: int = 120) -> None:
    """
    Treemap hierárquico para mostrar dominância de partidos, UFs e deputados.
    Tamanho = gasto_total; cor = custo_por_atividade.
    """
    required = {"siglaPartido", "siglaUf", "nome", "gasto_total"}
    if not required.issubset(df.columns):
        return

    base = df.copy()
    base = base.sort_values("gasto_total", ascending=False).head(top_n)
    color_col = (
        "custo_por_atividade"
        if "custo_por_atividade" in base.columns
        else "atividade_composta"
    )

    fig = px.treemap(
        base,
        path=[px.Constant("Câmara dos Deputados"), "siglaPartido", "siglaUf", "nome"],
        values="gasto_total",
        color=color_col,
        color_continuous_scale="RdYlBu_r",
        template=PLOTLY_TEMPLATE,
        title="Treemap: dominância de gasto por partido, UF e deputado",
        hover_data={
            "gasto_total": ":.2f",
            color_col: ":.2f",
            "atividade_composta": True
            if "atividade_composta" in base.columns
            else False,
        },
    )
    save_plotly_figure(fig, "treemap_partido_uf_deputado")


def plot_beeswarm_activity_by_party(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Versão tipo beeswarm/strip para mostrar a distribuição de atividade por partido.
    Útil quando a variável Y tem poucos níveis ou muita sobreposição.
    """
    if "siglaPartido" not in df.columns or "atividade_composta" not in df.columns:
        return

    top_parties = (
        df.groupby("siglaPartido")["gasto_total"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    base = df[df["siglaPartido"].isin(top_parties)].copy()
    color_col = "custo_por_atividade" if "custo_por_atividade" in base.columns else None

    fig = px.strip(
        base,
        x="siglaPartido",
        y="atividade_composta",
        color=color_col,
        hover_data=[
            c
            for c in ["nome", "siglaUf", "gasto_total", "custo_por_atividade"]
            if c in base.columns
        ],
        template=PLOTLY_TEMPLATE,
        title=f"Beeswarm / Strip: atividade composta nos top {top_n} partidos",
        labels={"siglaPartido": "Partido", "atividade_composta": "Atividade composta"},
        category_orders={"siglaPartido": top_parties},
    )
    fig.update_traces(
        jitter=0.45,
        marker=dict(size=8, opacity=0.75, line=dict(width=0.4, color="white")),
    )
    save_plotly_figure(fig, "beeswarm_atividade_por_partido")


def plot_parallel_coordinates_profiles(
    df: pd.DataFrame, top_n_parties: int = 10, agg: str = "median"
) -> None:
    """
    Parallel coordinates com UMA linha por partido (valores agregados).
    Substitui a versão anterior que plotava deputados individuais,
    tornando o gráfico ilegível com centenas de linhas sobrepostas.
    """
    cols = [
        c
        for c in [
            "gasto_total",
            "gasto_liquido",
            "qtd_despesas",
            "qtd_estornos",
            "custo_por_atividade",
        ]
        if c in df.columns
    ]
    if (
        len(cols) < 4
        or "siglaPartido" not in df.columns
        or "gasto_total" not in df.columns
    ):
        return

    top_parties = (
        df.groupby("siglaPartido")["gasto_total"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_parties)
        .index.tolist()
    )

    base = df[df["siglaPartido"].isin(top_parties)].copy()
    base["siglaPartido"] = base["siglaPartido"].replace({"PSL": "UNIÃO"})
    if base.empty:
        return

    agg_df = (
        base.groupby("siglaPartido")[cols]
        .agg(agg)
        .reindex(top_parties)
        .reset_index()
    )

    agg_df = agg_df[agg_df[cols].sum(axis=1) > 0].reset_index(drop=True)
    top_parties = agg_df["siglaPartido"].tolist()

    plot_df = agg_df.copy()
    original_df = agg_df.copy()  

    for col in cols:
        col_min = plot_df[col].min()
        col_max = plot_df[col].max()
        rng = col_max - col_min
        plot_df[col] = (plot_df[col] - col_min) / rng if rng > 0 else 0.0

    palette = px.colors.qualitative.Bold
    party_colors = {
        party: palette[i % len(palette)] for i, party in enumerate(top_parties)
    }

    label_map = {
        "gasto_total": "Gasto Total",
        "gasto_liquido": "Gasto Líquido",
        "qtd_despesas": "Qtd. Despesas",
        "qtd_estornos": "Qtd. Estornos",
        "total_proposicoes": "Total Proposições",
        "total_eventos": "Total Eventos",
        "atividade_composta": "Atividade Composta",
        "custo_por_atividade": "Custo por Atividade",
    }

    x_positions = list(range(len(cols)))
    axis_labels = [label_map.get(c, c) for c in cols]

    fig = go.Figure()

    for party in top_parties:
        row = plot_df[plot_df["siglaPartido"] == party].iloc[0]
        raw_row = original_df[original_df["siglaPartido"] == party].iloc[0]

        y_vals = [row[c] for c in cols]

        hover_parts = [f"<b>{party}</b>"]
        for col in cols:
            v = raw_row[col]
            if v >= 1_000_000:
                fmt = f"R$ {v / 1e6:.2f}M"
            elif v >= 1_000:
                fmt = f"{v:,.0f}"
            else:
                fmt = f"{v:.1f}"
            hover_parts.append(f"  {label_map.get(col, col)}: {fmt}")

        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_vals,
                mode="lines+markers",
                name=party,
                line=dict(color=party_colors[party], width=2.5),
                marker=dict(
                    color=party_colors[party],
                    size=8,
                    symbol="circle",
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
                opacity=0.85,
            )
        )

    for xp in x_positions:
        fig.add_vline(x=xp, line_color="#cccccc", line_width=1, layer="below")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=(
                "Perfis Multivariados dos Partidos"
                f"<br><sup>Agregação por {agg} · eixos normalizados [0–1]</sup>"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=x_positions,
            ticktext=axis_labels,
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=True,
            showgrid=True,
            gridcolor="#e8e8e8",
            title="Valor normalizado [0–1]",
            range=[-0.05, 1.05],
        ),
        legend=dict(
            title="Partido",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=60, r=220, t=100, b=60),
        hovermode="x unified",
        plot_bgcolor="white",
    )

    save_plotly_figure(fig, "parallel_coordinates_perfis")

def plot_stacked_area_party_over_time(
    df_mestre: pd.DataFrame, df_desp_cat: pd.DataFrame | None, top_n: int = 6
) -> None:
    """
    Área empilhada ao longo do tempo para mostrar a dominância de partidos.
    Usa despesas categorizadas, que preservam granularidade temporal.
    """
    if df_desp_cat is None:
        return
    if "idDeputado" not in df_desp_cat.columns:
        return

    base = df_desp_cat.copy()
    base["idDeputado"] = pd.to_numeric(base["idDeputado"], errors="coerce")
    base = base.dropna(subset=["idDeputado"]).copy()
    base["idDeputado"] = base["idDeputado"].astype(int)

    valor_col = _positive_value_column(base)
    time_col = _pick_time_column(base)
    if valor_col is None or time_col is None:
        return

    meta = df_mestre[
        [c for c in ["idDeputado", "siglaPartido"] if c in df_mestre.columns]
    ].drop_duplicates()
    base = base.merge(meta, on="idDeputado", how="left")
    base["siglaPartido"] = base["siglaPartido"].fillna("Desconhecido")
    base = base.dropna(subset=[time_col]).copy()
    base[time_col] = base[time_col].astype(int)

    base = base[base[time_col] < 2026].copy()
    base = base[base[time_col] > 2022].copy()

    top_parties = (
        base.groupby("siglaPartido")[valor_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    serie = (
        base[base["siglaPartido"].isin(top_parties)]
        .groupby([time_col, "siglaPartido"], as_index=False)[valor_col]
        .sum()
        .sort_values(time_col)
    )
    if serie.empty:
        return

    fig = px.area(
        serie,
        x=time_col,
        y=valor_col,
        color="siglaPartido",
        template=PLOTLY_TEMPLATE,
        title=f"Área empilhada: dominância de gasto por partido ao longo do tempo",
        labels={
            time_col: "Ano",
            valor_col: "Gasto (R$)",
            "siglaPartido": "Partido",
        },
    )

    fig.update_xaxes(
        tickmode="linear",
        dtick=1,
        tick0=int(serie[time_col].min()),
        tickformat="d",
    )
    save_plotly_figure(fig, "stacked_area_partido_tempo")


def _build_deputado_category_matrix(
    df_mestre: pd.DataFrame, df_desp_cat: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Monta a matriz partido x categoria usada no heatmap e em outras análises."""
    base = df_desp_cat.copy()
    base["idDeputado"] = pd.to_numeric(base["idDeputado"], errors="coerce")
    base = base.dropna(subset=["idDeputado"]).copy()
    base["idDeputado"] = base["idDeputado"].astype(int)

    valor_col = _positive_value_column(base)
    if valor_col is None:
        return pd.DataFrame()

    meta = df_mestre[
        [c for c in ["idDeputado", "siglaPartido", "nome"] if c in df_mestre.columns]
    ].drop_duplicates()
    base = base.merge(meta, on="idDeputado", how="left")
    base["siglaPartido"] = base["siglaPartido"].fillna("Desconhecido")

    top_parties = (
        base.groupby("siglaPartido")[valor_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    matriz = (
        base[base["siglaPartido"].isin(top_parties)]
        .groupby(["siglaPartido", "nome", "categoria_macro"])[valor_col]
        .sum()
        .unstack(fill_value=0.0)
    )
    return matriz


def plot_party_category_share_treemap(
    df_mestre: pd.DataFrame, df_desp_cat: pd.DataFrame | None, top_n: int = 10
) -> None:
    """
    Treemap alternativo focado na composição de despesa por partido e categoria.
    Ajuda a reforçar a leitura de parte-do-todo dentro de cada partido.
    """
    if df_desp_cat is None:
        return
    matriz = _build_deputado_category_matrix(df_mestre, df_desp_cat, top_n=top_n)
    if matriz.empty:
        return

    base = matriz.stack().reset_index(name="gasto_categoria")
    fig = px.treemap(
        base,
        path=[px.Constant("Despesas"), "siglaPartido", "categoria_macro", "nome"],
        values="gasto_categoria",
        color="gasto_categoria",
        color_continuous_scale="YlOrRd",
        template=PLOTLY_TEMPLATE,
        title="Treemap: composição de despesa por partido e categoria",
    )
    save_plotly_figure(fig, "treemap_partido_categoria")


def plot_clustered_heatmap_and_dendrogram(df: pd.DataFrame, top_n: int = 12) -> None:
    """
    Cria duas visualizações complementares:
      1) dendrograma hierárquico por partido;
      2) heatmap reordenado segundo clustering.
    Usamos médias por partido para manter a leitura legível.
    """
    if "siglaPartido" not in df.columns:
        return

    feature_cols = [
        c
        for c in [
            "gasto_total",
            "gasto_liquido",
            "qtd_despesas",
            "qtd_estornos",
            "total_proposicoes",
            "total_eventos",
            "atividade_composta",
            "custo_por_atividade",
        ]
        if c in df.columns
    ]
    if len(feature_cols) < 4:
        return

    top_parties = (
        df.groupby("siglaPartido")["gasto_total"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    base = (
        df[df["siglaPartido"].isin(top_parties)]
        .groupby("siglaPartido", as_index=True)[feature_cols]
        .mean()
        .fillna(0.0)
    )
    if len(base) < 3:
        return

    # Normalização simples por coluna para comparação relativa entre partidos
    norm = (base - base.mean()) / base.std(ddof=0).replace(0, 1)
    norm = norm.fillna(0.0)

    # Dendrograma com distância euclidiana e método ward
    fig_d = ff.create_dendrogram(
        norm.values,
        labels=norm.index.tolist(),
        linkagefun=lambda x: linkage(x, method="ward", metric="euclidean"),
        orientation="bottom",
    )
    fig_d.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Dendrograma: similaridade entre partidos (médias padronizadas)",
    )
    save_plotly_figure(fig_d, "dendrograma_partidos")

    # Heatmap reordenado pela ordem das folhas do dendrograma
    dist = pdist(norm.values, metric="euclidean")
    link = linkage(dist, method="ward")
    order = leaves_list(link)
    norm_ord = norm.iloc[order]

    fig_h = px.imshow(
        norm_ord,
        color_continuous_scale="RdBu_r",
        zmin=-2,
        zmax=2,
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        title="Heatmap clusterizado: perfis médios dos partidos",
        labels=dict(color="z-score"),
    )
    save_plotly_figure(fig_h, "heatmap_clusterizado_partidos")


# def plot_analise_por_deputado(df: pd.DataFrame, output_dir: Path, top_n: int = 10):
#     """
#     Gera a sessão de análise individual com dois gráficos:
#     1) Ranking de Eficiência (Gasto Normalizado / Proposição)
#     2) Scatter Plot de Clusters (Gasto Normalizado vs Atividade Composta)
#     """
#     # Preparação de dados
#     df = df.copy()
#     df['identificador'] = df['nome'] + " (" + df['siglaPartido'] + ")"
    
#     # 1. GRÁFICO DE BARRAS: Ranking de Eficiência
#     # Índice = Gasto Normalizado / (Propostas + 1) -> Menor é melhor.
#     # Vamos mostrar os 10 que MAIS gastam proporcionalmente à sua produção.
#     df['custo_proposta_norm'] = df['gasto_normalizado_dist'] / (df['total_proposicoes'] + 1)
#     top_ineficientes = df.sort_values('custo_proposta_norm', ascending=False).head(top_n)

#     fig_bar = px.bar(
#         top_ineficientes,
#         x='custo_proposta_norm',
#         y='identificador',
#         orientation='h',
#         color='custo_proposta_norm',
#         color_continuous_scale='Reds',
#         title=f"Top {top_n} Deputados: Maior Custo Normalizado por Proposição",
#         labels={'custo_proposta_norm': 'R$ (Norm) / Proposta', 'identificador': 'Deputado'}
#     )
#     fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, template=PLOTLY_TEMPLATE)
#     save_plotly_figure(fig_bar, "ranking_ineficiencia_deputado", output_dir)

#     # 2. SCATTER PLOT: Clusters de Performance
#     # Visualização da relação entre o que se gasta e o que se produz
#     fig_scatter = px.scatter(
#         df,
#         x="gasto_normalizado_dist",
#         y="atividade_composta",
#         color="cluster" if "cluster" in df.columns else "siglaUf",
#         hover_data=['nome', 'siglaPartido', 'distancia_km'],
#         title="Perfil de Performance: Gasto Normalizado vs. Atividade Composta",
#         labels={
#             "gasto_normalizado_dist": "Gasto Normalizado (R$ / km)",
#             "atividade_composta": "Índice de Atividade Composta"
#         }
#     )
#     fig_scatter.update_layout(template=PLOTLY_TEMPLATE)
#     save_plotly_figure(fig_scatter, "scatter_performance_deputado", output_dir)


def plot_choropleth_brasil(df_mestre: pd.DataFrame) -> None:
    """
    Gera um mapa choropleth mostrando o volume total de gastos por estado (UF).
    """
    if "siglaUf" not in df_mestre.columns or "gasto_total" not in df_mestre.columns:
        return

    # 1. Carregar GeoJSON dos estados brasileiros
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    try:
        with urlopen(geojson_url) as response:
            brazil_states = json.load(response)
    except Exception as e:
        print(f"Erro ao carregar GeoJSON: {e}")
        return

    # 2. Agrupar dados por UF
    df_map = (
        df_mestre.groupby("siglaUf")["gasto_total"]
        .sum()
        .reset_index()
    )

    # 3. Criar o mapa
    fig = px.choropleth(
        df_map,
        geojson=brazil_states,
        locations="siglaUf",
        featureidkey="properties.sigla",  # Mapeia 'siglaUf' para 'properties.sigla' no GeoJSON
        color="gasto_total",
        color_continuous_scale="Reds",
        template=PLOTLY_TEMPLATE,
        title="Distribuição Geográfica de Gastos Totais por UF",
        labels={"gasto_total": "Gasto Total (R$)", "siglaUf": "Estado"},
    )

    # Ajustar para focar no Brasil e esconder o resto do globo
    fig.update_geos(
        fitbounds="locations", 
        visible=False,
        projection_type="mercator"
    )
    
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

    save_plotly_figure(fig, "mapa_choropleth_gastos_uf")


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    print("Carregando dataset mestre...")
    df = load_dataset()
    df = prepare_master_dataset(df)

    print("Carregando despesas categorizadas (se existirem)...")
    df_desp_cat = load_categorized_expenses()

    print("Gerando resumo estatístico...")
    export_summary(df)

    print("Gerando histogramas...")
    plot_histograms(df)

    print("Gerando boxplots...")
    plot_boxplots(df)

    print("Gerando comparações por partido...")
    plot_parties(df)

    print("Gerando comparações por UF...")
    plot_ufs(df)

    print("Gerando scatter gasto vs atividade...")
    plot_scatter_gasto_atividade(df)

    plot_scatter_gasto_atividade_raw(df)

    print("Gerando heatmap de correlações...")
    plot_correlation_heatmap(df)

    print("Gerando heatmap partido x categoria...")
    plot_party_category_heatmap(df, df_desp_cat)

    print("Exportando tabelas de anomalias...")
    export_anomaly_tables(df)

    # print("Analise por Deputado...")
    # plot_analise_por_deputado(df, DATA_PROCESSED)

    # ------------------------------------------------------
    # Gráficos adicionais mais expressivos / inspirados nas referências
    # ------------------------------------------------------
    print("Gerando treemap partido -> UF -> deputado...")
    plot_treemap_party_uf_deputado(df)

    print("Gerando beeswarm / strip por partido...")
    plot_beeswarm_activity_by_party(df)

    print("Gerando parallel coordinates...")
    plot_parallel_coordinates_profiles(df)

    print("Gerando área empilhada ao longo do tempo...")
    plot_stacked_area_party_over_time(df, df_desp_cat)

    print("Gerando treemap partido x categoria...")
    plot_party_category_share_treemap(df, df_desp_cat)

    print("Gerando dendrograma e heatmap clusterizado...")
    plot_clustered_heatmap_and_dendrogram(df)

    print("✅ EDA concluída com sucesso!")
    print(f"Arquivos salvos em: {EDA_DIR}")

    print("Gerando mapa choropleth...")
    plot_choropleth_brasil(df)


if __name__ == "__main__":
    main()
