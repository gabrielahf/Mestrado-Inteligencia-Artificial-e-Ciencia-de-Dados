from pathlib import Path

import geopandas as gpd
from geopy.distance import geodesic
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)


def _ensure_id_deputado(df: pd.DataFrame) -> pd.DataFrame:
    """Garante a existência da coluna idDeputado sem complicar a lógica."""
    if "idDeputado" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "idDeputado"})
    elif "idDeputado" in df.columns and "id" in df.columns:
        # Se as duas colunas existirem, mantém idDeputado como referência principal.
        df = df.drop(columns=["id"])

    if "idDeputado" in df.columns:
        df["idDeputado"] = pd.to_numeric(df["idDeputado"], errors="coerce")
        df = df.dropna(subset=["idDeputado"]).copy()
        df["idDeputado"] = df["idDeputado"].astype(int)

    return df


def _ensure_ano_consulta(df: pd.DataFrame) -> pd.DataFrame:
    if "anoConsulta" in df.columns:
        df["anoConsulta"] = pd.to_numeric(df["anoConsulta"], errors="coerce").astype(
            "Int64"
        )
    return df


def clean_deputados() -> pd.DataFrame:
    """
    Lê e limpa a base de deputados.
    - Garante a coluna idDeputado
    - Remove duplicados
    - Mantém apenas colunas principais
    """
    df = pd.read_parquet(DATA_RAW / "deputados.parquet")
    df = _ensure_id_deputado(df)
    df = df.drop_duplicates(subset="idDeputado")

    cols = [
        c for c in ["idDeputado", "nome", "siglaPartido", "siglaUf"] if c in df.columns
    ]
    df = df[cols].copy()

    df.to_parquet(DATA_PROC / "deputados_clean.parquet", index=False)
    return df


def clean_despesas() -> pd.DataFrame:
    """
    Lê e limpa a base de despesas.
    - Garante tipo numérico em valorLiquido
    - Remove registros sem valor
    - Cria colunas auxiliares para lidar com negativos (estornos)
    - Preserva anoConsulta, quando existir
    """
    df = pd.read_parquet(DATA_RAW / "despesas.parquet")
    df = df.drop_duplicates()
    df = _ensure_id_deputado(df)
    df = _ensure_ano_consulta(df)

    df["valorLiquido"] = pd.to_numeric(df["valorLiquido"], errors="coerce")
    df = df[~df["valorLiquido"].isna()].copy()

    df["valorLiquido_pos"] = df["valorLiquido"].clip(lower=0)
    df["eh_estorno"] = df["valorLiquido"] < 0

    df.to_parquet(DATA_PROC / "despesas_clean.parquet", index=False)
    return df


def clean_proposicoes() -> pd.DataFrame:
    """
    Lê e limpa a base de proposições.
    - Remove duplicados por idProposicao + idDeputado (se existir)
    - Garante tipo numérico do idDeputado
    - Preserva anoConsulta, quando existir
    """
    df = pd.read_parquet(DATA_RAW / "proposicoes.parquet")
    df = _ensure_id_deputado(df)
    df = _ensure_ano_consulta(df)

    if "idDeputado" in df.columns:
        key_cols = ["idDeputado"]
        if "id" in df.columns:
            key_cols.append("id")
        elif "idProposicao" in df.columns:
            key_cols.append("idProposicao")
        df = df.drop_duplicates(subset=key_cols)
    else:
        df = df.drop_duplicates()

    df.to_parquet(DATA_PROC / "proposicoes_clean.parquet", index=False)
    return df


def clean_eventos() -> pd.DataFrame:
    """
    Lê e limpa a base de eventos.
    - Remove duplicados por idEvento + idDeputado (se existir)
    - Garante tipo numérico do idDeputado
    - Preserva anoConsulta, quando existir
    """
    df = pd.read_parquet(DATA_RAW / "eventos.parquet")
    df = _ensure_id_deputado(df)
    df = _ensure_ano_consulta(df)

    if "idDeputado" in df.columns:
        key_cols = ["idDeputado"]
        if "id" in df.columns:
            key_cols.append("id")
        elif "idEvento" in df.columns:
            key_cols.append("idEvento")
        df = df.drop_duplicates(subset=key_cols)
    else:
        df = df.drop_duplicates()

    df.to_parquet(DATA_PROC / "eventos_clean.parquet", index=False)
    return df


def build_dataset_modelagem(
    df_deps: pd.DataFrame,
    df_desp: pd.DataFrame,
    df_prop: pd.DataFrame,
    df_evt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera o dataset final já limpo e pronto para modelagem.
    - Agrega gastos por deputado
    - Conta proposições e eventos
    - Cria métrica de atividade composta
    """
    df = df_deps.copy()

    try:
        url_geojson = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
        gdf = gpd.read_file(url_geojson)
        brasilia_coords = (-15.7975, -47.8919)

        gdf['distancia_brasilia'] = gdf.apply(
            lambda row: geodesic((row.geometry.centroid.y, row.geometry.centroid.x), brasilia_coords).km, 
            axis=1
        )
        
        dist_map = gdf.set_index('sigla')['distancia_brasilia'].to_dict()
        df['distancia_km'] = df['siglaUf'].map(dist_map).fillna(0) 
    except Exception as e:
        print(f"Erro ao carregar dados geográficos: {e}")
        df['distancia_km'] = 0

    if not df_desp.empty and "idDeputado" in df_desp.columns:
        agg_desp = (
            df_desp.groupby("idDeputado")
            .agg(
                gasto_total=("valorLiquido_pos", "sum"),
                qtd_despesas=("valorLiquido_pos", "count"),
                gasto_liquido=("valorLiquido", "sum"),
                qtd_estornos=("eh_estorno", "sum"),
            )
            .reset_index()
        )
        df = df.merge(agg_desp, on="idDeputado", how="left")

    if not df_prop.empty and "idDeputado" in df_prop.columns:
        agg_prop = (
            df_prop.groupby("idDeputado").size().reset_index(name="total_proposicoes")
        )
        df = df.merge(agg_prop, on="idDeputado", how="left")

    if not df_evt.empty and "idDeputado" in df_evt.columns:
        agg_evt = df_evt.groupby("idDeputado").size().reset_index(name="total_eventos")
        df = df.merge(agg_evt, on="idDeputado", how="left")

    for col in [
        "gasto_total",
        "qtd_despesas",
        "gasto_liquido",
        "qtd_estornos",
        "total_proposicoes",
        "total_eventos",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    df["atividade_composta"] = df["total_proposicoes"] + df["total_eventos"] * 0.3
    df['gasto_normalizado_dist'] = df['gasto_total'] / (df['distancia_km'] + 1)

    df.to_csv(DATA_PROC / "dataset_modelagem.csv", index=False)
    df.to_parquet(DATA_PROC / "dataset_modelagem.parquet", index=False)
    return df


def main():
    print("Limpando dados brutos...")
    df_deps = clean_deputados()
    print(f"Deputados limpos: {len(df_deps)}")

    df_desp = clean_despesas()
    print(f"Despesas limpas: {len(df_desp)}")

    df_prop = clean_proposicoes()
    print(f"Proposições limpas: {len(df_prop)}")

    df_evt = clean_eventos()
    print(f"Eventos limpos: {len(df_evt)}")

    print("\nConstruindo dataset de modelagem...")
    df_model = build_dataset_modelagem(df_deps, df_desp, df_prop, df_evt)
    print(f"Dataset de modelagem gerado com {len(df_model)} deputados.")
    print("Arquivos salvos em data/processed/")


if __name__ == "__main__":
    main()
