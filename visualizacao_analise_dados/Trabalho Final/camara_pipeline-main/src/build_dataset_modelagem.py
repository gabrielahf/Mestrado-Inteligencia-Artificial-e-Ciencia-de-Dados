
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Configuração de caminhos
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

DEPUTADOS_PATH = DATA_PROCESSED / "deputados_clean.parquet"
DESPESAS_PATH = DATA_PROCESSED / "despesas_clean.parquet"
EVENTOS_PATH = DATA_PROCESSED / "eventos_clean.parquet"
PROPS_PATH = DATA_PROCESSED / "proposicoes_clean.parquet"

OUT_CSV = DATA_PROCESSED / "dataset_mestre_clean.csv"
OUT_PARQUET = DATA_PROCESSED / "dataset_mestre_clean.parquet"
DESP_CAT_CSV = DATA_PROCESSED / "despesas_categorizadas.csv"
PLOT_CAT = DATA_PROCESSED / "gastos_por_categoria.png"


# ==========================================================
# Regras simples de categorização das despesas
# ==========================================================
# Observação: esta classificação é heurística (por palavras-chave).
# É útil para exploração e dashboard, mas não substitui uma taxonomia oficial.

GRUPOS_CATEGORIA = {
    "Transporte": [
        "bilhete", "passagem", "locomoção", "locomocao", "transporte",
        "aluguel de veículos", "aluguel de veiculos",
        "combustível", "combustivel", "taxi", "uber", "veículo", "veiculo",
        "fretamento",
    ],
    "Alimentação": [
        "alimentação", "alimentacao", "restaurante", "refeição", "refeicao", "lanches",
    ],
    "Escritório e Funcionamento": [
        "escritório", "escritorio", "materiais", "serviços postais", "servicos postais",
        "correios", "telefone", "internet", "locação", "locacao",
        "espaço", "espaco", "condomínio", "condominio",
        "copiadora", "material de expediente",
    ],
    "Cursos e Capacitação": [
        "curso", "capacitação", "capacitacao", "treinamento", "inscrição", "inscricao",
    ],
    "Divulgação do Mandato": [
        "publicidade", "divulgação", "divulgacao", "assessoria de imprensa", "marketing",
    ],
}


def _classificar_macro_categoria(desc: str) -> str:
    """Classifica o texto de 'tipoDespesa' em uma macro-categoria."""
    if not isinstance(desc, str):
        return "Outros"

    texto = desc.lower()
    for categoria, palavras in GRUPOS_CATEGORIA.items():
        if any(p in texto for p in palavras):
            return categoria
    return "Outros"


# ==========================================================
# Helpers simples para manter o código legível e robusto
# ==========================================================


def _normalizar_id_deputado(df: pd.DataFrame, nome_df: str) -> pd.DataFrame:
    """
    Garante a presença de 'idDeputado' em formato numérico.
    Mantém a lógica simples para ficar fácil de explicar.
    """
    if "idDeputado" not in df.columns:
        raise ValueError(f"Coluna 'idDeputado' não encontrada em {nome_df}.")

    df = df.copy()
    df["idDeputado"] = pd.to_numeric(df["idDeputado"], errors="coerce")
    df = df.dropna(subset=["idDeputado"])
    df["idDeputado"] = df["idDeputado"].astype(int)
    return df



def _slug_coluna(texto: str) -> str:
    """Converte nomes de categorias em nomes de colunas simples."""
    mapa = str.maketrans({
        "á": "a", "à": "a", "ã": "a", "â": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        "ç": "c",
        "-": "_", "/": "_",
    })
    return texto.lower().translate(mapa).replace(" ", "_")



def _carregar_arquivos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lê os parquets limpos produzidos por clean.py."""
    print("Lendo arquivos limpos (.parquet)...")

    df_deps = pd.read_parquet(DEPUTADOS_PATH)
    df_desp = pd.read_parquet(DESPESAS_PATH)
    df_evt = pd.read_parquet(EVENTOS_PATH)
    df_prop = pd.read_parquet(PROPS_PATH)

    print(f"  deputados_clean:   {len(df_deps):,} linhas")
    print(f"  despesas_clean:    {len(df_desp):,} linhas")
    print(f"  eventos_clean:     {len(df_evt):,} linhas")
    print(f"  proposicoes_clean: {len(df_prop):,} linhas")

    return df_deps, df_desp, df_evt, df_prop


# ==========================================================
# Etapa 1: Categorizar despesas + salvar CSV detalhado
# ==========================================================


def adicionar_categoria_macro(df_desp: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna 'categoria_macro' a partir de 'tipoDespesa'.
    Também salva um CSV detalhado para análise exploratória e dashboard.
    """
    print("\nClassificando despesas em macro-categorias...")

    df = df_desp.copy()
    if "tipoDespesa" not in df.columns:
        print("  Aviso: coluna 'tipoDespesa' não encontrada. Todas as despesas ficarão em 'Outros'.")
        df["categoria_macro"] = "Outros"
    else:
        df["tipoDespesa"] = df["tipoDespesa"].astype(str).str.lower()
        df["categoria_macro"] = df["tipoDespesa"].apply(_classificar_macro_categoria)

    df.to_csv(DESP_CAT_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✅ Arquivo salvo: {DESP_CAT_CSV}")
    return df


# ==========================================================
# Etapa 2: Gráfico simples de apoio à EDA
# ==========================================================


def plot_gastos_por_categoria(df_desp: pd.DataFrame) -> None:
    """
    Gera um gráfico simples de apoio à EDA.
    Usa valorLiquido_pos quando existir, para evitar que estornos distorçam o total gasto.
    """
    print("\nGerando gráfico de gastos por macro-categoria...")

    if "categoria_macro" not in df_desp.columns:
        print("  Aviso: coluna 'categoria_macro' não encontrada. Gráfico não será gerado.")
        return

    if "valorLiquido_pos" in df_desp.columns:
        valor_col = "valorLiquido_pos"
    elif "valorLiquido" in df_desp.columns:
        df_desp = df_desp.copy()
        df_desp["valorLiquido"] = pd.to_numeric(df_desp["valorLiquido"], errors="coerce").fillna(0.0)
        df_desp["valorLiquido_pos"] = df_desp["valorLiquido"].clip(lower=0.0)
        valor_col = "valorLiquido_pos"
    else:
        print("  Aviso: nenhuma coluna de valor encontrada. Gráfico não será gerado.")
        return

    agrupado = (
        df_desp.groupby("categoria_macro")[valor_col]
        .sum()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(agrupado.index, agrupado.values, color="#4472C4")
    plt.xlabel("Total gasto positivo (R$)")
    plt.ylabel("Macro-categoria")
    plt.title("Gastos por Macro-Categoria (Câmara dos Deputados)")
    plt.tight_layout()
    plt.savefig(PLOT_CAT, dpi=150)
    plt.close()

    print(f"  ✅ Gráfico salvo: {PLOT_CAT}")


# ==========================================================
# Etapa 3: Agregações por deputado
# ==========================================================


def agregar_despesas(df_desp: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega despesas por deputado.
    Mantém duas visões importantes:
      - gasto_total: soma apenas dos valores positivos
      - gasto_liquido: soma considerando estornos/ajustes negativos
    Se houver categoria_macro, cria também colunas por categoria.
    """
    print("\nAgregando despesas...")

    df = _normalizar_id_deputado(df_desp, "despesas_clean.parquet")
    df = df.copy()

    if "valorLiquido" not in df.columns:
        raise ValueError("Coluna 'valorLiquido' não encontrada em despesas_clean.parquet")

    df["valorLiquido"] = pd.to_numeric(df["valorLiquido"], errors="coerce").fillna(0.0)

    # Se clean.py já criou valorLiquido_pos e eh_estorno, usamos essas colunas.
    # Caso contrário, recriamos aqui sem complicar a lógica.
    if "valorLiquido_pos" not in df.columns:
        df["valorLiquido_pos"] = df["valorLiquido"].clip(lower=0.0)
    else:
        df["valorLiquido_pos"] = pd.to_numeric(df["valorLiquido_pos"], errors="coerce").fillna(0.0)

    if "eh_estorno" not in df.columns:
        df["eh_estorno"] = df["valorLiquido"] < 0

    n_negativos = int((df["valorLiquido"] < 0).sum())
    if n_negativos > 0:
        print(f"  Aviso: {n_negativos} registros com valor negativo (possíveis estornos).")

    agg = (
        df.groupby("idDeputado", as_index=False)
        .agg(
            gasto_total=("valorLiquido_pos", "sum"),
            gasto_liquido=("valorLiquido", "sum"),
            qtd_despesas=("valorLiquido", "size"),
            qtd_estornos=("eh_estorno", "sum"),
        )
    )

    # Gastos por macro-categoria (também usando apenas os valores positivos)
    if "categoria_macro" in df.columns:
        print("  Encontrada 'categoria_macro' -> agregando gastos por categoria...")
        pivot_cat = (
            df.groupby(["idDeputado", "categoria_macro"])["valorLiquido_pos"]
            .sum()
            .reset_index()
            .pivot(index="idDeputado", columns="categoria_macro", values="valorLiquido_pos")
            .fillna(0.0)
        )

        pivot_cat.columns = [f"gasto_{_slug_coluna(col)}" for col in pivot_cat.columns]
        pivot_cat = pivot_cat.reset_index()
        agg = agg.merge(pivot_cat, on="idDeputado", how="left")

    return agg



def agregar_eventos(df_evt: pd.DataFrame) -> pd.DataFrame:
    """Agrega o número total de eventos por deputado."""
    print("\nAgregando eventos...")
    df = _normalizar_id_deputado(df_evt, "eventos_clean.parquet")

    agg_evt = (
        df.groupby("idDeputado", as_index=False)
        .size()
        .rename(columns={"size": "total_eventos"})
    )
    return agg_evt



def agregar_proposicoes(df_prop: pd.DataFrame) -> pd.DataFrame:
    """Agrega o número total de proposições por deputado."""
    print("\nAgregando proposições...")
    df = _normalizar_id_deputado(df_prop, "proposicoes_clean.parquet")

    agg_prop = (
        df.groupby("idDeputado", as_index=False)
        .size()
        .rename(columns={"size": "total_proposicoes"})
    )
    return agg_prop


# ==========================================================
# Etapa 4: Montar o dataset mestre
# ==========================================================


def montar_dataset_mestre(
    df_deps: pd.DataFrame,
    df_desp_agg: pd.DataFrame,
    df_evt_agg: pd.DataFrame,
    df_prop_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta metadados dos deputados com as agregações de despesas,
    eventos e proposições.

    A base principal é a tabela de deputados, pois assim preservamos
    nome, partido e UF no dataset final.
    """
    print("\nMontando dataset mestre...")

    df = _normalizar_id_deputado(df_deps, "deputados_clean.parquet").copy()
    df = df.drop_duplicates(subset="idDeputado")

    df = df.merge(df_desp_agg, on="idDeputado", how="left")
    df = df.merge(df_evt_agg, on="idDeputado", how="left")
    df = df.merge(df_prop_agg, on="idDeputado", how="left")

    # Preenche nulos numéricos com zero, pois ausência de registros aqui
    # significa ausência de despesa/atividade no período analisado.
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Métrica simples de atividade composta.
    # Mantemos a fórmula já usada no projeto para facilitar a explicação.
    df["atividade_composta"] = (
        pd.to_numeric(df.get("total_proposicoes", 0), errors="coerce").fillna(0.0)
        + 0.3 * pd.to_numeric(df.get("total_eventos", 0), errors="coerce").fillna(0.0)
    )

    # Gasto total ajustado: evita valores negativos na métrica de custo.
    df["gasto_total_ajustado"] = pd.to_numeric(df.get("gasto_total", 0), errors="coerce").fillna(0.0)
    df["gasto_total_ajustado"] = df["gasto_total_ajustado"].clip(lower=0.0)

    # Deputados sem despesa registrada.
    df["sem_despesa"] = (
        (pd.to_numeric(df.get("gasto_total", 0), errors="coerce").fillna(0.0) == 0.0)
        & (pd.to_numeric(df.get("qtd_despesas", 0), errors="coerce").fillna(0.0) == 0.0)
    )
    print(f"  Deputados sem despesa registrada: {int(df['sem_despesa'].sum())}")

    # Custo por atividade: quando não há atividade, deixamos 0 para manter a leitura simples.
    denominador = df["atividade_composta"].replace(0, pd.NA)
    df["custo_por_atividade"] = (df["gasto_total_ajustado"] / denominador).fillna(0.0)

    return df


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    df_deps, df_desp, df_evt, df_prop = _carregar_arquivos()

    # 1) Categoriza despesas e salva o CSV detalhado
    df_desp = adicionar_categoria_macro(df_desp)

    # 2) Gera um gráfico simples de apoio à EDA
    plot_gastos_por_categoria(df_desp)

    # 3) Agrega os dados por deputado
    df_desp_agg = agregar_despesas(df_desp)
    df_evt_agg = agregar_eventos(df_evt)
    df_prop_agg = agregar_proposicoes(df_prop)

    # 4) Monta o dataset mestre preservando nome, partido e UF
    df_mestre = montar_dataset_mestre(df_deps, df_desp_agg, df_evt_agg, df_prop_agg)

    # 5) Salva os resultados finais
    df_mestre.to_csv(OUT_CSV, index=False)
    df_mestre.to_parquet(OUT_PARQUET, index=False)

    print("\n✅ Dataset mestre gerado com sucesso!")
    print(f"  CSV:     {OUT_CSV}")
    print(f"  PARQUET: {OUT_PARQUET}")
    print(f"  Linhas:  {len(df_mestre):,}")


if __name__ == "__main__":
    main()
