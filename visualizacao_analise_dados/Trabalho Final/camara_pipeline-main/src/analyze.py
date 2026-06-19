from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"


def load_dataset():
    # O seu novo arquivo gerado pelo clean.py
    parquet = DATA_PROCESSED / "dataset_modelagem.parquet"
    csv = DATA_PROCESSED / "dataset_modelagem.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Nenhum dataset encontrado em data/processed/")

    return df


def main():
    df = load_dataset()

    # Garante colunas numéricas necessárias
    numeric_cols = ["gasto_total", "gasto_normalizado_dist", "atividade_composta", "distancia_km"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 1. Correlação: Gasto Normalizado vs Atividade
    # Isso mostra se a eficiência de gasto está ligada à produção
    x = df["gasto_normalizado_dist"].astype(float)
    y = df["atividade_composta"].astype(float)
    r = float(np.corrcoef(x, y)[0, 1]) if len(df) >= 2 else float("nan")

    # Scatter: Gasto Normalizado vs Atividade
    plt.figure()
    plt.scatter(x, y, alpha=0.6, c=df["distancia_km"], cmap='viridis')
    plt.colorbar(label='Distância até Brasília (km)')
    plt.xlabel("Gasto normalizado (R$ / km de distância)")
    plt.ylabel("Atividade composta")
    plt.title("Relação: Gasto Normalizado por Distância vs Atividade")
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / "scatter_normalizado_atividade.png", dpi=130)
    plt.close()

    # 2. K-Means clustering usando as métricas normalizadas
    # Agora o algoritmo agrupa por eficiência, ignorando o "custo geográfico" inevitável
    features_list = ["gasto_normalizado_dist", "atividade_composta"]
    features = df[features_list].copy()
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    inertias = []
    models = []

    # Testando k de 2 a 5
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        models.append(km)

    # Seleção do melhor K (pelo método da inércia mínima neste caso)
    best_k = int(np.argmin(inertias)) + 2
    best_model = models[best_k - 2]
    df["cluster"] = best_model.labels_

    # 3. Resumo dos Clusters
    resumo = (
        df.groupby("cluster")
        .agg(
            n=("idDeputado", "count"),
            gasto_norm_medio=("gasto_normalizado_dist", "mean"),
            atividade_media=("atividade_composta", "mean"),
            distancia_media=("distancia_km", "mean")
        )
        .reset_index()
        .to_dict(orient="records")
    )

    # Scatter de Clusters Normalizados
    plt.figure()
    for c in sorted(df["cluster"].unique()):
        m = df["cluster"] == c
        plt.scatter(
            df.loc[m, "gasto_normalizado_dist"],
            df.loc[m, "atividade_composta"],
            alpha=0.6,
            label=f"Cluster {c}",
        )
    plt.legend()
    plt.xlabel("Gasto normalizado (R$ / km)")
    plt.ylabel("Atividade composta")
    plt.title("Perfis de Parlamentares (Normalizado por Distância)")
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / "clusters_normalizados_scatter.png", dpi=130)
    plt.close()

    # Salvando resultados
    (DATA_PROCESSED / "correlacao_pearson_normalizada.json").write_text(
        json.dumps({"pearson_r_normalizado": r}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA_PROCESSED / "kmeans_resumo_normalizado.json").write_text(
        json.dumps({"best_k": best_k, "resumo": resumo}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Análise finalizada com sucesso!")
    print(f"Correlação normalizada (r): {r:.4f}")
    print(f"Melhor k encontrado: {best_k}")
    print(f"Arquivos salvos em: {DATA_PROCESSED}")


if __name__ == "__main__":
    main()