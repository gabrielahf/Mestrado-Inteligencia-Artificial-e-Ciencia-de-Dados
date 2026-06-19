# Câmara dos Deputados — Pipeline (Leg. 56 e 57)

Pipeline **ETL + Análise** para relacionar **gastos** (Cota Parlamentar) e **atividade parlamentar**
(proposições, presença em votações, participação em eventos).

## Estrutura
```
camara_pipeline/
  data/
    raw/         # dumps intermediários (JSON/CSV baixados)
    processed/   # dataset mestre final (CSV/Parquet)
  src/
    config.py
    utils.py
    fetch.py
    build_dataset.py
    analyze.py
  requirements.txt
  .env.example
```

## Como usar
1. **Instale dependências**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```

2. **Configure variáveis** em `.env` (copie de `.env.example`).

3. **Baixe e construa dataset**
   ```bash
   python -m src.build_dataset
   ```

   Saídas:
   - `data/processed/dataset_mestre.parquet`
   - `data/processed/dataset_mestre.csv`

4. **Rode análises (correlação e K-Means)**
   ```bash
   python -m src.analyze
   ```

   Saídas:
   - `data/processed/correlacao_pearson.json`
   - `data/processed/kmeans_resumo.json`
   - Gráficos: `data/processed/scatter_gasto_atividade.png`, `data/processed/clusters_scatter.png`

## Notas importantes
- Os **endpoints oficiais** estão documentados em `https://dadosabertos.camara.leg.br/swagger/api.html`.
- Para reduzir chamadas, o pipeline mistura **API REST** e **arquivos anuais** (CSV/JSON) públicos.
- Você pode limitar o escopo por **anos** ou **legislaturas** via `.env`.

## Licença
MIT — uso acadêmico/educacional.
