import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm

BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"
DEBUG_LIMIT = int(os.getenv("DEBUG_LIMIT", "0"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "1"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "100"))
REQUEST_SLEEP_SECONDS = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.5"))
CHECKPOINT_EVERY_REQUESTS = int(os.getenv("CHECKPOINT_EVERY_REQUESTS", "50"))

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def _parse_int_list(env_name: str, default: List[int]) -> List[int]:
    raw = os.getenv(env_name)
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _get(url: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
    all_data: List[Dict[str, Any]] = []
    page = 1

    while True:
        p = params.copy() if params else {}
        p["pagina"] = page
        p["itens"] = PAGE_SIZE

        response: Optional[requests.Response] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=p, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                wait = RETRY_BASE_SECONDS * (2 ** attempt)
                print(f"[{url}] page {page} attempt {attempt + 1}/{MAX_RETRIES}: {exc} -> wait {wait:.1f}s")
                if attempt == MAX_RETRIES - 1:
                    print(f"Abandoning page {page} for {url}. Returning {len(all_data)} rows collected so far.")
                    return all_data
                time.sleep(wait)

        if response is None:
            return all_data

        data = response.json().get("dados", [])
        if not data:
            break

        all_data.extend(data)

        if DEBUG_LIMIT and len(all_data) >= DEBUG_LIMIT:
            return all_data[:DEBUG_LIMIT]

        page += 1
        time.sleep(REQUEST_SLEEP_SECONDS)

    return all_data


def get_deputados(legislatura: int) -> List[Dict[str, Any]]:
    return _get(f"{BASE_URL}/deputados", {"idLegislatura": legislatura})


def get_despesas_deputado(deputado_id: int, ano: Optional[int] = None) -> List[Dict[str, Any]]:
    params = {"ano": ano} if ano else {}
    return _get(f"{BASE_URL}/deputados/{deputado_id}/despesas", params)


def get_proposicoes_por_autor(deputado_id: int, ano: int) -> List[Dict[str, Any]]:
    return _get(f"{BASE_URL}/proposicoes", {"idDeputadoAutor": deputado_id, "ano": ano})


def get_eventos_deputado(deputado_id: int, ano: int) -> List[Dict[str, Any]]:
    return _get(
        f"{BASE_URL}/deputados/{deputado_id}/eventos",
        {"dataInicio": f"{ano}-01-01", "dataFim": f"{ano}-12-31"},
    )


def _load_existing_records(tmp_path: Path) -> List[Dict[str, Any]]:
    if not tmp_path.exists():
        return []
    df = pd.read_parquet(tmp_path)
    print(f"Resuming records from {tmp_path.name}: {len(df)} rows")
    return df.to_dict("records")


def _load_done_keys(progress_path: Path) -> Set[Tuple[int, int]]:
    if not progress_path.exists():
        return set()
    df = pd.read_parquet(progress_path)
    keys = set(zip(df["idDeputado"].astype(int), df["anoConsulta"].astype(int)))
    print(f"Resuming progress from {progress_path.name}: {len(keys)} deputy-year pairs")
    return keys


def _save_records(records: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


def _save_done_keys(done_keys: Set[Tuple[int, int]], path: Path) -> None:
    if done_keys:
        df = pd.DataFrame(sorted(done_keys), columns=["idDeputado", "anoConsulta"])
    else:
        df = pd.DataFrame(columns=["idDeputado", "anoConsulta"])
    df.to_parquet(path, index=False)


def _maybe_checkpoint(
    *,
    records: List[Dict[str, Any]],
    done_keys: Set[Tuple[int, int]],
    data_tmp_path: Path,
    progress_tmp_path: Path,
    requests_since_checkpoint: int,
    label: str,
) -> None:
    if requests_since_checkpoint < CHECKPOINT_EVERY_REQUESTS:
        return
    _save_records(records, data_tmp_path)
    _save_done_keys(done_keys, progress_tmp_path)
    print(f"Checkpoint [{label}]: {len(records)} rows, {len(done_keys)} deputy-year pairs")


def _finalize_dataset(
    *,
    records: List[Dict[str, Any]],
    out_path: Path,
    data_tmp_path: Path,
    progress_tmp_path: Path,
    label: str,
) -> None:
    _save_records(records, out_path)
    if data_tmp_path.exists():
        data_tmp_path.unlink()
    if progress_tmp_path.exists():
        progress_tmp_path.unlink()
    print(f"Saved {out_path.name} with {len(records)} rows [{label}]")


def fetch_dataset_by_deputado_ano(
    *,
    deputados: List[int],
    anos: List[int],
    label: str,
    fetch_func: Callable[[int, int], List[Dict[str, Any]]],
    out_path: Path,
    data_tmp_path: Path,
    progress_tmp_path: Path,
) -> None:
    records = _load_existing_records(data_tmp_path)
    done_keys = _load_done_keys(progress_tmp_path)
    requests_since_checkpoint = 0

    for dep_id in tqdm(deputados, desc=label):
        for ano in anos:
            key = (int(dep_id), int(ano))
            if key in done_keys:
                continue

            try:
                rows = fetch_func(dep_id, ano)
            except Exception as exc:
                print(f"[{label}] Skipping dep {dep_id} ano {ano}: {exc}")
                done_keys.add(key)
                requests_since_checkpoint += 1
                _maybe_checkpoint(
                    records=records,
                    done_keys=done_keys,
                    data_tmp_path=data_tmp_path,
                    progress_tmp_path=progress_tmp_path,
                    requests_since_checkpoint=requests_since_checkpoint,
                    label=label,
                )
                if requests_since_checkpoint >= CHECKPOINT_EVERY_REQUESTS:
                    requests_since_checkpoint = 0
                continue

            for row in rows:
                row["idDeputado"] = dep_id
                row["anoConsulta"] = ano

            records.extend(rows)
            done_keys.add(key)
            requests_since_checkpoint += 1

            _maybe_checkpoint(
                records=records,
                done_keys=done_keys,
                data_tmp_path=data_tmp_path,
                progress_tmp_path=progress_tmp_path,
                requests_since_checkpoint=requests_since_checkpoint,
                label=label,
            )
            if requests_since_checkpoint >= CHECKPOINT_EVERY_REQUESTS:
                requests_since_checkpoint = 0

    _finalize_dataset(
        records=records,
        out_path=out_path,
        data_tmp_path=data_tmp_path,
        progress_tmp_path=progress_tmp_path,
        label=label,
    )


def main() -> None:
    # print("Fetching Deputados...")
    # legislaturas = _parse_int_list("LEGISLATURAS", [56, 57])
    # anos = _parse_int_list("ANOS", [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])

    # deps_all: List[Dict[str, Any]] = []
    # for leg in tqdm(legislaturas, desc="Deputados"):
    #     deps_all.extend(get_deputados(leg))

    # df_deps = pd.DataFrame(deps_all).drop_duplicates(subset="id")
    # if "id" in df_deps.columns:
    #     df_deps["idDeputado"] = df_deps["id"].astype(int)
    # df_deps.to_parquet(DATA_RAW / "deputados.parquet", index=False)
    # deputados = df_deps["idDeputado"].astype(int).tolist()
    # print(f"Saved deputados.parquet with {len(df_deps)} rows")

    df_deps = pd.read_parquet(DATA_RAW / "deputados.parquet")
    deputados = df_deps["idDeputado"].astype(int).tolist()
    anos = _parse_int_list("ANOS", [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])

    # print("Fetching Despesas...")
    # fetch_dataset_by_deputado_ano(
    #     deputados=deputados,
    #     anos=anos,
    #     label="Despesas",
    #     fetch_func=get_despesas_deputado,
    #     out_path=DATA_RAW / "despesas.parquet",
    #     data_tmp_path=DATA_RAW / "despesas_tmp.parquet",
    #     progress_tmp_path=DATA_RAW / "despesas_progress.parquet",
    # )
    

    print("Fetching Proposições...")
    fetch_dataset_by_deputado_ano(
        deputados=deputados,
        anos=anos,
        label="Proposicoes",
        fetch_func=get_proposicoes_por_autor,
        out_path=DATA_RAW / "proposicoes.parquet",
        data_tmp_path=DATA_RAW / "proposicoes_tmp.parquet",
        progress_tmp_path=DATA_RAW / "proposicoes_progress.parquet",
    )

    print("Fetching Eventos...")
    fetch_dataset_by_deputado_ano(
        deputados=deputados,
        anos=anos,
        label="Eventos",
        fetch_func=get_eventos_deputado,
        out_path=DATA_RAW / "eventos.parquet",
        data_tmp_path=DATA_RAW / "eventos_tmp.parquet",
        progress_tmp_path=DATA_RAW / "eventos_progress.parquet",
    )


if __name__ == "__main__":
    main()
