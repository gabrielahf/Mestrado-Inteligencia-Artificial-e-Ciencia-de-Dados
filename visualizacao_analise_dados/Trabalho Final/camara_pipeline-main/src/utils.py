import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Iterator, Dict, Any, Optional
from .config import BASE_API

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.headers.update({"Accept": "application/json"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def paginate(session: requests.Session, path: str, params: Optional[Dict[str, Any]] = None, page_size: int = 100) -> Iterator[Dict[str, Any]]:
    """Yield items from a paginated API endpoint (api/v2) that returns {'dados': [...], 'links': [...]}"""
    url = f"{BASE_API.rstrip('/')}/{path.lstrip('/')}"
    page = 1
    params = dict(params or {})
    params.setdefault("itens", page_size)
    while True:
        params["pagina"] = page
        resp = session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        dados = payload.get("dados") or []
        if not dados:
            break
        for item in dados:
            yield item
        links = payload.get("links") or []
        has_next = any((l.get("rel") == "next") for l in links)
        if not has_next:
            break
        page += 1

def safe_get(session: requests.Session, path: str, params=None) -> Dict[str, Any]:
    url = f"{BASE_API.rstrip('/')}/{path.lstrip('/')}"
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

def parse_money_br(value) -> float:
    """Converte 'R$ 1.234,56' ou '1.234,56' para float 1234.56"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("R$", "").replace("\u00A0", " ").strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0
