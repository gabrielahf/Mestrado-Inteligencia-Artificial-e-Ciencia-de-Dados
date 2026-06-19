from dataclasses import dataclass
import os

BASE_API = "https://dadosabertos.camara.leg.br/api/v2"

@dataclass
class Settings:
    legislaturas: list[int]
    anos: list[int]
    rate_sleep_seconds: float = 0.3
    page_size: int = 100

def _parse_list(env_value: str) -> list[int]:
    return [int(x.strip()) for x in env_value.split(",") if x.strip()]

def load_settings() -> Settings:
    legislaturas = os.getenv("LEGISLATURAS", "56,57")
    anos = os.getenv("ANOS", "")
    rate = float(os.getenv("RATE_SLEEP_SECONDS", "0.3"))
    page = int(os.getenv("PAGE_SIZE", "100"))
    return Settings(
        legislaturas=_parse_list(legislaturas) if legislaturas else [],
        anos=_parse_list(anos) if anos else [],
        rate_sleep_seconds=rate,
        page_size=page,
    )
