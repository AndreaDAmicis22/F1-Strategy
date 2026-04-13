"""
OpenF1 API Client - Modulo wrapper per l'API OpenF1
Gestisce: paginazione, errori di rete, parsing risposte, caching locale
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_TTL = 3600 * 24  # 24 ore


def _cache_key(endpoint: str, params: dict) -> str:
    raw = endpoint + json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache(key: str) -> Any | None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    meta_path = CACHE_DIR / f"{key}.meta"
    if meta_path.exists():
        age = time.time() - float(meta_path.read_text())
        if age > CACHE_TTL:
            return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cache(key: str, data: Any):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    meta_path = CACHE_DIR / f"{key}.meta"
    path.write_text(json.dumps(data))
    meta_path.write_text(str(time.time()))


def fetch(endpoint: str, params: dict | None = None, use_cache: bool = True, max_retries: int = 3) -> list:
    """
    Fetch dati dall'API OpenF1 con caching e retry automatici.

    Args:
        endpoint: es. "laps", "pit", "weather", "race_control"
        params: parametri query (es. {"session_key": 9158})
        use_cache: usa cache locale se disponibile
        max_retries: tentativi in caso di errore rete

    Returns:
        Lista di record (dict)
    """
    if params is None:
        params = {}

    cache_key = _cache_key(endpoint, params)
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            logger.info(f"[CACHE HIT] {endpoint} {params}")
            return cached

    query = urlencode(params)
    url = f"{BASE_URL}/{endpoint}?{query}" if query else f"{BASE_URL}/{endpoint}"

    all_results = []
    attempt = 0

    while attempt < max_retries:
        try:
            logger.info(f"[API] GET {url}")
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                all_results = data if isinstance(data, list) else [data]
            break
        except HTTPError as e:
            if e.code == 429:
                wait_time = 2 ** (attempt + 3)  # 8s, 16s, 32s invece di 10, 20, 30
                logger.warning(f"Rate limit (429). Attesa {wait_time}s prima del retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                attempt += 1
                continue
            if e.code == 503:
                wait_time = 2**attempt
                time.sleep(wait_time)
                attempt += 1
                continue
            logger.exception(f"HTTP {e.code} su {url}: {e.reason}")
            break
        except URLError as e:
            logger.warning(f"URLError (tentativo {attempt + 1}): {e.reason}")
            time.sleep(2**attempt)
            attempt += 1
        except Exception as e:
            logger.exception(f"Errore inatteso: {e}")
            break

    if use_cache and all_results:
        _save_cache(cache_key, all_results)

    return all_results


# ── Helpers specifici per entità ──────────────────────────────────────────────


def get_sessions(circuit_key: str | None = None, year: int | None = None, session_type: str = "Race") -> list:
    params = {"session_type": session_type}
    if circuit_key:
        params["circuit_key"] = circuit_key
    if year:
        params["year"] = year
    return fetch("sessions", params)


def get_laps(session_key: int) -> list:
    return fetch("laps", {"session_key": session_key})


def get_pit_stops(session_key: int) -> list:
    return fetch("pit", {"session_key": session_key})


def get_weather(session_key: int) -> list:
    return fetch("weather", {"session_key": session_key})


def get_race_control(session_key: int) -> list:
    return fetch("race_control", {"session_key": session_key})


def get_stints(session_key: int) -> list:
    return fetch("stints", {"session_key": session_key})


def get_drivers(session_key: int) -> list:
    return fetch("drivers", {"session_key": session_key})


def get_meetings(circuit_key: str | None = None, year: int | None = None) -> list:
    params = {}
    if circuit_key:
        params["circuit_key"] = circuit_key
    if year:
        params["year"] = year
    return fetch("meetings", params)
