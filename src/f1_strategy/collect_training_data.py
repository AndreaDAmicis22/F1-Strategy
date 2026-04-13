"""
collect_training_data.py
========================
Scarica dati reali dall'API OpenF1 e costruisce un dataset CSV per il training ML.

Eseguire PRIMA di train_models.py:
    python collect_training_data.py

Richiede accesso a api.openf1.org (eseguire in locale, non nel container CI).

Output:
    data/laps_raw.csv      — ogni riga = un giro con compound, tempo, meteo, ecc.
    data/stints_raw.csv    — stint con compound e giri
    data/sessions_meta.json — metadata sessioni scaricate
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import openf1_client as api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collector")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


YEARS = [2023, 2024, 2025]
SESSION_TYPE = "Race"


# ── Trova sessioni ─────────────────────────────────────────────────────────────
def get_all_race_sessions(years=YEARS):
    """Scarica tutte le sessioni Race degli anni specificati."""
    sessions = []
    for year in years:
        logger.info(f"Recupero sessioni Race {year}...")
        raw = api.fetch("sessions", {"year": year, "session_type": SESSION_TYPE})
        time.sleep(1.5)
        logger.info(f"  Trovate {len(raw)} sessioni")
        sessions.extend(raw)
    return sessions


# ── Scarica laps con compound ──────────────────────────────────────────────────
def get_laps_with_compound(session_key: int) -> list[dict]:
    """
    Unisce laps + stints per ottenere il compound su ogni giro.
    OpenF1 non include il compound direttamente in /laps, quindi
    bisogna matchare con /stints per stint_lap range.
    """
    laps = api.get_laps(session_key)
    time.sleep(1)
    stints = api.get_stints(session_key)
    time.sleep(1)

    if not laps:
        return []

    # Costruisci mappa: driver_number → lista stint con compound
    driver_stints: dict[int, list] = {}
    for s in stints:
        drv = s.get("driver_number")
        if not drv:
            continue
        if drv not in driver_stints:
            driver_stints[drv] = []
        driver_stints[drv].append(s)

    def compound_for_lap(driver: int, lap_num: int) -> str | None:
        for s in driver_stints.get(driver, []):
            lap_start = s.get("lap_start", 0) or 0
            lap_end = s.get("lap_end", 9999) or 9999
            if lap_start <= lap_num <= lap_end:
                return (s.get("compound") or "").lower() or None
        return None

    enriched = []
    for lap in laps:
        drv = lap.get("driver_number")
        lap_num = lap.get("lap_number")
        compound = compound_for_lap(drv, lap_num)
        if not compound or compound not in ("soft", "medium", "hard", "intermediate", "wet"):
            continue

        duration = lap.get("lap_duration")
        if not duration or duration <= 0 or duration > 300:
            continue  # salta giri invalidi / in pit / SC troppo lenti

        enriched.append(
            {
                "session_key": session_key,
                "driver_number": drv,
                "lap_number": lap_num,
                "lap_duration": duration,
                "compound": compound,
                "is_pit_out_lap": lap.get("is_pit_out_lap", False),
                "duration_sector_1": lap.get("duration_sector_1"),
                "duration_sector_2": lap.get("duration_sector_2"),
                "duration_sector_3": lap.get("duration_sector_3"),
                "i1_speed": lap.get("i1_speed"),
                "i2_speed": lap.get("i2_speed"),
                "st_speed": lap.get("st_speed"),
            }
        )

    return enriched


def get_weather_for_session(session_key: int) -> list[dict]:
    """Scarica dati meteo della sessione."""
    time.sleep(1)
    return api.get_weather(session_key)


def get_race_control_events(session_key: int) -> list[dict]:
    """Recupera eventi SC/VSC."""
    time.sleep(1)
    events = api.get_race_control(session_key)
    sc_laps = set()
    for e in events:
        msg = (e.get("message") or "").upper()
        lap = e.get("lap_number")
        if lap and ("SAFETY CAR" in msg or "VIRTUAL" in msg):
            sc_laps.add(lap)
    return sc_laps


# ── Merge meteo su ogni giro ───────────────────────────────────────────────────
def merge_weather_to_laps(laps: list[dict], weather_data: list[dict], sc_laps: set) -> list[dict]:
    """
    Associa a ogni giro le condizioni meteo più vicine temporalmente.
    Usa lap_number come proxy (meteo è campionato ogni ~20s).
    """
    if not weather_data:
        for lap in laps:
            lap["air_temp"] = None
            lap["track_temp"] = None
            lap["rainfall"] = 0.0
            lap["is_sc_lap"] = lap["lap_number"] in sc_laps
        return laps

    # Raggruppa meteo per sample index
    # OpenF1 /weather non ha lap_number diretto, usiamo date/meeting_key
    # Approssimazione: distribuiamo uniformemente i sample sui giri
    n_weather = len(weather_data)

    for lap in laps:
        lap_num = lap.get("lap_number", 1)
        # Indice proporzionale nel dataset meteo
        max_lap = max(l.get("lap_number", 1) for l in laps)
        idx = min(int((lap_num / max(max_lap, 1)) * n_weather), n_weather - 1)
        w = weather_data[idx]

        lap["air_temp"] = w.get("air_temperature")
        lap["track_temp"] = w.get("track_temperature")
        lap["rainfall"] = w.get("rainfall", 0.0) or 0.0
        lap["humidity"] = w.get("humidity")
        lap["wind_speed"] = w.get("wind_speed")
        lap["is_sc_lap"] = lap_num in sc_laps

    return laps


# ── Calcola stint_lap ──────────────────────────────────────────────────────────
def add_stint_lap(laps: list[dict]) -> list[dict]:
    """
    Aggiunge stint_lap = giro relativo all'interno dello stint per ogni giro.
    Raggruppa per (driver, compound sequence) e conta la posizione.
    """
    # Raggruppa per driver
    by_driver: dict = {}
    for lap in laps:
        d = lap["driver_number"]
        by_driver.setdefault(d, []).append(lap)

    result = []
    for driver_laps in by_driver.values():
        driver_laps.sort(key=lambda x: x["lap_number"])
        current_compound = None
        stint_lap = 0
        for lap in driver_laps:
            if lap["compound"] != current_compound:
                current_compound = lap["compound"]
                stint_lap = 0
            lap["stint_lap"] = stint_lap
            stint_lap += 1
            result.append(lap)

    return result


# ── Salva CSV ──────────────────────────────────────────────────────────────────
LAP_FIELDS = [
    "session_key",
    "driver_number",
    "lap_number",
    "lap_duration",
    "compound",
    "stint_lap",
    "is_pit_out_lap",
    "is_sc_lap",
    "air_temp",
    "track_temp",
    "rainfall",
    "humidity",
    "wind_speed",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "i1_speed",
    "i2_speed",
    "st_speed",
]


def save_laps_csv(all_laps: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LAP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_laps)
    logger.info(f"Salvati {len(all_laps)} giri in {path}")


def save_sessions_meta(sessions: list[dict], path: Path):
    with open(path, "w") as f:
        json.dump(sessions, f, indent=2, default=str)
    logger.info(f"Metadata {len(sessions)} sessioni -> {path}")


# ── Pipeline principale ────────────────────────────────────────────────────────
def run(years=YEARS, circuit_filter: str | None = None):
    """
    Scarica dati per tutte le gare (o solo un circuito se specificato).

    Args:
        years: lista anni da scaricare
        circuit_filter: se specificato, filtra per location (es. "Monza")
    """
    sessions = get_all_race_sessions(years)

    if circuit_filter:
        sessions = [
            s
            for s in sessions
            if circuit_filter.lower() in (s.get("location") or "").lower()
            or circuit_filter.lower() in (s.get("circuit_short_name") or "").lower()
        ]
        logger.info(f"Filtro '{circuit_filter}': {len(sessions)} sessioni trovate")

    if not sessions:
        logger.error("Nessuna sessione trovata. Controlla la connessione a api.openf1.org")
        return

    all_laps = []
    sessions_meta = []

    for i, session in enumerate(sessions):
        sk = session.get("session_key")
        location = session.get("location", "?")
        date = (session.get("date_start") or "")[:10]
        logger.info(f"[{i + 1}/{len(sessions)}] session_key={sk}  {location}  {date}")

        try:
            # Scarica laps con compound
            laps = get_laps_with_compound(sk)
            if not laps:
                logger.warning(f"  Nessun giro valido per session {sk}")
                continue

            # Meteo e SC
            weather_data = get_weather_for_session(sk)
            sc_laps = get_race_control_events(sk)

            # Merge e arricchimento
            laps = merge_weather_to_laps(laps, weather_data, sc_laps)
            laps = add_stint_lap(laps)

            # Aggiungi metadata sessione
            for lap in laps:
                lap["year"] = session.get("year")
                lap["location"] = location
                lap["circuit_short_name"] = session.get("circuit_short_name")

            all_laps.extend(laps)
            sessions_meta.append(
                {
                    "session_key": sk,
                    "location": location,
                    "date": date,
                    "year": session.get("year"),
                    "laps_collected": len(laps),
                    "sc_laps": list(sc_laps),
                }
            )

            logger.info(f"  {len(laps)} giri raccolti, {len(sc_laps)} giri SC")
            time.sleep(0.3)  # rispetta rate limit API

        except Exception as e:
            logger.exception(f"  Errore session {sk}: {e}")
            continue

    if not all_laps:
        logger.error("Nessun dato raccolto.")
        return

    # Salva
    prefix = circuit_filter.lower().replace(" ", "_") if circuit_filter else "all_circuits"
    save_laps_csv(all_laps, DATA_DIR / f"{prefix}_laps.csv")
    save_sessions_meta(sessions_meta, DATA_DIR / f"{prefix}_sessions_meta.json")

    logger.info("\nRiepilogo:")
    logger.info(f"  Sessioni processate: {len(sessions_meta)}")
    logger.info(f"  Giri totali: {len(all_laps)}")
    logger.info(f"  Output: {DATA_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scarica dati reali OpenF1 per training ML")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument(
        "--circuit", type=str, default=None, help="Filtra per circuito (es. 'Monza'). Default: tutte le gare."
    )
    args = parser.parse_args()

    run(years=args.years, circuit_filter=args.circuit)
