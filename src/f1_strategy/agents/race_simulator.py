"""
Agent 2: Race Simulator
Simula scenari strategici e calcola tempo gara stimato.
Input: condizioni gara + strategia
Output: tempo totale, confronto scenari, raccomandazione
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Costanti Monza ─────────────────────────────────────────────────────────────
BASE_LAP_TIME = 83.5  # secondi - tempo base giro Monza
SAFETY_CAR_DELTA = 25.0  # secondi persi per SC (giro SC ~40s vs 83s)
SC_LAP_TIME = 108.0  # tempo giro sotto SC
PIT_LANE_DEFAULT = 22.5  # secondi perdita pit lane

# Parametri degrado per compound (secondi/giro di aggiuntivi per ogni giro stint)
COMPOUND_CONFIG = {
    "soft": {
        "base_delta": -0.8,  # più veloce del medium
        "degradation_per_lap": 0.12,  # degrado rapido
        "max_optimal_laps": 20,
        "wet_performance": 1.8,  # molto peggio sul bagnato
    },
    "medium": {
        "base_delta": 0.0,
        "degradation_per_lap": 0.07,
        "max_optimal_laps": 30,
        "wet_performance": 1.2,
    },
    "hard": {
        "base_delta": 0.4,  # leggermente più lento del medium
        "degradation_per_lap": 0.04,
        "max_optimal_laps": 42,
        "wet_performance": 1.5,
    },
    "intermediate": {
        "base_delta": 2.0,  # molto lento su asciutto
        "degradation_per_lap": 0.03,
        "max_optimal_laps": 40,
        "wet_performance": -1.5,  # MOLTO meglio su bagnato leggero
    },
    "wet": {
        "base_delta": 5.0,  # inutilizzabile su asciutto
        "degradation_per_lap": 0.02,
        "max_optimal_laps": 45,
        "wet_performance": -3.0,  # ideale su bagnato intenso
    },
}

# Impatto meteo sul tempo giro (secondi aggiuntivi)
WEATHER_IMPACT = {
    "dry": 0.0,
    "light_rain": 3.5,  # pioggia leggera
    "heavy_rain": 8.0,
}


@dataclass
class StintSpec:
    stint: int
    compound: str
    start_lap: int
    end_lap: int = 0  # calcolato automaticamente


@dataclass
class LapResult:
    lap: int
    lap_time: float
    compound: str
    stint_lap: int  # giro all'interno dello stint
    weather: str
    is_pit_lap: bool = False
    is_sc_lap: bool = False
    notes: str = ""


@dataclass
class SimulationResult:
    strategy: list
    total_time: float
    lap_results: list = field(default_factory=list)
    pit_stops: int = 0
    penalties: float = 0.0
    warnings: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


def _get_weather_at_lap(lap: int, conditions: dict) -> str:
    """Determina condizione meteo al giro specificato."""
    weather = conditions.get("weather", {})
    rain_start = weather.get("rain_start_lap", 999)
    rain_intensity = weather.get("rain_intensity", "light")

    if lap >= rain_start:
        if rain_intensity == "heavy":
            return "heavy_rain"
        return "light_rain"
    return "dry"


def _calc_lap_time(
    compound: str,
    stint_lap: int,
    weather: str,
    base_time: float = BASE_LAP_TIME,
) -> float:
    """
    Calcola tempo singolo giro con degrado e meteo.

    Formula:
      lap_time = base + compound_delta + (degradation_per_lap * stint_lap) + weather_adj
    """
    cfg = COMPOUND_CONFIG.get(compound, COMPOUND_CONFIG["medium"])

    # Delta compound
    compound_delta = cfg["base_delta"]

    # Degrado progressivo (accelera dopo max_optimal_laps)
    opt_laps = cfg["max_optimal_laps"]
    deg_rate = cfg["degradation_per_lap"]

    if stint_lap <= opt_laps:
        degradation = deg_rate * stint_lap
    else:
        # Degrado accelerato oltre il cliff
        extra = stint_lap - opt_laps
        degradation = (deg_rate * opt_laps) + (deg_rate * 2.5 * extra)

    # Impatto meteo: wet_perf negativo = compound beneficia della pioggia
    wet_perf = cfg.get("wet_performance", 0.0)
    if weather == "dry":
        weather_adj = 0.0
    elif weather == "light_rain":
        # es. intermediate: 3.5 + (-1.5) = 2.0s  → meglio di slick che paga 3.5+1.2=4.7s
        weather_adj = WEATHER_IMPACT["light_rain"] + wet_perf
    else:
        weather_adj = WEATHER_IMPACT["heavy_rain"] + wet_perf

    lap_time = base_time + compound_delta + degradation + weather_adj
    return round(lap_time, 3)


def simulate_strategy(strategy: list, conditions: dict) -> SimulationResult:
    """
    Simula una strategia completa giro per giro.

    Args:
        strategy: lista di dict {"stint", "compound", "start_lap"}
        conditions: race_conditions.json

    Returns:
        SimulationResult con tempo totale e dettagli
    """
    total_laps = conditions.get("total_laps", 53)
    pit_time_loss = conditions.get("pit_lane_time_loss_seconds", PIT_LANE_DEFAULT)
    sc_info = conditions.get("safety_car", {})
    sc_active = sc_info.get("active", False)
    sc_lap = sc_info.get("lap", -1)
    sc_duration = sc_info.get("duration_laps", 0)

    # Normalizza strategia
    stints = []
    sorted_strat = sorted(strategy, key=lambda x: x["start_lap"])
    for i, s in enumerate(sorted_strat):
        end_lap = sorted_strat[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_strat) else total_laps
        stints.append(
            StintSpec(
                stint=s["stint"],
                compound=s["compound"].lower(),
                start_lap=s["start_lap"],
                end_lap=end_lap,
            )
        )

    # Validazione
    warnings = []
    if stints[0].start_lap != 1:
        warnings.append("Il primo stint non inizia dal giro 1")
    if stints[-1].end_lap != total_laps:
        warnings.append(f"L'ultimo stint non arriva al giro {total_laps}")

    # Costruisci mappa stint per giro
    lap_to_stint = {}
    for st in stints:
        for lap in range(st.start_lap, st.end_lap + 1):
            lap_to_stint[lap] = st

    # Simula giro per giro
    total_time = 0.0
    lap_results = []
    pit_stops = 0

    for lap in range(1, total_laps + 1):
        stint = lap_to_stint.get(lap)
        if not stint:
            warnings.append(f"Giro {lap} non coperto da nessuno stint!")
            total_time += BASE_LAP_TIME + 5  # penalità
            continue

        stint_lap = lap - stint.start_lap  # 0-indexed
        weather = _get_weather_at_lap(lap, conditions)

        # Giro Safety Car?
        is_sc = sc_active and sc_lap <= lap < sc_lap + sc_duration

        if is_sc:
            lap_time = SC_LAP_TIME
            note = "SC"
        else:
            lap_time = _calc_lap_time(stint.compound, stint_lap, weather)
            note = ""

        # Pit stop (cambio stint)
        is_pit = lap == stint.start_lap and lap > 1
        if is_pit:
            total_time += pit_time_loss
            pit_stops += 1
            note = f"PIT +{pit_time_loss}s"

        total_time += lap_time
        lap_results.append(
            LapResult(
                lap=lap,
                lap_time=round(lap_time, 3),
                compound=stint.compound,
                stint_lap=stint_lap,
                weather=weather,
                is_pit_lap=is_pit,
                is_sc_lap=is_sc,
                notes=note,
            )
        )

    # Breakdown
    time_by_compound = {}
    for lr in lap_results:
        c = lr.compound
        time_by_compound[c] = time_by_compound.get(c, 0) + lr.lap_time

    breakdown = {
        "time_by_compound": {k: round(v, 2) for k, v in time_by_compound.items()},
        "total_pit_time": round(pit_stops * pit_time_loss, 2),
        "sc_laps": sum(1 for lr in lap_results if lr.is_sc_lap),
    }

    return SimulationResult(
        strategy=strategy,
        total_time=round(total_time, 2),
        lap_results=[vars(lr) for lr in lap_results],
        pit_stops=pit_stops,
        warnings=warnings,
        breakdown=breakdown,
    )


def generate_candidate_strategies(conditions: dict, historical_patterns: dict | None = None) -> list:
    """
    Genera strategie candidate da confrontare.
    Considera condizioni meteo, SC, e pattern storici.
    """
    total_laps = conditions.get("total_laps", 53)
    rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
    conditions.get("weather", {}).get("rain_intensity", "none")
    sc_lap = conditions.get("safety_car", {}).get("lap", -1) if conditions.get("safety_car", {}).get("active") else -1

    candidates = []

    # ── STRATEGIE DRY ────────────────────────────────────────────────────────
    # 1-stop classico Monza: Medium → Hard
    candidates.append(
        {
            "name": "1-stop M→H (classico)",
            "strategy": [
                {"stint": 1, "compound": "medium", "start_lap": 1},
                {"stint": 2, "compound": "hard", "start_lap": 27},
            ],
        }
    )

    # 1-stop aggresivo: Soft → Hard
    candidates.append(
        {
            "name": "1-stop S→H (aggressivo)",
            "strategy": [
                {"stint": 1, "compound": "soft", "start_lap": 1},
                {"stint": 2, "compound": "hard", "start_lap": 18},
            ],
        }
    )

    # 1-stop Hard → Medium
    candidates.append(
        {
            "name": "1-stop H→M (conservativo)",
            "strategy": [
                {"stint": 1, "compound": "hard", "start_lap": 1},
                {"stint": 2, "compound": "medium", "start_lap": 30},
            ],
        }
    )

    # 2-stop S→M→H
    candidates.append(
        {
            "name": "2-stop S→M→H",
            "strategy": [
                {"stint": 1, "compound": "soft", "start_lap": 1},
                {"stint": 2, "compound": "medium", "start_lap": 15},
                {"stint": 3, "compound": "hard", "start_lap": 33},
            ],
        }
    )

    # 2-stop M→H→M undercut
    candidates.append(
        {
            "name": "2-stop M→H→M (undercut SC)",
            "strategy": [
                {"stint": 1, "compound": "medium", "start_lap": 1},
                {"stint": 2, "compound": "hard", "start_lap": 15},  # pit durante SC
                {"stint": 3, "compound": "medium", "start_lap": 35},
            ],
        }
    )

    # ── STRATEGIE CON PIOGGIA (condizioni variabili) ─────────────────────────
    if rain_start and rain_start < total_laps:
        # pit prima della pioggia o al SC
        pit1 = max(15, rain_start - 3)
        if sc_lap > 0:
            pit1 = sc_lap  # Sfrutta SC per pit (gratis!)

        # 2-stop con intermediate per pioggia (OBBLIGATORIO con pioggia reale)
        candidates.append(
            {
                "name": "Meteo: M→Inter@pioggia (2 stint)",
                "strategy": [
                    {"stint": 1, "compound": "medium", "start_lap": 1},
                    {"stint": 2, "compound": "intermediate", "start_lap": rain_start},
                ],
            }
        )

        # 3-stop S→H→Inter: partenza aggressiva, hard nel mezzo, inter per pioggia
        candidates.append(
            {
                "name": "Meteo OTTIMALE: S→H@SC→Inter@pioggia",
                "strategy": [
                    {"stint": 1, "compound": "soft", "start_lap": 1},
                    {"stint": 2, "compound": "hard", "start_lap": pit1},
                    {"stint": 3, "compound": "intermediate", "start_lap": rain_start},
                ],
            }
        )

        # Pit al SC + inter per pioggia (la più intelligente con SC+pioggia)
        if sc_lap > 0 and sc_lap < rain_start:
            candidates.append(
                {
                    "name": "Meteo SMART: M→H@SC→Inter@pioggia",
                    "strategy": [
                        {"stint": 1, "compound": "medium", "start_lap": 1},
                        {"stint": 2, "compound": "hard", "start_lap": sc_lap},
                        {"stint": 3, "compound": "intermediate", "start_lap": rain_start},
                    ],
                }
            )
            # Variante: soft start per più grip iniziale
            candidates.append(
                {
                    "name": "Meteo SMART+: S→M@SC→Inter@pioggia",
                    "strategy": [
                        {"stint": 1, "compound": "soft", "start_lap": 1},
                        {"stint": 2, "compound": "medium", "start_lap": sc_lap},
                        {"stint": 3, "compound": "intermediate", "start_lap": rain_start},
                    ],
                }
            )

        # Scommessa: rimani su dry e sperare che asciughi (pioggia leggera)
        if conditions.get("weather", {}).get("rain_intensity") == "light":
            candidates.append(
                {
                    "name": "Scommessa: M→H ignora pioggia leggera",
                    "strategy": [
                        {"stint": 1, "compound": "medium", "start_lap": 1},
                        {"stint": 2, "compound": "hard", "start_lap": 28},
                    ],
                }
            )

    return candidates


def find_optimal_strategy(conditions: dict, historical_patterns: dict | None = None) -> dict:
    """
    Confronta tutte le strategie candidate e restituisce la migliore.
    """
    candidates = generate_candidate_strategies(conditions, historical_patterns)

    results = []
    for candidate in candidates:
        sim = simulate_strategy(candidate["strategy"], conditions)
        results.append(
            {
                "name": candidate["name"],
                "strategy": candidate["strategy"],
                "total_time": sim.total_time,
                "pit_stops": sim.pit_stops,
                "warnings": sim.warnings,
                "breakdown": sim.breakdown,
                "lap_results": sim.lap_results,
            }
        )
        logger.info(f"  {candidate['name']}: {sim.total_time:.2f}s ({sim.pit_stops} stop)")

    # Ordina per tempo totale
    results.sort(key=lambda x: x["total_time"])

    best = results[0]
    logger.info(f"Strategia ottimale: {best['name']} → {best['total_time']:.2f}s")

    return {
        "optimal": best,
        "all_scenarios": results,
        "ranking": [
            {
                "rank": i + 1,
                "name": r["name"],
                "total_time": r["total_time"],
                "delta": round(r["total_time"] - results[0]["total_time"], 2),
            }
            for i, r in enumerate(results)
        ],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    conditions = {
        "circuit": "Monza",
        "total_laps": 53,
        "weather": {
            "type": "variable",
            "rain_start_lap": 30,
            "rain_intensity": "light",
        },
        "safety_car": {"active": True, "lap": 15, "duration_laps": 3},
        "grid_position": 6,
        "pit_lane_time_loss_seconds": 22.5,
    }

    result = find_optimal_strategy(conditions)
    print(json.dumps(result["ranking"], indent=2))
