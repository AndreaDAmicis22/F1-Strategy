"""
race_simulator.py
=================
Genera strategie candidate e le valuta usando i modelli ML (ml_predictor.py).
NON contiene più coefficienti hardcoded: tutti i tempi sono stimati dal
LapTimePredictor addestrato su dati reali OpenF1.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SC_LAP_TIME = 108.0  # unico valore fisso: giro sotto Safety Car (misurato)
PIT_LANE_DEFAULT = 22.5  # default se non specificato in race_conditions.json


@dataclass
class SimulationResult:
    strategy: list
    total_time: float
    lap_results: list = field(default_factory=list)
    pit_stops: int = 0
    warnings: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


def generate_candidate_strategies(conditions: dict, historical_patterns: dict | None = None) -> list:
    """
    Genera tutte le strategie candidate da valutare.
    Non calcola nessun tempo — si limita a definire sequenze di stint.
    La valutazione è delegata interamente a StrategyEvaluator.evaluate_strategy().
    """
    total_laps = conditions.get("total_laps", 53)
    rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
    rain_intensity = conditions.get("weather", {}).get("rain_intensity", "none")
    sc_lap = conditions.get("safety_car", {}).get("lap", -1) if conditions.get("safety_car", {}).get("active") else -1

    candidates = []

    # ── STRATEGIE DRY ──────────────────────────────────────────────────────────
    candidates.append(
        {
            "name": "1-stop M→H",
            "strategy": [
                {"stint": 1, "compound": "medium", "start_lap": 1},
                {"stint": 2, "compound": "hard", "start_lap": 27},
            ],
        }
    )
    candidates.append(
        {
            "name": "1-stop S→H",
            "strategy": [
                {"stint": 1, "compound": "soft", "start_lap": 1},
                {"stint": 2, "compound": "hard", "start_lap": 18},
            ],
        }
    )
    candidates.append(
        {
            "name": "1-stop H→M",
            "strategy": [
                {"stint": 1, "compound": "hard", "start_lap": 1},
                {"stint": 2, "compound": "medium", "start_lap": 30},
            ],
        }
    )
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
    candidates.append(
        {
            "name": "2-stop M→H→M",
            "strategy": [
                {"stint": 1, "compound": "medium", "start_lap": 1},
                {"stint": 2, "compound": "hard", "start_lap": 15},
                {"stint": 3, "compound": "medium", "start_lap": 35},
            ],
        }
    )

    # Varianti pit window: stessa combinazione compound, pit stop in finestre diverse
    for pit_lap in [22, 25, 27, 30, 32]:
        candidates.append(
            {
                "name": f"1-stop M→H @G{pit_lap}",
                "strategy": [
                    {"stint": 1, "compound": "medium", "start_lap": 1},
                    {"stint": 2, "compound": "hard", "start_lap": pit_lap},
                ],
            }
        )

    # ── STRATEGIE CON PIOGGIA ──────────────────────────────────────────────────
    if rain_start < total_laps:
        pit1 = sc_lap if sc_lap > 0 else max(15, rain_start - 3)

        candidates.append(
            {
                "name": "Meteo: M→Inter@pioggia",
                "strategy": [
                    {"stint": 1, "compound": "medium", "start_lap": 1},
                    {"stint": 2, "compound": "intermediate", "start_lap": rain_start},
                ],
            }
        )
        candidates.append(
            {
                "name": "Meteo: S→H→Inter",
                "strategy": [
                    {"stint": 1, "compound": "soft", "start_lap": 1},
                    {"stint": 2, "compound": "hard", "start_lap": pit1},
                    {"stint": 3, "compound": "intermediate", "start_lap": rain_start},
                ],
            }
        )

        if sc_lap > 0 and sc_lap < rain_start:
            candidates.append(
                {
                    "name": "Meteo: M→H@SC→Inter@pioggia",
                    "strategy": [
                        {"stint": 1, "compound": "medium", "start_lap": 1},
                        {"stint": 2, "compound": "hard", "start_lap": sc_lap},
                        {"stint": 3, "compound": "intermediate", "start_lap": rain_start},
                    ],
                }
            )
            candidates.append(
                {
                    "name": "Meteo: S→M@SC→Inter@pioggia",
                    "strategy": [
                        {"stint": 1, "compound": "soft", "start_lap": 1},
                        {"stint": 2, "compound": "medium", "start_lap": sc_lap},
                        {"stint": 3, "compound": "intermediate", "start_lap": rain_start},
                    ],
                }
            )

        if rain_intensity == "light":
            candidates.append(
                {
                    "name": "Scommessa: ignora pioggia leggera",
                    "strategy": [
                        {"stint": 1, "compound": "medium", "start_lap": 1},
                        {"stint": 2, "compound": "hard", "start_lap": 28},
                    ],
                }
            )

    return candidates


def find_optimal_strategy(
    conditions: dict,
    historical_patterns: dict | None = None,
    ml_evaluator=None,
) -> dict:
    """
    Genera i candidati e li valuta tutti con ML.
    Se ml_evaluator è None (modelli non addestrati), lancia un errore esplicito
    invece di usare coefficienti hardcoded.
    """
    if ml_evaluator is None:
        msg = (
            "ml_evaluator è None: impossibile valutare le strategie senza i modelli ML.\n"
            "Esegui prima:\n"
            "  1. python collect_training_data.py\n"
            "  2. python train_models.py"
        )
        raise RuntimeError(msg)

    candidates = generate_candidate_strategies(conditions, historical_patterns)
    results = []

    for candidate in candidates:
        evaluated = ml_evaluator.evaluate_strategy(candidate["strategy"], conditions)
        results.append(
            {
                "name": candidate["name"],
                "strategy": candidate["strategy"],
                "total_time": evaluated["total_time"],
                "pit_stops": evaluated["pit_stops"],
                "warnings": [],
                "breakdown": evaluated["breakdown"],
                "lap_results": evaluated["lap_results"],
            }
        )
        logger.info(f"  {candidate['name']}: {evaluated['total_time']:.2f}s ({evaluated['pit_stops']} stop)")

    results.sort(key=lambda x: x["total_time"])
    best = results[0]
    logger.info(f"Strategia ottimale (ML): {best['name']} → {best['total_time']:.2f}s")

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
