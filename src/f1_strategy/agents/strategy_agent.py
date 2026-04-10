"""
Agent 3: Strategy Agent
Sintetizza analisi dati storici + simulazione → strategia finale ottimale
Produce il file strategy.json nel formato richiesto
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEAM_NAME = "Scuderia Algoritmo"


def _build_rationale(optimal: dict, conditions: dict, patterns: dict, analysis: dict) -> str:
    """Genera spiegazione strategica human-readable."""
    strategy = optimal["strategy"]
    compounds = [s["compound"] for s in strategy]
    pit_laps = [s["start_lap"] for s in strategy if s["start_lap"] > 1]
    total_time = optimal["total_time"]

    rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
    conditions.get("weather", {}).get("rain_intensity", "none")
    sc_lap = conditions.get("safety_car", {}).get("lap", -1) if conditions.get("safety_car", {}).get("active") else -1
    total_laps = conditions.get("total_laps", 53)
    grid = conditions.get("grid_position", 1)

    parts = []

    # Descrizione base
    n_stops = len(strategy) - 1
    stops_str = "1 pit stop" if n_stops == 1 else f"{n_stops} pit stop"
    strategy_str = " → ".join(c.upper() for c in compounds)
    parts.append(f"Strategia {stops_str} ({strategy_str}), stimato {total_time:.0f}s di gara totale.")

    # Logica pit stop
    for _i, lap in enumerate(pit_laps):
        if sc_lap > 0 and abs(lap - sc_lap) <= 2:
            parts.append(
                f"Pit stop al giro {lap} sincronizzato con Safety Car (giro {sc_lap}): azzeramento del gap di pit lane."
            )
        elif lap == rain_start or abs(lap - rain_start) <= 2:
            parts.append(f"Cambio al giro {lap} anticipato per transizione meteo (pioggia attesa dal giro {rain_start}).")
        else:
            parts.append(f"Pit stop al giro {lap} nella finestra strategica ottimale per Monza.")

    # Scelta compound
    if "intermediate" in compounds:
        parts.append(
            f"Gomme intermedie per la fase di pioggia leggera (giri {rain_start}–{total_laps}): adattamento alle condizioni variabili senza rischiare il Full Wet."
        )

    if "soft" in compounds[:1]:
        parts.append("Partenza su Soft per sfruttare la posizione di griglia e creare gap nei primi giri.")
    elif "medium" in compounds[:1]:
        parts.append("Partenza su Medium per bilanciare performance iniziale e stint lungo.")

    # SC e gestione gara
    if sc_lap > 0:
        parts.append(f"Safety Car al giro {sc_lap} sfruttata strategicamente per ridurre il costo del pit stop al minimo.")

    if grid >= 5:
        parts.append(f"Partenza dalla posizione {grid}: strategia differenziata per gain tramite undercut.")

    # Pattern storici
    if patterns:
        hist_strategy = patterns.get("most_used_strategy", "1-stop")
        parts.append(
            f"Confermato dai dati storici OpenF1: Monza favorisce strategia {hist_strategy} per il basso degrado gomme su asfalto ad alta velocità."
        )

    return " ".join(parts)


def produce_strategy_json(conditions: dict, simulation_result: dict, historical_analysis: dict | None = None) -> dict:
    """
    Produce il file strategy.json nel formato richiesto dalla competizione.
    """
    optimal = simulation_result["optimal"]
    patterns = (historical_analysis or {}).get("winning_patterns", {})

    rationale = _build_rationale(optimal, conditions, patterns, historical_analysis or {})

    return {
        "team_name": TEAM_NAME,
        "strategy": optimal["strategy"],
        "rationale": rationale,
        "estimated_total_time_seconds": optimal["total_time"],
    }


def validate_strategy(strategy_json: dict, total_laps: int) -> list:
    """Valida formato e logica della strategy.json."""
    errors = []
    stints = strategy_json.get("strategy", [])

    if not stints:
        errors.append("strategy vuota")
        return errors

    valid_compounds = {"soft", "medium", "hard", "intermediate", "wet"}

    for i, s in enumerate(stints):
        if s.get("compound", "").lower() not in valid_compounds:
            errors.append(f"Stint {i + 1}: compound non valido '{s.get('compound')}'")
        if "start_lap" not in s:
            errors.append(f"Stint {i + 1}: manca start_lap")
        if "stint" not in s:
            errors.append(f"Stint {i + 1}: manca campo stint")

    if stints[0].get("start_lap") != 1:
        errors.append("Il primo stint deve iniziare al giro 1")

    # Controlla copertura totale
    for i in range(len(stints) - 1):
        curr_start = stints[i].get("start_lap", 0)
        next_start = stints[i + 1].get("start_lap", 0)
        if next_start <= curr_start:
            errors.append(f"Stint {i + 2}: start_lap deve essere > start_lap stint precedente")

    return errors


def run_strategy_agent(conditions: dict, simulation_result: dict, historical_analysis: dict | None = None) -> dict:
    """Entry point principale dell'agente strategia."""
    logger.info("=== Agente Strategia: produzione strategia finale ===")

    strategy_json = produce_strategy_json(conditions, simulation_result, historical_analysis)

    errors = validate_strategy(strategy_json, conditions.get("total_laps", 53))
    if errors:
        logger.error(f"Errori validazione: {errors}")
    else:
        logger.info("Validazione strategia: OK ✓")

    logger.info(f"Strategia finale: {strategy_json['strategy']}")
    logger.info(f"Tempo stimato: {strategy_json['estimated_total_time_seconds']}s")

    return {
        "strategy_json": strategy_json,
        "validation_errors": errors,
        "simulation_ranking": simulation_result.get("ranking", []),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from f1_strategy.agents import race_simulator

    conditions = {
        "circuit": "Monza",
        "total_laps": 53,
        "weather": {"type": "variable", "rain_start_lap": 30, "rain_intensity": "light"},
        "safety_car": {"active": True, "lap": 15, "duration_laps": 3},
        "grid_position": 6,
        "pit_lane_time_loss_seconds": 22.5,
    }

    sim = race_simulator.find_optimal_strategy(conditions)
    result = run_strategy_agent(conditions, sim)
    print(json.dumps(result["strategy_json"], indent=2, ensure_ascii=False))
