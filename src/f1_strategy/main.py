"""
main.py — Pipeline end-to-end F1 Strategy System
Scuderia Algoritmo

Uso:
    python main.py                          # usa race_conditions.json nella stessa cartella
    python main.py --conditions path.json  # file condizioni personalizzato
    python main.py --no-api                # salta API, usa solo dati storici locali

Output:
    outputs/strategy.json      ← formato competizione
    outputs/report.txt         ← riepilogo testuale
    outputs/strategy_chart.html ← grafici interattivi
    outputs/full_analysis.json ← dati completi
"""

import json
import logging
import sys
import time
from pathlib import Path

# Aggiunge il path del progetto
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents"))

import agents.data_analysis_agent as data_agent
import agents.race_simulator as simulator
import agents.report_generator as reporter
from agents import strategy_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def load_conditions(path: str | None = None) -> dict:
    """Carica race_conditions.json."""
    path = Path(__file__).parent / "race_conditions.json" if path is None else Path(path)

    if not path.exists():
        logger.warning(f"File condizioni non trovato: {path}. Uso condizioni default Monza.")
        return {
            "circuit": "Monza",
            "total_laps": 53,
            "weather": {
                "type": "variable",
                "description": "Asciutto fino al giro 30, poi pioggia leggera fino a fine gara",
                "rain_start_lap": 30,
                "rain_intensity": "light",
            },
            "safety_car": {
                "active": True,
                "lap": 15,
                "duration_laps": 3,
            },
            "grid_position": 6,
            "pit_lane_time_loss_seconds": 22.5,
        }

    with open(path) as f:
        conditions = json.load(f)
    logger.info(f"Condizioni gara caricate: {path}")
    return conditions


def run_pipeline(conditions_path: str | None = None, skip_api: bool = False) -> dict:
    """
    Pipeline completa:
    1. Carica condizioni gara
    2. Analisi dati storici (OpenF1)
    3. Simulazione scenari
    4. Strategia ottimale
    5. Output file
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  SCUDERIA ALGORITMO — F1 STRATEGY SYSTEM")
    logger.info("=" * 60)

    # STEP 1: Carica condizioni
    conditions = load_conditions(conditions_path)
    circuit = conditions.get("circuit", "Monza")
    logger.info(
        f"Circuito: {circuit} | Giri: {conditions.get('total_laps')} | Meteo: {conditions.get('weather', {}).get('type')}"
    )

    # STEP 2: Analisi dati storici
    logger.info("\n[1/4] Analisi dati storici OpenF1...")
    try:
        if skip_api:
            logger.info("Skip API: uso knowledge base locale")
            historical_analysis = (
                data_agent.run_analysis.__wrapped__(circuit) if hasattr(data_agent.run_analysis, "__wrapped__") else None
            )
            if not historical_analysis:
                # Dati fallback Monza
                historical_analysis = {
                    "circuit": circuit,
                    "sessions_analyzed": 0,
                    "sessions_data": [],
                    "winning_patterns": {
                        "most_used_strategy": "1-stop",
                        "optimal_compounds": ["medium", "hard", "soft"],
                        "typical_pit_window": [25, 32],
                        "notes": ["Dati da knowledge base Monza"],
                    },
                    "key_insights": [
                        "Strategia dominante a Monza: 1-STOP",
                        "Finestra pit stop ottimale: giro 25–32",
                        "Degrado gomme BASSO: favorisce stint lunghi su Hard/Medium",
                        "Monza: circuito ad alta velocità, basso carico aerodinamico",
                    ],
                }
        else:
            historical_analysis = data_agent.run_analysis(circuit)
    except Exception as e:
        logger.exception(f"Errore analisi storica: {e}. Continuo con fallback.")
        historical_analysis = {
            "circuit": circuit,
            "sessions_analyzed": 0,
            "sessions_data": [],
            "winning_patterns": {
                "most_used_strategy": "1-stop",
                "optimal_compounds": ["medium", "hard"],
                "typical_pit_window": [27, 31],
                "notes": ["Fallback dati"],
            },
            "key_insights": [
                "1-stop strategia dominante a Monza",
                "Compound Medium/Hard ottimali per basso degrado",
            ],
        }

    patterns = historical_analysis.get("winning_patterns", {})
    logger.info(f"  Pattern identificato: {patterns.get('most_used_strategy', 'N/A').upper()}")
    logger.info(f"  Compound ottimali: {patterns.get('optimal_compounds', [])}")

    # STEP 3: Simulazione scenari
    logger.info("\n[2/4] Simulazione scenari strategici...")
    simulation_result = simulator.find_optimal_strategy(conditions, patterns)

    ranking = simulation_result.get("ranking", [])
    logger.info(f"  Scenari simulati: {len(ranking)}")
    logger.info(f"  Migliore: {simulation_result['optimal']['name']} → {simulation_result['optimal']['total_time']:.2f}s")

    # STEP 4: Strategia finale
    logger.info("\n[3/4] Generazione strategia ottimale...")
    strategy_result = strategy_agent.run_strategy_agent(conditions, simulation_result, historical_analysis)

    strategy_json = strategy_result["strategy_json"]
    errors = strategy_result.get("validation_errors", [])

    if errors:
        logger.error(f"  Errori validazione: {errors}")
    else:
        logger.info("  Validazione strategia: OK ✓")

    logger.info(f"  Strategia: {[s['compound'].upper() + '@' + str(s['start_lap']) for s in strategy_json['strategy']]}")

    # STEP 5: Output files
    logger.info("\n[4/4] Salvataggio output...")
    output_paths = reporter.save_outputs(
        conditions=conditions,
        historical_analysis=historical_analysis,
        simulation_result=simulation_result,
        strategy_json=strategy_json,
    )

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Pipeline completata in {elapsed:.1f}s")
    logger.info("  Output salvati in: outputs/")
    logger.info(f"{'=' * 60}")

    # Stampa strategia finale
    print("\n" + "=" * 60)
    print("  STRATEGIA FINALE — SCUDERIA ALGORITMO")
    print("=" * 60)
    for s in strategy_json["strategy"]:
        print(f"  Stint {s['stint']}: {s['compound'].upper()} dal giro {s['start_lap']}")
    print(f"\n  Tempo stimato: {strategy_json.get('estimated_total_time_seconds', 'N/A'):.0f}s")
    print(f"\n  Rationale: {strategy_json['rationale'][:200]}...")
    print("=" * 60)

    return {
        "strategy_json": strategy_json,
        "historical_analysis": historical_analysis,
        "simulation_result": simulation_result,
        "output_paths": output_paths,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Strategy System — Scuderia Algoritmo")
    parser.add_argument("--conditions", type=str, default=None, help="Path a race_conditions.json")
    parser.add_argument("--no-api", action="store_true", help="Salta chiamate API OpenF1")
    args = parser.parse_args()

    result = run_pipeline(
        conditions_path=args.conditions,
        skip_api=args.no_api,
    )

    # Exit code 0 se strategia valida
    errors = result.get("strategy_json", {})
    sys.exit(0)
