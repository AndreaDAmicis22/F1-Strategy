"""
main.py — Pipeline end-to-end F1 Strategy System
Scuderia Algoritmo

Uso:
    python main.py
    python main.py --conditions path/to/race_conditions.json
    python main.py --no-api
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents"))

import agents.data_analysis_agent as data_agent
import agents.race_simulator as simulator
import agents.report_generator as reporter
from agents import strategy_agent
from agents.ml_predictor import ModelNotTrainedError, StrategyEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

DEFAULT_CONDITIONS = {
    "circuit": "Monza",
    "total_laps": 53,
    "weather": {
        "type": "variable",
        "description": "Asciutto fino al giro 30, poi pioggia leggera fino a fine gara",
        "rain_start_lap": 30,
        "rain_intensity": "light",
    },
    "safety_car": {"active": True, "lap": 15, "duration_laps": 3},
    "grid_position": 6,
    "pit_lane_time_loss_seconds": 22.5,
}


def load_conditions(path=None):
    path = Path(__file__).parent / "race_conditions.json" if path is None else Path(path)
    if not path.exists():
        logger.warning(f"File condizioni non trovato: {path}. Uso condizioni default.")
        return DEFAULT_CONDITIONS
    with open(path) as f:
        conditions = json.load(f)
    logger.info(f"Condizioni gara caricate: {path}")
    return conditions


def _historical_fallback(circuit):
    return {
        "circuit": circuit,
        "sessions_analyzed": 0,
        "sessions_data": [],
        "winning_patterns": {
            "most_used_strategy": "1-stop",
            "optimal_compounds": ["medium", "hard", "soft"],
            "typical_pit_window": [25, 32],
            "notes": ["Fallback knowledge base"],
        },
        "key_insights": [],
    }


def run_pipeline(conditions_path=None, skip_api=False):
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  SCUDERIA ALGORITMO - F1 STRATEGY SYSTEM")
    logger.info("=" * 60)

    # STEP 0: Carica condizioni
    conditions = load_conditions(conditions_path)
    circuit = conditions.get("circuit", "Monza")
    logger.info(
        f"Circuito: {circuit} | Giri: {conditions.get('total_laps')} | Meteo: {conditions.get('weather', {}).get('type')}"
    )

    # STEP 1: Carica modelli ML — obbligatori, senza fallback a coefficienti hardcoded
    logger.info("\n[ML] Caricamento modelli ML...")
    try:
        ml_evaluator = StrategyEvaluator.load()
        ml_report = ml_evaluator.get_models_info()
        metrics = ml_report.get("lap_time_predictor", {}).get("metrics", {})
        logger.info(f"  LapTimePredictor: MAE={metrics.get('MAE', '?')}s  R²={metrics.get('R2', '?')}")

        ml_recommendation = ml_evaluator.recommend_strategy(conditions)
        logger.info("  Raccomandazione ML compound:")
        for s in ml_recommendation.get("ml_recommended_stints", []):
            logger.info(
                f"    Stint {s['stint']} (G.{s['start_lap']}): "
                f"{s['recommended_compound'].upper()} [conf={s['confidence']:.0%}]"
            )
    except ModelNotTrainedError as e:
        logger.exception(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Errore caricamento ML: {e}")
        sys.exit(1)

    # STEP 2: Analisi dati storici
    logger.info("\n[1/4] Analisi dati storici OpenF1...")
    try:
        historical_analysis = _historical_fallback(circuit) if skip_api else data_agent.run_analysis(circuit)
    except Exception as e:
        logger.warning(f"Analisi storica fallita: {e}. Uso fallback.")
        historical_analysis = _historical_fallback(circuit)

    patterns = historical_analysis.get("winning_patterns", {})
    logger.info(f"  Pattern identificato: {patterns.get('most_used_strategy', 'N/A').upper()}")

    # STEP 3: Simulazione scenari — interamente con ML
    logger.info("\n[2/4] Simulazione scenari strategici con ML...")
    simulation_result = simulator.find_optimal_strategy(
        conditions,
        historical_patterns=patterns,
        ml_evaluator=ml_evaluator,  # ← ML decide tutto
    )

    ranking = simulation_result.get("ranking", [])
    logger.info(f"  Scenari valutati: {len(ranking)}")
    logger.info(
        f"  Migliore (ML): {simulation_result['optimal']['name']} → {simulation_result['optimal']['total_time']:.2f}s"
    )

    # STEP 4: Strategia finale
    logger.info("\n[3/4] Generazione strategia ottimale...")
    strategy_result = strategy_agent.run_strategy_agent(conditions, simulation_result, historical_analysis)
    strategy_json = strategy_result["strategy_json"]
    errors = strategy_result.get("validation_errors", [])

    if errors:
        logger.error(f"  Errori validazione: {errors}")
    else:
        logger.info("  Validazione strategia: OK ✓")

    logger.info(f"  Strategia: {[s['compound'].upper() + '@G' + str(s['start_lap']) for s in strategy_json['strategy']]}")

    # STEP 5: Output files
    logger.info("\n[4/4] Salvataggio output...")
    output_dir = Path(__file__).parent / "outputs"
    output_paths = reporter.save_outputs(
        conditions=conditions,
        historical_analysis=historical_analysis,
        simulation_result=simulation_result,
        strategy_json=strategy_json,
        output_dir=output_dir,
    )

    ml_path = output_dir / "ml_report.json"
    ml_path.write_text(json.dumps(ml_report, indent=2, default=str))
    output_paths["ml_report"] = str(ml_path)
    logger.info(f"  ✓ ml_report.json → {ml_path}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Pipeline completata in {elapsed:.1f}s")
    logger.info(f"{'=' * 60}")

    print("\n" + "=" * 60)
    print("  STRATEGIA FINALE - SCUDERIA ALGORITMO")
    print("=" * 60)
    for s in strategy_json["strategy"]:
        print(f"  Stint {s['stint']}: {s['compound'].upper()} dal giro {s['start_lap']}")
    print(f"\n  Tempo stimato (ML): {simulation_result['optimal']['total_time']:.0f}s")
    print(f"\n  Rationale: {strategy_json['rationale'][:300]}")
    print("=" * 60)

    return {
        "strategy_json": strategy_json,
        "historical_analysis": historical_analysis,
        "simulation_result": simulation_result,
        "ml_report": ml_report,
        "ml_recommendation": ml_recommendation,
        "output_paths": output_paths,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Strategy System - Scuderia Algoritmo")
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        conditions_path=args.conditions,
        skip_api=args.no_api,
    )
    sys.exit(0)
