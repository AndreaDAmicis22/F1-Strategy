"""
main.py — F1 Strategy Validator & Ranker
=========================================
Legge race_conditions.json e una o più strategy.json,
le valida e le classifica usando modelli ML addestrati su dati reali OpenF1.

Uso:
    # Valida una singola strategia
    poetry run python main.py --strategy outputs/strategy.json

    # Classifica più strategie (una cartella con tutti i team)
    poetry run python main.py --strategies-dir outputs/teams/

    # Senza chiamate API OpenF1
    poetry run python main.py --strategy outputs/strategy.json --no-api
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.ml_predictor import ModelNotTrainedError, StrategyEvaluator, StrategyValidator

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
        "rain_intensity": "none",
    },
    "safety_car": {"active": True, "lap": 15, "duration_laps": 3},
    "grid_position": 6,
    "pit_lane_time_loss_seconds": 22.5,
}


def load_conditions(path=None) -> dict:
    path = Path(__file__).parent.parent.parent / "inputs" / "race_conditions.json" if path is None else Path(path)
    logger.info(f"Caricamento condizioni da: {path}")
    if not path.exists():
        logger.warning(f"race_conditions.json non trovato: {path}. Uso default Monza.")
        return DEFAULT_CONDITIONS
    with open(path) as f:
        return json.load(f)


def load_strategy(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_strategies_dir(dir_path: Path) -> list[dict]:
    """Carica tutte le strategy.json in una cartella."""
    strategies = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            s = load_strategy(p)
            if "strategy" in s:
                strategies.append(s)
                logger.info(f"  Caricata: {p.name} ({s.get('team_name', '?')})")
        except Exception as e:
            logger.warning(f"  Errore caricamento {p.name}: {e}")
    return strategies


def _historical_fallback(circuit: str) -> dict:
    return {
        "circuit": circuit,
        "sessions_analyzed": 0,
        "winning_patterns": {"most_used_strategy": "1-stop", "optimal_compounds": ["medium", "hard"]},
        "key_insights": [],
    }


def run_pipeline(
    conditions_path: str | None = None,
    strategy_path: str | None = None,
    strategies_dir: str | None = None,
    skip_api: bool = False,
) -> dict:
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  SCUDERIA ALGORITMO — F1 STRATEGY VALIDATOR")
    logger.info("=" * 60)

    # ── Carica condizioni ──────────────────────────────────────────────────
    conditions = load_conditions(conditions_path)
    circuit = conditions.get("circuit", "Monza")
    logger.info(
        f"Circuito: {circuit} | Giri: {conditions.get('total_laps')} | "
        f"Meteo: {conditions.get('weather', {}).get('type')} | "
        f"SC: giro {conditions.get('safety_car', {}).get('lap', 'N/A')}"
    )

    # ── Carica strategie da validare ───────────────────────────────────────
    strategies = []
    if strategies_dir:
        logger.info(f"\nCaricamento strategie da: {strategies_dir}")
        strategies = load_strategies_dir(Path(strategies_dir))
    elif strategy_path:
        logger.info(f"\nCaricamento strategia: {strategy_path}")
        strategies = [load_strategy(Path(strategy_path))]
    else:
        default_path = Path(__file__).parent.parent.parent.parent / "inputs" / "strategy.json"
        if default_path.exists():
            strategies = [load_strategy(default_path)]
            logger.info(f"Strategia caricata da default: {default_path}")
        else:
            logger.error("Nessuna strategia trovata. Usa --strategy o --strategies-dir")
            sys.exit(1)

    if not strategies:
        logger.error("Nessuna strategia valida trovata.")
        sys.exit(1)

    logger.info(f"Strategie da valutare: {len(strategies)}")

    # ── Carica modelli ML ──────────────────────────────────────────────────
    logger.info("\n[ML][1/3] Caricamento modelli ML...")
    try:
        evaluator = StrategyEvaluator.load()
        ml_info = evaluator.get_models_info()
        metrics = ml_info.get("lap_time_predictor", {}).get("metrics", {})
        logger.info(f"  LapTimePredictor: MAE={metrics.get('MAE', '?')}s  R²={metrics.get('R2', '?')}")
        cliffs = ml_info.get("degradation_risk_model", {}).get("cliffs", {})
        logger.info(f"  DegradationRiskModel cliffs: {cliffs}")
        sc_fallback = ml_info.get("sc_impact_model", {}).get("fallback", True)
        logger.info(f"  SafetyCarImpactModel: {'analitico (fallback)' if sc_fallback else 'ML'}")
    except ModelNotTrainedError as e:
        logger.exception(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Errore caricamento ML: {e}")
        sys.exit(1)

    # ── Validazione e ranking ──────────────────────────────────────────────
    logger.info("\n[2/3] Validazione e ranking strategie con ML...")
    validator = StrategyValidator(evaluator)
    ranking = validator.rank_strategies(strategies, conditions)

    for r in ranking:
        status = "✓" if r["valid"] else "✗"
        time_str = f"{r['estimated_total_time']:.0f}s" if r["estimated_total_time"] else "N/A"
        logger.info(
            f"  #{r['rank']} {status} {r['team_name']:<30} "
            f"score={r['scores']['composite']:.1f}  "
            f"tempo={time_str}  "
            f"warnings={len(r['warnings'])}"
        )

    # ── Output ─────────────────────────────────────────────────────────────
    logger.info("\n[3/3] Salvataggio output...")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # ranking.json
    ranking_path = output_dir / "ranking.json"
    ranking_path.write_text(json.dumps(ranking, indent=2, default=str))
    logger.info(f"  ✓ ranking.json → {ranking_path}")

    # ml_report.json
    ml_path = output_dir / "ml_report.json"
    ml_path.write_text(json.dumps(ml_info, indent=2, default=str))
    logger.info(f"  ✓ ml_report.json → {ml_path}")

    # Report HTML
    html = _generate_ranking_html(ranking, conditions)
    html_path = output_dir / "ranking_report.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"  ✓ ranking_report.html → {html_path}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Completato in {elapsed:.1f}s")

    # Stampa riepilogo
    print("\n" + "=" * 60)
    print(f"  RANKING STRATEGIE — {circuit.upper()}")
    print("=" * 60)
    for r in ranking:
        status = "✓" if r["valid"] else "✗ INVALIDA"
        time_str = f"{r['estimated_total_time']:.0f}s" if r["estimated_total_time"] else "N/A"
        print(f"  #{r['rank']} {r['team_name']}")
        print(f"     Score: {r['scores']['composite']:.1f}/100  |  Tempo ML: {time_str}  {status}")
        print(
            f"     Tempi: {r['scores']['time']:.0f}  Degrado: {r['scores']['degradation']:.0f}  "
            f"SC: {r['scores']['sc_timing']:.0f}  Formato: {r['scores']['format']:.0f}"
        )
        if r["warnings"]:
            for w in r["warnings"][:2]:
                print(f"     ⚠ {w}")
        print()
    print("=" * 60)

    return {
        "ranking": ranking,
        "ml_info": ml_info,
        "elapsed": elapsed,
    }


def _generate_ranking_html(ranking: list, conditions: dict) -> str:
    circuit = conditions.get("circuit", "N/A")
    total_laps = conditions.get("total_laps", "N/A")

    rows = ""
    for r in ranking:
        status = "✓" if r["valid"] else "✗"
        color = "#00cc66" if r["valid"] else "#cc3333"
        time_str = f"{r['estimated_total_time']:.0f}s" if r["estimated_total_time"] else "N/A"
        sc_save = r.get("sc_report", {}).get("total_sc_saving_seconds", 0)
        warn_str = "<br>".join(f"⚠ {w}" for w in r.get("warnings", []))
        stints_str = (
            " → ".join(
                f"{s['compound'].upper()}@G{s['start_lap']}"
                for s in sorted(
                    r.get("ml_lap_detail", {}).get("strategy", r.get("ml_lap_detail", {}).get("strategy", [])) or [],
                    key=lambda x: x.get("start_lap", 0),
                )
            )
            if r.get("ml_lap_detail")
            else "N/A"
        )

        rows += f"""
        <tr>
          <td style="color:{color};font-size:1.4em;font-weight:bold">{r["rank"]}</td>
          <td><strong>{r["team_name"]}</strong><br><small>{stints_str}</small></td>
          <td style="font-size:1.3em;font-weight:bold;color:#e10600">{r["scores"]["composite"]:.1f}</td>
          <td>{time_str}</td>
          <td>{r["scores"]["time"]:.0f}</td>
          <td>{r["scores"]["degradation"]:.0f}</td>
          <td>{r["scores"]["sc_timing"]:.0f} (+{sc_save:.0f}s)</td>
          <td>{r["scores"]["format"]:.0f}</td>
          <td style="color:{color}">{status}</td>
          <td style="font-size:0.8em;color:#aaa">{warn_str}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>F1 Strategy Ranking — {circuit}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #e10600; text-align: center; letter-spacing: 3px; }}
  h3 {{ color: #888; text-align: center; font-weight: normal; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
  th {{ background: #e10600; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #222; vertical-align: top; }}
  tr:hover td {{ background: #1a1a1a; }}
  .score-bar {{ display: inline-block; height: 6px; background: #e10600;
                border-radius: 3px; margin-left: 6px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>🏎 F1 STRATEGY RANKING</h1>
<h3>{circuit} · {total_laps} giri · Valutazione ML su dati reali OpenF1</h3>
<table>
  <tr>
    <th>#</th><th>Team / Strategia</th><th>Score</th><th>Tempo ML</th>
    <th>Velocità</th><th>Degrado</th><th>SC Timing</th><th>Formato</th>
    <th>Valida</th><th>Note</th>
  </tr>
  {rows}
</table>
<p style="text-align:center;color:#444;margin-top:30px;font-size:0.8em">
  Score composito: 50% tempo · 25% degrado · 15% SC timing · 10% formato
</p>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Strategy Validator — Scuderia Algoritmo")
    parser.add_argument("--conditions", type=str, default=None, help="Path race_conditions.json")
    parser.add_argument("--strategy", type=str, default=None, help="Path a una singola strategy.json")
    parser.add_argument("--strategies-dir", type=str, default=None, help="Cartella con più strategy.json")
    parser.add_argument("--no-api", action="store_true", help="Salta API OpenF1")
    args = parser.parse_args()

    run_pipeline(
        conditions_path=args.conditions,
        strategy_path=args.strategy,
        strategies_dir=args.strategies_dir,
        skip_api=args.no_api,
    )
    sys.exit(0)
