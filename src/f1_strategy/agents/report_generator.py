"""
Agent 4: Report Generator
Produce output grafici e file di riepilogo
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Formatta secondi in mm:ss.xxx"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:06.3f}"


def generate_lap_chart_html(simulation_result: dict, optimal_name: str) -> str:
    """Genera un grafico HTML interattivo dei tempi sul giro."""
    simulation_result.get("all_scenarios", [])
    optimal = simulation_result.get("optimal", {})

    # Prepara dati per il grafico
    lap_data = optimal.get("lap_results", [])

    labels = [str(l["lap"]) for l in lap_data]
    times = [l["lap_time"] for l in lap_data]
    compounds = [l["compound"] for l in lap_data]
    [l["weather"] for l in lap_data]

    compound_colors = {
        "soft": "#FF3333",
        "medium": "#FFFF00",
        "hard": "#FFFFFF",
        "intermediate": "#33FF33",
        "wet": "#3366FF",
    }

    point_colors = json.dumps([compound_colors.get(c, "#888888") for c in compounds])

    ranking_rows = ""
    for r in simulation_result.get("ranking", []):
        delta_str = f"+{r['delta']:.1f}s" if r["delta"] > 0 else "BEST"
        time_fmt = format_time(r["total_time"])
        ranking_rows += f"<tr><td>{r['rank']}</td><td>{r['name']}</td><td>{time_fmt}</td><td>{delta_str}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>F1 Strategy Analysis - Monza</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #e10600; text-align: center; letter-spacing: 2px; font-size: 2em; }}
  h2 {{ color: #aaa; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
  .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; }}
  .card.full {{ grid-column: 1 / -1; }}
  canvas {{ max-height: 350px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #e10600; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #2a2a2a; }}
  tr:hover td {{ background: #222; }}
  .best {{ color: #00ff88; font-weight: bold; }}
  .compound-badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.8em; font-weight: bold; color: #000; }}
  .soft {{ background: #FF3333; color: white; }}
  .medium {{ background: #FFFF00; }}
  .hard {{ background: #ccc; }}
  .intermediate {{ background: #33FF33; }}
  .wet {{ background: #3366FF; color: white; }}
  .summary-box {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .metric {{ background: #111; border: 1px solid #e10600; border-radius: 6px; padding: 12px 20px; flex: 1; min-width: 120px; text-align: center; }}
  .metric .value {{ font-size: 1.8em; color: #e10600; font-weight: bold; }}
  .metric .label {{ font-size: 0.8em; color: #888; margin-top: 4px; }}
</style>
</head>
<body>
<h1>🏎 F1 STRATEGY ANALYSIS — MONZA</h1>
<p style="text-align:center;color:#666">Scuderia Algoritmo — Strategia ottimale</p>

<div class="card full">
  <h2>Strategia Ottimale: {optimal_name}</h2>
  <div class="summary-box">
    <div class="metric">
      <div class="value">{format_time(optimal.get("total_time", 0))}</div>
      <div class="label">Tempo totale stimato</div>
    </div>
    <div class="metric">
      <div class="value">{optimal.get("pit_stops", 0)}</div>
      <div class="label">Pit stop</div>
    </div>
    <div class="metric">
      <div class="value">{len(optimal.get("strategy", []))}</div>
      <div class="label">Stint totali</div>
    </div>
    <div class="metric">
      <div class="value">{", ".join(s["compound"].upper() for s in optimal.get("strategy", []))}</div>
      <div class="label">Compound</div>
    </div>
  </div>
</div>

<div class="grid">
  <div class="card full">
    <h2>Tempi sul Giro — Strategia Ottimale</h2>
    <canvas id="lapChart"></canvas>
  </div>

  <div class="card">
    <h2>Classifica Scenari</h2>
    <table>
      <tr><th>#</th><th>Strategia</th><th>Tempo</th><th>Delta</th></tr>
      {ranking_rows}
    </table>
  </div>

  <div class="card">
    <h2>Distribuzione Tempo per Compound</h2>
    <canvas id="compoundChart"></canvas>
  </div>
</div>

<script>
const labels = {json.dumps(labels)};
const times = {json.dumps(times)};
const pointColors = {point_colors};

new Chart(document.getElementById('lapChart'), {{
  type: 'line',
  data: {{
    labels: labels,
    datasets: [{{
      label: 'Tempo giro (s)',
      data: times,
      borderColor: '#e10600',
      backgroundColor: 'rgba(225,6,0,0.1)',
      fill: true,
      tension: 0.3,
      pointBackgroundColor: pointColors,
      pointRadius: 4,
      pointHoverRadius: 6,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#aaa' }} }},
      tooltip: {{
        callbacks: {{
          afterLabel: (ctx) => 'Compound: ' + {json.dumps(compounds)}[ctx.dataIndex]
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }},
      y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }}, min: 70 }}
    }}
  }}
}});

// Compound breakdown
const breakdown = {json.dumps(optimal.get("breakdown", {}).get("time_by_compound", {}))};
new Chart(document.getElementById('compoundChart'), {{
  type: 'doughnut',
  data: {{
    labels: Object.keys(breakdown),
    datasets: [{{
      data: Object.values(breakdown),
      backgroundColor: Object.keys(breakdown).map(c => ({{
        soft:'#FF3333', medium:'#FFFF00', hard:'#CCCCCC',
        intermediate:'#33FF33', wet:'#3366FF'
      }})[c] || '#888'),
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }}
  }}
}});
</script>
</body>
</html>"""


def generate_text_report(conditions: dict, historical_analysis: dict, simulation_result: dict, strategy_json: dict) -> str:
    """Genera report testuale di riepilogo."""
    optimal = simulation_result.get("optimal", {})
    ranking = simulation_result.get("ranking", [])
    patterns = (historical_analysis or {}).get("winning_patterns", {})
    insights = (historical_analysis or {}).get("key_insights", [])

    lines = [
        "=" * 65,
        "  F1 STRATEGY SYSTEM — SCUDERIA ALGORITMO",
        f"  Circuito: {conditions.get('circuit', 'N/A')} | Giri: {conditions.get('total_laps', 'N/A')}",
        "=" * 65,
        "",
        "── ANALISI DATI STORICI ─────────────────────────────────────",
        f"  Sessioni analizzate: {(historical_analysis or {}).get('sessions_analyzed', 0)}",
        f"  Pattern dominante: {patterns.get('most_used_strategy', 'N/A').upper()}",
        f"  Compound ottimali: {', '.join(patterns.get('optimal_compounds', [])).upper()}",
        "",
        "  Insight chiave:",
    ]

    for insight in insights:
        lines.append(f"  • {insight}")

    lines += [
        "",
        "── CLASSIFICA SCENARI SIMULATI ──────────────────────────────",
    ]

    for r in ranking:
        delta = f"+{r['delta']:.1f}s" if r["delta"] > 0 else " BEST"
        lines.append(f"  {r['rank']:2}. {r['name']:<38} {format_time(r['total_time'])}  {delta}")

    lines += [
        "",
        "── STRATEGIA OTTIMALE ───────────────────────────────────────",
        f"  Nome: {optimal.get('name', 'N/A')}",
        f"  Tempo totale stimato: {format_time(optimal.get('total_time', 0))}",
        f"  Pit stop: {optimal.get('pit_stops', 0)}",
        "",
        "  Stint:",
    ]

    for s in strategy_json.get("strategy", []):
        lines.append(f"  • Stint {s['stint']}: {s['compound'].upper()} dal giro {s['start_lap']}")

    lines += [
        "",
        "  Rationale:",
        f"  {strategy_json.get('rationale', '')}",
        "",
        "=" * 65,
    ]

    return "\n".join(lines)


def save_outputs(
    conditions: dict, historical_analysis: dict, simulation_result: dict, strategy_json: dict, output_dir: Path | None = None
):
    """Salva tutti gli output nella cartella outputs/."""
    if output_dir is None:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        output_dir = BASE_DIR / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. strategy.json (Aggiungi encoding="utf-8")
    strat_path = output_dir / "strategy.json"
    strat_path.write_text(json.dumps(strategy_json, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"✓ strategy.json -> {strat_path}")

    # 2. Report testuale (Aggiungi encoding="utf-8")
    report = generate_text_report(conditions, historical_analysis, simulation_result, strategy_json)
    report_path = output_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"✓ report.txt -> {report_path}")

    # 3. Grafico HTML (Aggiungi encoding="utf-8")
    optimal_name = simulation_result.get("optimal", {}).get("name", "Optimal")
    html = generate_lap_chart_html(simulation_result, optimal_name)
    html_path = output_dir / "strategy_chart.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"✓ strategy_chart.html -> {html_path}")

    # 4. Dati completi JSON (Aggiungi encoding="utf-8")
    full_data = {
        "conditions": conditions,
        "historical_analysis": historical_analysis,
        "simulation_ranking": simulation_result.get("ranking", []),
        "optimal_strategy": simulation_result.get("optimal", {}),
    }
    full_path = output_dir / "full_analysis.json"
    full_path.write_text(json.dumps(full_data, indent=2, default=str), encoding="utf-8")

    return {
        "strategy_json": str(strat_path),
        "report": str(report_path),
        "chart": str(html_path),
        "full_analysis": str(full_path),
    }
