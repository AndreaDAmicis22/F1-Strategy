# 🏎️ F1 Strategy System — Scuderia Algoritmo

> Sistema intelligente di analisi e strategia per gare di Formula 1, basato su dati reali [OpenF1](https://api.openf1.org).

---

## 📋 Indice

- [Panoramica](#-panoramica)
- [Architettura](#-architettura)
- [Installazione](#-installazione)
- [Avvio Rapido](#-avvio-rapido)
- [Componenti](#-componenti)
- [Formato Output](#-formato-output-strategyjson)
- [Logica Strategica](#-logica-strategica--monza-con-meteo-variabile)
- [Dipendenze](#-dipendenze)

---

## 🔍 Panoramica

**F1 Strategy System** è una pipeline multi-agente che scarica dati storici di gara dall'API OpenF1, li analizza statisticamente e produce una strategia pit-stop ottimale per le condizioni di gara fornite.

Il sistema è progettato per la competizione **Scuderia Algoritmo** e supporta:

- Analisi di sessioni storiche reali (stint, pit stop, meteo, safety car)
- Simulazione giro per giro di 10+ scenari strategici
- Output in formato competizione (`strategy.json`) + report + grafici interattivi

---

## ⚙️ Installazione

### Prerequisiti

- Python `>=3.11, <3.13`
- [Poetry](https://python-poetry.org/) `>=2.0.0`

### Setup

```bash
# Clona la repository
git clone https://github.com/AndreaDAmicis22/F1-Strategy.git
cd F1-Strategy

# Installa le dipendenze con Poetry
poetry install

# Attiva l'ambiente virtuale
poetry shell
```

---

## 🚀 Avvio Rapido

```bash
# Pipeline completa (legge race_conditions.json)
poetry run python src/f1_strategy/train_models.py --data data/all_circuits_laps.csv

poetry run python src/f1_strategy/main.py --strategies-dir inputs/teams/
```

### Configurare le condizioni di gara

Modifica `race_conditions.json` per specificare le condizioni della sessione da analizzare:

```json
{
  "circuit": "Monza",
  "total_laps": 53,
  "weather": {
    "rain_start_lap": 30,
    "rain_intensity": "light"
  },
  "safety_car": {
    "lap": 15,
    "duration_laps": 3
  }
}
```

---

## 🧩 Componenti

### 1. `openf1_client.py` — API Wrapper

Gestisce tutte le comunicazioni con l'[API OpenF1](https://api.openf1.org):

- **Caching locale**: risposte salvate in `cache/` con TTL 24h (evita richieste ridondanti)
- **Retry automatici**: 3 tentativi con backoff esponenziale
- **Paginazione automatica**
- **Funzioni esposte**: `get_sessions()`, `get_laps()`, `get_pit_stops()`, `get_stints()`, `get_weather()`, `get_race_control()`

---

### 2. `agents/data_analysis_agent.py` — Analisi Storica

Scarica e analizza le sessioni Race di Monza (2023–2024):

| Analisi | Dettaglio |
|---|---|
| **Stint** | Distribuzione compound, lunghezza media per tipo |
| **Pit stop** | Timing, giri più frequenti, numero stop per driver |
| **Tempi giro** | Stima degrado gomme, best/worst lap |
| **Meteo** | Presenza pioggia, temperature |
| **Race control** | Safety car, VSC, bandiere |

> **Fallback**: se l'API non è disponibile, usa una knowledge base locale di Monza.

Identifica automaticamente i **pattern vincenti** multi-sessione.

---

### 3. `agents/race_simulator.py` — Simulatore Giri ⭐ Core Algoritmico

Simula giro per giro il tempo gara dato un set di parametri strategici.

**Input:**
```json
{
  "strategy": [
    {"stint": 1, "compound": "medium", "start_lap": 1},
    {"stint": 2, "compound": "intermediate", "start_lap": 30}
  ],
  "conditions": {
    "total_laps": 53,
    "weather": {"rain_start_lap": 30},
    "safety_car": {"lap": 15}
  }
}
```

**Modello di simulazione:**
```
lap_time = base_time + compound_delta + degradation(stint_lap) + weather_adj
```

| Fattore | Comportamento |
|---|---|
| **Degrado gomme** | Lineare fino a `max_optimal_laps`, poi accelerato 2.5× |
| **Meteo** | Ogni compound ha `wet_performance`; intermediate beneficia della pioggia |
| **Safety Car** | Giri SC a ~108s invece di ~83s |
| **Pit stop** | Aggiunge `pit_lane_time_loss_seconds` al totale |

**Scenari confrontati automaticamente (10+):**

- 1-stop: `M→H`, `S→H`, `H→M`
- 2-stop: `S→M→H`, `M→H→M`
- Meteo: `M→Inter`, `S→H→Inter`, `S→M→Inter`
- Scommessa dry

---

### 4. `agents/strategy_agent.py` — Strategia Finale

- Sintetizza analisi storica + risultati simulazione
- Produce `strategy.json` nel formato competizione con validazione automatica
- Genera rationale dettagliato con motivazioni della scelta

---

### 5. `agents/report_generator.py` — Output & Visualizzazioni

Genera quattro artefatti di output:

| File | Contenuto |
|---|---|
| `strategy.json` | Strategia ottimale in formato competizione |
| `report.txt` | Classifica di tutti gli scenari simulati |
| `strategy_chart.html` | Grafici interattivi (Chart.js) |
| `full_analysis.json` | Dump completo di tutti i dati analizzati |

---

## 📄 Formato Output `strategy.json`

```json
{
  "team_name": "Scuderia Algoritmo",
  "strategy": [
    {"stint": 1, "compound": "medium", "start_lap": 1},
    {"stint": 2, "compound": "intermediate", "start_lap": 30}
  ],
  "rationale": "Strategia 1-stop ottimale: stint lungo su Medium sfrutta la Safety Car al giro 15 per evitare pit a costo pieno, poi passaggio a Intermediate sincronizzato con l'arrivo della pioggia al giro 30.",
  "estimated_total_time_seconds": 4651.05
}
```

---

## 🧠 Logica Strategica — Monza con Meteo Variabile

Per le condizioni di default (`race_conditions.json`):

| Fase | Giri | Condizione | Scelta |
|---|---|---|---|
| Stint 1 | 1–14 | Asciutto | **Medium** (bassa usura, stint lungo) |
| Safety Car | 15 | Neutro | Pit stop a costo zero |
| Stint 2 | 15–29 | Asciutto | **Hard** (risparmia pit stop) |
| Pit | 30 | Inizio pioggia | Pit obbligatorio per Intermediate |
| Stint 3 | 30–53 | Pioggia leggera | **Intermediate** |

**Strategia ottimale identificata: `MEDIUM (G.1) → INTERMEDIATE (G.30)`**

- ⏱️ Tempo stimato: **4651s** (1h 17m 31s)
- 🔧 Pit stop: **1** (al giro 30, sincronizzato con inizio pioggia)

> **Perché non `S→H` senza intermediate?**
> Guidare 23 giri su slick con pioggia leggera costa ~4–5s/giro extra rispetto alle Intermediate → perdita di ~100s totali, molto superiore al costo di un pit stop aggiuntivo.

---

## 📦 Dipendenze

| Libreria | Versione | Uso |
|---|---|---|
| `numpy` | >=2.4.4 | Calcoli numerici (degrado, tempi) |
| `pandas` | >=3.0.2 | Analisi dati storici |
| `scikit-learn` | >=1.8.0 | Modelli statistici (degrado gomme) |

Tutte le comunicazioni HTTP usano `urllib` (libreria standard Python), senza dipendenze esterne aggiuntive per il networking.

### Build System

```toml
[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

---

## 👤 Autore

**Andrea D'Amicis** — [@AndreaDAmicis22](https://github.com/AndreaDAmicis22)

---

*Built with 🏁 Python & OpenF1 API*
