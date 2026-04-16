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
# Training
poetry run python src/f1_strategy/train_models.py --data data/all_circuits_laps.csv

# Pipeline
poetry run python src/f1_strategy/main.py --strategies-dir inputs/teams/
```

### Configurare le condizioni di gara

Modifica `race_conditions.json` per specificare le condizioni della sessione da analizzare:

```json
{ 
  "circuit": "Monza", 
  "total_laps": 53, 
  "weather": { 
    "rain_start_lap": 999, 
    "rain_intensity": "none" 
  }, 
  "safety_car": { 
    "active": true, 
    "lap": 16, 
    "duration_laps": 4
  },
   "pit_lane_avg_time_loss_seconds": 22.5, 
   "pit_lane_std_time_loss_seconds ": 0.4
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

### 2. Simulatore Giri ⭐ Core Algoritmico

Simula giro per giro il tempo gara dato un set di parametri strategici.

**Input:**
```json
{
  "team_name": "Aggressive-Sprint",
  "strategy": [
    { "stint": 1, "compound": "soft", "start_lap": 1 },
    { "stint": 2, "compound": "medium", "start_lap": 18 },
    { "stint": 3, "compound": "soft", "start_lap": 38 }
  ],
  "rationale": "Strategia a 2 soste per sfruttare la velocità della gomma Soft a fine gara con auto leggera (session_progression > 0.7). Punta a compensare il costo extra del pit con giri record."
}
```

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
