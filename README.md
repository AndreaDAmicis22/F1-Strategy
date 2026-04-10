# 🏎 F1 Strategy System — Scuderia Algoritmo

Sistema intelligente di analisi e strategia per gare di Formula 1, basato su dati reali OpenF1.

## Architettura

```
f1_strategy/
├── main.py                      ← Entry point pipeline end-to-end
├── race_conditions.json         ← Condizioni gara input
├── openf1_client.py             ← Wrapper API OpenF1 (caching, retry, paginazione)
├── agents/
│   ├── data_analysis_agent.py   ← Agent 1: Analisi dati storici
│   ├── race_simulator.py        ← Agent 2: Simulatore scenari strategici
│   ├── strategy_agent.py        ← Agent 3: Produzione strategia finale
│   └── report_generator.py      ← Agent 4: Output grafici e report
├── cache/                       ← Cache locale dati API (TTL 24h)
└── outputs/
    ├── strategy.json            ← ← OUTPUT PRINCIPALE (formato competizione)
    ├── report.txt               ← Report testuale
    ├── strategy_chart.html      ← Grafici interattivi
    └── full_analysis.json       ← Dati completi analisi
```

## Avvio rapido

```bash
# Pipeline completa (legge race_conditions.json)
python main.py

# Con file condizioni personalizzato
python main.py --conditions /path/to/race_conditions.json

# Senza chiamate API (usa knowledge base locale)
python main.py --no-api
```

## Componenti

### 1. `openf1_client.py` — API Wrapper
- Gestisce chiamate HTTP all'API OpenF1 (https://api.openf1.org)
- **Caching locale**: salva risposte in `cache/` con TTL 24h
- **Retry automatici**: 3 tentativi con backoff esponenziale
- **Paginazione**: gestita automaticamente
- **Funzioni**: `get_sessions()`, `get_laps()`, `get_pit_stops()`, `get_stints()`, `get_weather()`, `get_race_control()`

### 2. `agents/data_analysis_agent.py` — Analisi Storica
- Scarica e analizza sessioni Race di Monza (2023–2024)
- **Analisi stint**: distribuzione compound, lunghezza media per tipo
- **Analisi pit stop**: timing, giri più frequenti, numero stop per driver
- **Analisi tempi giro**: stima degrado gomme, best/worst lap
- **Analisi meteo**: presenza pioggia, temperature
- **Race control**: safety car, VSC, bandiere
- Identifica **pattern vincenti** multi-sessione
- Fallback su knowledge base Monza se API non disponibile

### 3. `agents/race_simulator.py` — Simulatore Giri ⭐ Core algoritmico
Dato un set di parametri, simula giro per giro il tempo gara:

**Input:**
```json
{
  "strategy": [{"stint": 1, "compound": "medium", "start_lap": 1}, ...],
  "conditions": { "total_laps": 53, "weather": {...}, "safety_car": {...} }
}
```

**Modello di simulazione:**
```
lap_time = base_time + compound_delta + degradation(stint_lap) + weather_adj
```

- **Degrado gomme**: lineare fino a `max_optimal_laps`, poi accelerato 2.5x
- **Impatto meteo**: ogni compound ha `wet_performance` → intermediate beneficia della pioggia
- **Safety Car**: giri SC a velocità ridotta (~108s invece di 83s)
- **Pit stop**: aggiunge `pit_lane_time_loss_seconds` al tempo totale

**Genera e confronta automaticamente 10+ scenari:**
- 1-stop (M→H, S→H, H→M)
- 2-stop (S→M→H, M→H→M)
- Strategie meteo (M→Inter, S→H→Inter, S→M→Inter)
- Scommessa dry

### 4. `agents/strategy_agent.py` — Strategia Finale
- Sintetizza analisi storica + simulazione
- Produce `strategy.json` nel formato competizione
- Valida il formato prima dell'output
- Genera rationale dettagliato

### 5. `agents/report_generator.py` — Output & Visualizzazioni
- `strategy.json` (formato competizione)
- `report.txt` con classifica scenari
- `strategy_chart.html` con grafici interattivi (Chart.js)
- `full_analysis.json` con tutti i dati

## Formato Output strategy.json

```json
{
  "team_name": "Scuderia Algoritmo",
  "strategy": [
    {"stint": 1, "compound": "medium", "start_lap": 1},
    {"stint": 2, "compound": "intermediate", "start_lap": 30}
  ],
  "rationale": "...",
  "estimated_total_time_seconds": 4651.05
}
```

## Logica Strategica — Monza con Meteo Variabile

Dato il `race_conditions.json`:
- **Giro 1–14**: Asciutto → Medium ottimale (bassa usura, stint lungo)
- **Giro 15**: Safety Car → opportunità pit stop a costo zero
- **Giro 15–29**: Asciutto con Hard → stint lungo, risparmia pit stop
- **Giro 30**: Pioggia leggera → pit obbligatorio per Intermediate

**Strategia ottimale identificata**: `MEDIUM (G.1) → INTERMEDIATE (G.30)`
- Tempo stimato: **4651s** (1h17m31s)
- Pit stop: **1** (al giro 30, sincronizzato con inizio pioggia)

**Perché non la strategia S→H senza inter?**
Il simulatore mostra che guidare 23 giri su gomme slick con pioggia leggera costa
~4-5s/giro in più rispetto alle intermediate → perdita ~100s totale, molto più del pit stop aggiuntivo.

## Dipendenze

Nessuna dipendenza esterna richiesta — usa solo librerie Python standard:
- `urllib` per HTTP
- `json`, `pathlib`, `statistics`, `hashlib`, `logging`