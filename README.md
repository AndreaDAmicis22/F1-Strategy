# 🏎 F1 Strategy System — Scuderia Algoritmo

Sistema intelligente di analisi e strategia per gare di Formula 1, con **modelli di Machine Learning** per la previsione dei tempi giro e la raccomandazione dei compound.

## Architettura

```
src/f1_strategy/
├── main.py                      ← Entry point pipeline (legge race_conditions.json)
├── race_conditions.json         ← Condizioni gara input
├── ml_predictor.py              ← ★ Modelli ML (GBR, Ridge, RandomForest)
├── openf1_client.py             ← Wrapper API OpenF1 (caching, retry, paginazione)
└── agents/
    ├── data_analysis_agent.py   ← Analisi dati storici OpenF1
    ├── race_simulator.py        ← Simulatore scenari giro-per-giro
    ├── strategy_agent.py        ← Produzione strategy.json
    └── report_generator.py      ← Output grafici e report
```

## Avvio rapido

```bash
pip install numpy scikit-learn
cd src/f1_strategy
python main.py                  # pipeline completa con ML
python main.py --no-ml          # solo simulatore fisico
python main.py --no-api         # salta OpenF1, usa knowledge base
```

## Modelli ML (`ml_predictor.py`)

### 1. `LapTimePredictor` — Gradient Boosting Regressor
Predice il tempo giro dato compound, stint_lap, meteo, temperatura.

- **Algoritmo**: `GradientBoostingRegressor` (200 alberi, depth=4, lr=0.08)
- **Feature**: compound_enc, stint_lap, weather_enc, track_temp, air_temp, lap_number, stint_lap²
- **Metriche**: MAE ≈ 0.14s · RMSE ≈ 0.17s · R² ≈ 0.997
- **Training**: 8000 campioni sintetici calibrati su dati reali Monza 2023-2024

### 2. `DegradationModel` — Ridge Regression (polynomial)
Stima il tasso di degrado di ogni compound tramite regressione polinomiale grado 2.

- **Stima stint ottimale per compound**: Soft≈15, Medium≈28, Hard≈44 giri
- **Modella il "cliff"**: accelerazione del degrado oltre il limite ottimale
- **Usato per**: calcolare la finestra ottimale di pit stop

### 3. `CompoundRecommender` — Random Forest Classifier
Classifica il compound migliore per uno stint, dati lunghezza e condizioni meteo.

- **Target**: compound col minor tempo totale stimato per quello stint
- **Input**: n_laps, weather_enc, stint_position, track_temp, sqrt(n_laps)
- **Output**: compound raccomandato + probabilità per ogni opzione
- **Esempio Monza con pioggia**: Stint1→SOFT, Stint2→MEDIUM@SC, Stint3→INTERMEDIATE@pioggia

### 4. `StrategyEvaluator` — Integrazione ML + Simulatore
Combina i tre modelli per:
- Valutare qualsiasi strategia usando tempi ML invece di formule fisse
- Confrontare stima fisica vs stima ML (cross-validazione)
- Raccomandare compound ottimali per ogni fase della gara

## Simulatore Fisico (`agents/race_simulator.py`)

```
lap_time = base_time + compound_delta + degradation(stint_lap) + weather_adj
```

Confronta 10+ strategie candidate:
- 1-stop (M→H, S→H, H→M)
- 2-stop (S→M→H, M→H→M undercut SC)
- Con meteo (M→Inter, S→H→Inter, S→M→Inter, scommessa dry)

## Output

| File | Contenuto |
|------|-----------|
| `strategy.json` | Strategia finale formato competizione |
| `ml_report.json` | Metriche modelli ML, feature importance |
| `report.txt` | Classifica scenari, insight |
| `strategy_chart.html` | Grafici interattivi tempi giro |
| `full_analysis.json` | Dati completi analisi |

## Strategia per Monza (race_conditions.json)

- **Condizioni**: pioggia leggera dal giro 30, Safety Car al giro 15
- **Strategia ottimale**: `MEDIUM (G.1) → INTERMEDIATE (G.30)`
- **Tempo stimato**: 4651s (fisico) / 4655s (ML)
- **Logica**: SC al G.15 non giustifica pit perché Medium è ancora fresco; la pioggia al G.30 rende obbligatorio l'Intermediate

## Dipendenze

```
numpy>=1.26
scikit-learn>=1.4
```
