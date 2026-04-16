"""
train_models.py
===============
Addestra 3 modelli ML su dati reali OpenF1:

  1. LapTimePredictor      — GBR: stima il tempo giro (compound, stint_lap, meteo, temp)
  2. DegradationRiskModel  — Ridge poly per compound: stima degrado e cliff
  3. SafetyCarImpactModel  — GBR: stima secondi guadagnati/persi in base alla
                             sincronizzazione pit stop ↔ Safety Car

Uso:
    python src/f1_strategy/train_models.py --data data/all_circuits_laps.csv
    python src/f1_strategy/train_models.py --data data/all_circuits_laps.csv --test-circuit Monza
"""

import argparse
import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

COMPOUND_ENC = {"soft": 0, "medium": 1, "hard": 2, "intermediate": 3, "wet": 4}
LAP_TIME_MIN = 60.0
LAP_TIME_MAX = 200.0


# ── Carica e pulisce ───────────────────────────────────────────────────────────
def load_and_clean(csv_path: Path, erase_stop_sc: bool = True) -> pd.DataFrame:
    logger.info(f"Caricamento dati: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    logger.info(f"  Righe raw: {len(df)}")

    df = df[df["lap_duration"].between(LAP_TIME_MIN, LAP_TIME_MAX)]
    df = df[df["compound"].isin(COMPOUND_ENC.keys())]
    if erase_stop_sc:
        df = df[df["is_pit_out_lap"] != True]  # noqa: E712
        df = df[df["is_sc_lap"] != True]  # noqa: E712

    df["compound_enc"] = df["compound"].map(COMPOUND_ENC)
    df["rainfall"] = df["rainfall"].fillna(0.0)
    df["weather_enc"] = df["rainfall"].apply(lambda r: 2 if r > 2.0 else (1 if r > 0.1 else 0))

    for col in ["air_temp", "track_temp"]:
        df[col] = df.groupby("session_key")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    df["stint_lap_sq"] = df["stint_lap"] ** 2
    df["lap_duration_norm"] = df.groupby("session_key")["lap_duration"].transform(lambda x: x - x.quantile(0.05))

    df["speed_mean"] = df[["i1_speed", "i2_speed", "st_speed"]].mean(axis=1)
    df["speed_range"] = df["st_speed"] - df[["i1_speed", "i2_speed"]].min(axis=1)

    for sec in ["duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        df[f"{sec}_norm"] = df.groupby("session_key")[sec].transform(lambda x: x - x.quantile(0.05))

    for col in ["speed_mean", "speed_range", "duration_sector_1_norm", "duration_sector_2_norm", "duration_sector_3_norm"]:
        df[col] = df.groupby(["session_key", "compound"])[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    df["humidity"] = df["humidity"].fillna(df["humidity"].median())
    df["wind_speed"] = df["wind_speed"].fillna(0.0)

    df["avg_energy_session"] = (
        df.groupby("session_key")
        .apply(lambda x: (x["speed_mean"] * x["track_temp"]).mean())
        .reindex(df["session_key"])
        .values
    )

    # 1. Feature di Progressione e Efficienza
    df["session_progression"] = df.groupby("session_key")["lap_number"].transform(lambda x: x / x.max())
    df["max_speed_session"] = df.groupby("session_key")["st_speed"].transform("max")
    df["speed_efficiency"] = (df["st_speed"] / df["max_speed_session"]).fillna(1.0)  # 1.0 è il default neutro
    df["speed_efficiency"] = df["speed_efficiency"].replace([np.inf, -np.inf], 1.0)

    # 2. LAG FEATURES (Corretto per Driver)
    # Ordiniamo prima per assicurarci che lo shift sia temporale
    df = df.sort_values(["session_key", "driver_number", "lap_number"])

    # Usiamo driver_number solo per il calcolo, non serve che sia in LAP_FEATURES
    df["prev_lap_duration"] = df.groupby(["session_key", "driver_number"])["lap_duration"].shift(1)

    # Riempimento dei primi giri dello stint (dove shift crea NaN)
    df["prev_lap_duration"] = df.groupby(["session_key", "driver_number"])["prev_lap_duration"].transform(
        lambda x: x.fillna(x.median())
    )
    # Fallback globale se un pilota ha un solo giro
    df["prev_lap_duration"] = df["prev_lap_duration"].fillna(df["lap_duration"].median())

    logger.info(f"  Righe dopo pulizia: {len(df)}")
    logger.info(f"  Sessioni: {df['session_key'].nunique()}")
    logger.info(f"  Circuiti: {df['location'].nunique() if 'location' in df.columns else 'N/A'}")
    logger.info(f"  Compound:\n{df['compound'].value_counts().to_string()}")

    return df


def _temporal_split(
    df: pd.DataFrame, test_circuit: str | None = None, test_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_circuit:
        for col in ["location", "circuit_short_name"]:
            if col in df.columns:
                mask = df[col].str.lower().str.contains(test_circuit.lower(), na=False)
                if mask.any():
                    logger.info(f"  Split per circuito '{test_circuit}': train={len(df[~mask])}, test={len(df[mask])}")
                    return df[~mask], df[mask]
        logger.warning(f"  Circuito '{test_circuit}' non trovato, uso split temporale")

    sessions = sorted(df["session_key"].unique())
    n_test = max(1, int(len(sessions) * test_ratio))
    test_sess = sessions[-n_test:]
    train_sess = sessions[:-n_test]
    train_df = df[df["session_key"].isin(train_sess)]
    test_df = df[df["session_key"].isin(test_sess)]
    logger.info(f"  Split temporale: train={len(train_sess)} sess, test={len(test_sess)} sess")
    return train_df, test_df


# ── Modello 1: LapTimePredictor ────────────────────────────────────────────────
LAP_FEATURES = [
    "compound_enc",
    "stint_lap",
    "stint_lap_sq",
    "weather_enc",
    "air_temp",
    "track_temp",
    "lap_number",
    "duration_sector_1_norm",
    "duration_sector_2_norm",
    "duration_sector_3_norm",
    "speed_mean",
    "speed_range",
    "humidity",
    "wind_speed",
    "session_progression",
    "speed_efficiency",
    "prev_lap_duration",
    "avg_energy_session",
]


def train_lap_time_model(df: pd.DataFrame, test_circuit: str | None = None) -> dict:
    logger.info("\n=== Training LapTimePredictor (GradientBoosting) ===")

    # Rimuovi feature con troppi NaN
    active_features = []
    for f in LAP_FEATURES:
        if f in df.columns and df[f].isna().mean() <= 0.3:
            active_features.append(f)
        else:
            logger.warning(f"  Feature '{f}' esclusa (>30% NaN o mancante)")

    train_df, test_df = _temporal_split(df, test_circuit)
    logger.info(f"  Train: {len(train_df)} giri  |  Test: {len(test_df)} giri")

    nan_counts = train_df[active_features].isna().sum()
    if nan_counts.any():
        logger.error(f"NaN trovati nelle feature prima del fit:\n{nan_counts[nan_counts > 0]}")

    X_train = train_df[active_features].values.astype(np.float32)
    y_train = train_df["lap_duration"].values.astype(np.float32)
    X_test = test_df[active_features].values.astype(np.float32)
    y_test = test_df["lap_duration"].values.astype(np.float32)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=600,
                    max_depth=7,
                    learning_rate=0.04,
                    subsample=0.8,
                    min_samples_leaf=10,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = model.score(X_test, y_test)
    cv = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()

    logger.info(f"  MAE={mae:.4f}s  RMSE={rmse:.4f}s  R²={r2:.4f}  CV_MAE={cv:.4f}s")

    gbr = model.named_steps["gbr"]
    importance = dict(zip(active_features, gbr.feature_importances_.tolist(), strict=False))
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]:
        logger.info(f"    {feat:<30} {imp:.3f}  {'█' * int(imp * 40)}")

    path = MODEL_DIR / "lap_time_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": active_features}, f)
    logger.info(f"  Salvato: {path}")

    return {
        "type": "GradientBoostingRegressor",
        "features": active_features,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "CV_MAE": round(cv, 4)},
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
    }


# ── Modello 2: DegradationRiskModel ───────────────────────────────────────────
# Stima il degrado giro per giro per compound.
# Permette di calcolare il "rischio degrado" di una strategia:
# quanti secondi perde uno stint che va oltre il cliff del compound.

DEGR_FEATURES = [
    "stint_lap",
    "stint_lap_sq",
    "lap_number",
    "session_progression",
    "compound_enc",
    "weather_enc",
    "track_temp",
    "air_temp",
    "humidity",
    "wind_speed",
    "avg_energy_session",
    "speed_mean",
    "speed_range",
    "speed_efficiency",
    "prev_lap_duration",
    "duration_sector_1_norm",
    "duration_sector_2_norm",
    "duration_sector_3_norm",
]


def train_degradation_model(df: pd.DataFrame, test_circuit: str | None = None) -> dict:
    logger.info("\n=== Training DegradationRiskModel (Incremental Step GBR) ===")

    train_df, test_df = _temporal_split(df, test_circuit)
    results = {}
    models = {}

    for compound in COMPOUND_ENC:
        # 1. Preparazione Training Set
        sub_train = train_df[train_df["compound"] == compound].copy()
        if len(sub_train) < 50:
            continue

        # ORDINE CRUCIALE: Ordiniamo per sessione, pilota e giro per calcolare il delta
        sub_train = sub_train.sort_values(["session_key", "driver_number", "lap_number"])

        # --- LOGICA INCREMENTALE ---
        # Calcoliamo la differenza di tempo tra questo giro e il precedente (nello stesso stint)
        # Se lo stint_lap è 1, il delta è 0 (punto di partenza)
        sub_train["degr_step"] = sub_train.groupby(["session_key", "driver_number"])["lap_duration_norm"].diff().fillna(0)
        sub_train.loc[sub_train["stint_lap"] == 1, "degr_step"] = 0

        # Pulizia: un incremento sensato sta tra -0.1 (track evolution) e 0.6 (usura pesante)
        # Escludiamo errori, traffico o pit stop che sporcano il delta istantaneo
        sub_train = sub_train[sub_train["degr_step"].between(-0.2, 0.8)]

        # 2. Selezione feature
        avail_features = [f for f in DEGR_FEATURES if f in sub_train.columns]
        X_train = sub_train[avail_features].values.astype(np.float32)
        y_train = sub_train["degr_step"].values.astype(np.float32)

        # 3. Preparazione Test Set
        sub_test = test_df[test_df["compound"] == compound].copy()
        X_test, y_test = None, None
        if len(sub_test) >= 10:
            sub_test = sub_test.sort_values(["session_key", "driver_number", "lap_number"])
            sub_test["degr_step"] = sub_test.groupby(["session_key", "driver_number"])["lap_duration_norm"].diff().fillna(0)
            sub_test.loc[sub_test["stint_lap"] == 1, "degr_step"] = 0

            X_test = sub_test[avail_features].values.astype(np.float32)
            y_test = sub_test["degr_step"].values.astype(np.float32)

        # 4. Training GBR
        # Usiamo Huber loss per essere robusti ai giri con traffico
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, loss="huber", random_state=42)
        model.fit(X_train, y_train)

        # 5. Validazione
        test_mae = "N/A"
        if X_test is not None:
            test_mae = round(float(mean_absolute_error(y_test, model.predict(X_test))), 4)

        # 6. Simulazione Cliff (per metadati)
        thresholds = {
            "soft": 1.0,
            "medium": 1.7,
            "hard": 2.25,
            "intermediate": 3.8,  # Alzata da 1.3: permetterà circa 20-25 giri prima del cliff
            "wet": 5.0,
        }
        current_threshold = thresholds.get(compound.lower(), 1.5)
        meds = {f: float(sub_train[f].median()) for f in avail_features}
        meds["session_progression"] = 0.5
        meds["lap_number"] = 27

        def get_step_pred(lap):
            row = []
            for f in avail_features:
                if f == "stint_lap":
                    row.append(float(lap))
                elif f == "stint_lap_sq":
                    row.append(float(lap**2))
                else:
                    row.append(meds.get(f, 0.0))
            return float(model.predict(np.array([row], dtype=np.float32))[0])

        accumulated_degr = 0.0
        cliff_lap = 54  # Fallback se non raggiunge mai la soglia

        for lap in range(1, 61):
            step = get_step_pred(lap)
            # Solo incrementi positivi: la gomma non rigenera mai performance
            accumulated_degr += max(0.0, step)

            # Verifichiamo se l'usura accumulata ha sfondato la soglia specifica del compound
            if accumulated_degr > current_threshold:
                cliff_lap = lap
                break

        # Salvataggio nel dizionario dei modelli e risultati
        models[compound] = model
        results[compound] = {
            "features": avail_features,
            "cliff_lap": cliff_lap,
            "threshold_used": current_threshold,
            "test_mae": test_mae,
            "n_train": len(sub_train),
        }

        logger.info(f"  {compound:<12} | Cliff: {cliff_lap:>2} (Soglia: {current_threshold}s) | MAE: {test_mae}")

    # Salvataggio su disco del file pkl
    path = MODEL_DIR / "degradation_model.pkl"
    with open(path, "wb") as f:
        # Salviamo la struttura completa per il validatore
        pickle.dump({"models": models, "metadata": results}, f)

    logger.info(f"=== Modello Degradazione salvato in {path} ===")
    return {"type": "GBR Incremental Step", "compounds": results}


# ── Modello 3: SafetyCarImpactModel ───────────────────────────────────────────
# Stima quanti secondi guadagna/perde un pit stop in relazione alla Safety Car.
# Features: distanza pit stop dalla SC, durata SC, compound montato, giro assoluto.
# Target: delta_time = lap_time_sc_lap - mediana_lap_time_sessione
#         (negativo = il pilota ha guadagnato rispetto al passo normale)
#
# Logica: durante una SC tutti rallentano → chi fa il pit "gratis" guadagna
# rispetto a chi paga il pit in condizioni normali (22+ secondi di delta).
# Il modello impara questa relazione dai dati reali.


def _build_sc_dataset(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Costruisce il dataset per il SafetyCarImpactModel.
    Per ogni sessione che ha avuto una SC, calcola:
      - distanza tra ogni pit stop e il giro SC più vicino
      - composto montato dopo il pit
      - vantaggio/svantaggio in secondi rispetto al pit in condizioni normali
    """
    # Abbiamo bisogno dei dati raw con SC laps — li ricaviamo da is_sc_lap
    # ma nel CSV abbiamo già il flag. Usiamo pit_out_lap come proxy del pit stop.

    normal_speed = df[~df["is_sc_lap"]].groupby("session_key")["speed_mean"].median().rename("speed_ref")
    session_median = df.groupby("session_key")["lap_duration"].median().rename("median_lap")

    sc_first = df[df["is_sc_lap"]].groupby("session_key")["lap_number"].min().rename("sc_lap_number")

    # Calcolo progressivo della durata SC (quanti giri di SC sono passati finora)
    df = df.sort_values(["session_key", "lap_number"])
    df["sc_duration_step"] = df.groupby("session_key")["is_sc_lap"].cumsum()

    pit_laps = df[df["is_pit_out_lap"] == True].copy()  # noqa: E712
    if len(pit_laps) == 0:
        return None

    pit_laps = pit_laps.merge(normal_speed, on="session_key", how="left")
    pit_laps = pit_laps.merge(session_median, on="session_key", how="left")  # Mediana del tempo giro per sessione
    pit_laps = pit_laps.merge(
        sc_first, on="session_key", how="left"
    )  # Identifica il giro SC per sessione: primo giro con is_sc_lap=True

    # Rapporto velocità: quanto si va piano rispetto al normale?
    pit_laps["sc_speed_ratio"] = (pit_laps["speed_mean"] / pit_laps["speed_ref"]).fillna(1.0)

    # Calcolo distanza (gestendo i NaN per le sessioni senza SC)
    # Distanza pit ↔ SC (negativa = pit PRIMA della SC, positiva = pit DOPO)
    pit_laps["dist_to_sc"] = pit_laps["lap_number"] - pit_laps["sc_lap_number"]
    pit_laps["dist_to_sc"] = pit_laps["dist_to_sc"].fillna(999)
    pit_laps["has_sc"] = pit_laps["sc_lap_number"].notna().astype(int)
    pit_laps["sc_duration_step"] = pit_laps["sc_duration_step"].fillna(0)

    # Target: delta rispetto al passo normale
    # Un pit durante SC costa ~0s di delta (il tempo perso nel pit lane è recuperato
    # perché tutti rallentano); un pit in condizioni normali costa ~22s di delta.
    # Usiamo la lap_duration_norm (già calcolata) del giro di pit-out come proxy del costo.
    pit_laps["pit_cost_delta"] = pit_laps["lap_duration_norm"]  # quanto più lento del best pace

    # Features
    records = []
    for _, row in pit_laps.iterrows():
        records.append(
            {
                "dist_to_sc": float(row["dist_to_sc"]),
                "has_sc": int(row["has_sc"]),
                "sc_speed_ratio": float(row["sc_speed_ratio"]),
                "sc_duration_step": float(row["sc_duration_step"]),
                "compound_enc": COMPOUND_ENC.get(str(row["compound"]).lower(), 1),
                "lap_number": int(row["lap_number"]),
                "weather_enc": int(row["weather_enc"]),
                "track_temp": float(row["track_temp"]),
                "rainfall": float(row.get("rainfall", 0.0)),
                "pit_cost_delta": float(row["pit_cost_delta"]),
            }
        )

    if not records:
        return None

    return pd.DataFrame(records).dropna()


SC_FEATURES = [
    "dist_to_sc",
    "has_sc",
    "sc_speed_ratio",  # Intensità neutralizzazione (0.4 = Full SC, 0.7 = VSC)
    "sc_duration_step",  # Da quanto tempo la SC è fuori
    "compound_enc",
    "lap_number",
    "weather_enc",
    "track_temp",
    "rainfall",  # Importante per i tempi di reazione del pit
]


def train_sc_impact_model(df: pd.DataFrame, test_circuit: str | None = None) -> dict:
    logger.info("\n=== Training SafetyCarImpactModel (GradientBoosting) ===")

    train_df, test_df = _temporal_split(df, test_circuit)

    sc_train = _build_sc_dataset(train_df)
    sc_test = _build_sc_dataset(test_df)

    if sc_train is None or len(sc_train) < 30:
        logger.warning(f"  Dati SC insufficienti ({len(sc_train) if sc_train is not None else 0} record), skip")
        path = MODEL_DIR / "sc_impact_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": None, "features": SC_FEATURES, "fallback": True}, f)
        return {"type": "SafetyCarImpactModel", "status": "skipped_insufficient_data"}

    X_train = sc_train[SC_FEATURES].values.astype(np.float32)
    y_train = sc_train["pit_cost_delta"].values.astype(np.float32)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=7,
                    learning_rate=0.04,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    test_mae = None
    if sc_test is not None and len(sc_test) > 0:
        X_test = sc_test[SC_FEATURES].values.astype(np.float32)
        y_test = sc_test["pit_cost_delta"].values.astype(np.float32)
        test_mae = round(float(mean_absolute_error(y_test, model.predict(X_test))), 4)

    logger.info(f"  n_train={len(sc_train)}  test_MAE={test_mae}s")

    gbr = model.named_steps["gbr"]
    importance = dict(zip(SC_FEATURES, gbr.feature_importances_.tolist(), strict=False))
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {feat:<20} {imp:.3f}  {'█' * int(imp * 30)}")

    path = MODEL_DIR / "sc_impact_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": SC_FEATURES, "fallback": False}, f)
    logger.info(f"  Salvato: {path}")

    return {
        "type": "GradientBoostingRegressor (pit cost vs SC timing)",
        "n_train": len(sc_train),
        "test_mae": test_mae,
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
    }


# ── Training report ────────────────────────────────────────────────────────────
def save_training_report(reports: dict, df: pd.DataFrame):
    report = {
        "dataset": {
            "total_laps": len(df),
            "sessions": int(df["session_key"].nunique()),
            "circuits": df["location"].nunique() if "location" in df.columns else None,
            "compound_distribution": df["compound"].value_counts().to_dict(),
            "lap_time_stats": {
                "mean": round(df["lap_duration"].mean(), 3),
                "std": round(df["lap_duration"].std(), 3),
                "min": round(df["lap_duration"].min(), 3),
                "max": round(df["lap_duration"].max(), 3),
            },
        },
        "models": reports,
    }
    path = MODEL_DIR / "training_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"\nReport salvato: {path}")
    return report


# ── Entry point ────────────────────────────────────────────────────────────────
def run(csv_path: Path | None = None, test_circuit: str | None = None):
    if csv_path is None:
        csv_path = DATA_DIR / "all_circuits_laps.csv"

    if not csv_path.exists():
        logger.error(f"Dataset non trovato: {csv_path}\nEsegui prima: python collect_training_data.py")
        return None

    df = load_and_clean(csv_path)
    df_raw = load_and_clean(csv_path, erase_stop_sc=False)
    if len(df) < 100:
        logger.error("Dataset troppo piccolo.")
        return None

    reports = {
        # "lap_time_model": train_lap_time_model(df, test_circuit),
        "degradation_model": train_degradation_model(df, test_circuit),
        "sc_impact_model": train_sc_impact_model(df_raw, test_circuit),
    }
    save_training_report(reports, df)
    logger.info("\n=== Training completato. Modelli in: models/ ===")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--test-circuit", type=str, default=None)
    args = parser.parse_args()
    if args.test_circuit is None:
        logger.info("NO TEST CIRCUIT: uso split temporale invece di un circuito dedicato")
    run(csv_path=Path(args.data) if args.data else None, test_circuit=args.test_circuit)
