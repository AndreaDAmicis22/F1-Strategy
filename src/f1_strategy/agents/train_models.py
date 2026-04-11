"""
train_models.py
===============
Addestra i modelli ML sui dati REALI scaricati da OpenF1 (via collect_training_data.py).

Uso:
    # Prima scarica i dati reali:
    python collect_training_data.py --years 2023 2024

    # Poi addestra i modelli:
    python train_models.py

    # Solo su Monza:
    python collect_training_data.py --circuit Monza
    python train_models.py --data ../../data/laps_raw.csv

Output:
    models/lap_time_model.pkl        — GradientBoostingRegressor tempi giro
    models/degradation_model.pkl     — Ridge polinomiale degrado per compound
    models/compound_recommender.pkl  — RandomForest classificatore compound
    models/training_report.json      — metriche, feature importance, distribuzione dati
"""

import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
MODEL_DIR  = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

COMPOUND_ENC = {"soft": 0, "medium": 1, "hard": 2, "intermediate": 3, "wet": 4}

# Filtra giri anomali (SC, pit out, outlier)
LAP_TIME_MIN = 60.0   # secondi — nessun circuito F1 sotto 60s
LAP_TIME_MAX = 200.0  # secondi — oltre questo = SC o problema


# ── Carica e pulisce il dataset ────────────────────────────────────────────────

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Caricamento dati: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Righe raw: {len(df)}")

    # Rimuovi giri invalidi
    df = df[df["lap_duration"].between(LAP_TIME_MIN, LAP_TIME_MAX)]
    df = df[df["compound"].isin(COMPOUND_ENC.keys())]
    df = df[df["is_pit_out_lap"] != True]    # noqa: E712
    df = df[df["is_sc_lap"] != True]          # giri SC falsano il tempo

    # Encode compound
    df["compound_enc"] = df["compound"].map(COMPOUND_ENC)

    # Meteo: pioggia → categoria
    df["rainfall"] = df["rainfall"].fillna(0.0)
    df["weather_enc"] = df["rainfall"].apply(
        lambda r: 2 if r > 2.0 else (1 if r > 0.1 else 0)
    )

    # Imputa temperature mancanti con mediana per circuito/sessione
    df["air_temp"]   = df.groupby("session_key")["air_temp"].transform(lambda x: x.fillna(x.median()))
    df["track_temp"] = df.groupby("session_key")["track_temp"].transform(lambda x: x.fillna(x.median()))
    df["air_temp"]   = df["air_temp"].fillna(25.0)
    df["track_temp"] = df["track_temp"].fillna(38.0)

    # stint_lap al quadrato per degrado non lineare
    df["stint_lap_sq"] = df["stint_lap"] ** 2

    logger.info(f"  Righe dopo pulizia: {len(df)}")
    logger.info(f"  Circuiti: {df['location'].nunique() if 'location' in df.columns else 'N/A'}")
    logger.info(f"  Compound distribution:\n{df['compound'].value_counts().to_string()}")

    return df


# ── Feature engineering ────────────────────────────────────────────────────────

LAP_FEATURES = [
    "compound_enc",
    "stint_lap",
    "stint_lap_sq",
    "weather_enc",
    "air_temp",
    "track_temp",
    "lap_number",
]

def build_lap_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[LAP_FEATURES].values.astype(np.float32)
    y = df["lap_duration"].values.astype(np.float32)
    return X, y


# ── Modello 1: LapTimePredictor ────────────────────────────────────────────────

def train_lap_time_model(df: pd.DataFrame) -> dict:
    logger.info("\n=== Training LapTimePredictor (GradientBoosting) ===")

    X, y = build_lap_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # Valutazione
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = math.sqrt(mean_squared_error(y_test, preds))
    r2    = model.score(X_test, y_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    logger.info(f"  MAE  test: {mae:.4f}s")
    logger.info(f"  RMSE test: {rmse:.4f}s")
    logger.info(f"  R²   test: {r2:.4f}")
    logger.info(f"  CV MAE (5-fold): {cv_mae:.4f}s")

    # Feature importance
    gbr = model.named_steps["gbr"]
    importance = dict(zip(LAP_FEATURES, gbr.feature_importances_.tolist()))
    logger.info("  Feature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        logger.info(f"    {feat:<16} {imp:.3f}  {bar}")

    # Salva
    path = MODEL_DIR / "lap_time_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": LAP_FEATURES}, f)
    logger.info(f"  Salvato: {path}")

    return {
        "type": "GradientBoostingRegressor",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "CV_MAE": round(cv_mae, 4)},
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
    }


# ── Modello 2: DegradationModel (per compound) ────────────────────────────────

def train_degradation_model(df: pd.DataFrame) -> dict:
    logger.info("\n=== Training DegradationModel (Ridge polynomial) ===")

    results = {}
    models = {}

    for compound in COMPOUND_ENC:
        sub = df[df["compound"] == compound].copy()
        if len(sub) < 30:
            logger.warning(f"  {compound}: troppo pochi dati ({len(sub)} giri), skip")
            continue

        # Solo giri dry (senza pioggia) per misurare degrado puro
        sub_dry = sub[sub["weather_enc"] == 0]
        if len(sub_dry) < 20:
            sub_dry = sub

        X = sub_dry[["stint_lap"]].values.astype(np.float32)
        y = sub_dry["lap_duration"].values.astype(np.float32)

        # Normalizza: riferimento = mediana del primo stint_lap (0-2)
        baseline_mask = sub_dry["stint_lap"] <= 2
        baseline = sub_dry.loc[baseline_mask, "lap_duration"].median() if baseline_mask.any() else y.mean()
        y_delta = y - baseline  # lavoriamo sul delta rispetto al giro fresco

        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=True)),
            ("ridge", Ridge(alpha=1.0)),
        ])
        pipe.fit(X, y_delta)

        # Stima stint ottimale: quanti giri prima di +2s di degrado
        opt_lap = 1
        for lap in range(1, 55):
            delta_pred = pipe.predict(np.array([[lap]]))[0]
            if delta_pred > 2.0:
                break
            opt_lap = lap

        # Coefficienti
        ridge = pipe.named_steps["ridge"]
        coefs = {
            "intercept": round(float(ridge.intercept_), 4),
            "coef_linear": round(float(ridge.coef_[1]), 5),
            "coef_quad": round(float(ridge.coef_[2]), 6),
            "estimated_optimal_stint_laps": opt_lap,
            "baseline_lap_time": round(float(baseline), 3),
            "n_samples": len(sub_dry),
        }
        results[compound] = coefs
        models[compound] = pipe

        logger.info(f"  {compound:<14} deg/lap≈{coefs['coef_linear']:.4f}s  "
                    f"cliff≈{opt_lap}giri  n={len(sub_dry)}")

    path = MODEL_DIR / "degradation_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"models": models, "coefficients": results}, f)
    logger.info(f"  Salvato: {path}")

    return {"type": "Ridge(polynomial, degree=2) per compound", "compounds": results}


# ── Modello 3: CompoundRecommender ────────────────────────────────────────────

def train_compound_recommender(df: pd.DataFrame) -> dict:
    """
    Per ogni stint (groupby driver+session+compound sequence), calcola il
    tempo totale dello stint e confronta con gli altri compound disponibili
    in condizioni simili. Label = compound che ha dato il minor tempo medio
    in quella finestra meteo e lunghezza stint.
    """
    logger.info("\n=== Training CompoundRecommender (RandomForest) ===")

    # Aggrega per stint
    stint_groups = df.groupby(["session_key", "driver_number", "compound"]).agg(
        n_laps=("lap_duration", "count"),
        total_time=("lap_duration", "sum"),
        avg_lap=("lap_duration", "mean"),
        weather_enc=("weather_enc", "median"),
        air_temp=("air_temp", "median"),
        track_temp=("track_temp", "median"),
    ).reset_index()

    if len(stint_groups) < 50:
        logger.warning("  Troppo pochi stint aggregati, il recommender potrebbe essere inaccurato")

    # Feature per ogni stint
    X = stint_groups[["n_laps", "weather_enc", "air_temp", "track_temp"]].values.astype(np.float32)

    # Aggiungi sqrt(n_laps) per catturare non-linearità
    X = np.hstack([X, np.sqrt(X[:, 0:1])])

    # Label = compound (classificazione)
    le = LabelEncoder()
    y = le.fit_transform(stint_groups["compound"].values)

    # NOTA: il label ideale sarebbe "il compound col minor avg_lap in quella
    # sessione per quella lunghezza stint". Con più dati puoi raffinare questo
    # confrontando stint dello stesso giro con compound diversi.
    # Con dati reali, il proxy migliore è: RandomForest impara quali compound
    # vengono scelti dagli strategist in quelle condizioni (revealed preference).

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    logger.info(f"  Accuracy test: {acc:.3f}")
    logger.info(f"  Classi: {list(le.classes_)}")

    # Feature importance
    feat_names = ["n_laps", "weather_enc", "air_temp", "track_temp", "sqrt_n_laps"]
    importance = dict(zip(feat_names, model.feature_importances_.tolist()))
    logger.info("  Feature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {feat:<16} {imp:.3f}")

    path = MODEL_DIR / "compound_recommender.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "features": feat_names}, f)
    logger.info(f"  Salvato: {path}")

    return {
        "type": "RandomForestClassifier",
        "n_train": len(X_train),
        "accuracy_test": round(acc, 4),
        "classes": list(le.classes_),
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
            "weather_distribution": df["weather_enc"].value_counts().to_dict(),
            "lap_time_stats": {
                "mean": round(df["lap_duration"].mean(), 3),
                "std":  round(df["lap_duration"].std(), 3),
                "min":  round(df["lap_duration"].min(), 3),
                "max":  round(df["lap_duration"].max(), 3),
            },
        },
        "models": reports,
    }
    path = MODEL_DIR / "training_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"\nReport salvato: {path}")
    return report


# ── Entry point ────────────────────────────────────────────────────────────────

def run(csv_path: Path = None):
    if csv_path is None:
        csv_path = DATA_DIR / "laps_raw.csv"

    if not csv_path.exists():
        logger.error(
            f"Dataset non trovato: {csv_path}\n"
            f"Esegui prima: python collect_training_data.py"
        )
        return None

    df = load_and_clean(csv_path)

    if len(df) < 100:
        logger.error(f"Dataset troppo piccolo ({len(df)} giri). Scarica più dati.")
        return None

    reports = {}
    reports["lap_time_model"]       = train_lap_time_model(df)
    reports["degradation_model"]    = train_degradation_model(df)
    reports["compound_recommender"] = train_compound_recommender(df)

    report = save_training_report(reports, df)

    logger.info("\n=== Training completato ===")
    logger.info(f"Modelli salvati in: {MODEL_DIR}/")
    logger.info(f"Prossimo step: python main.py  (usa automaticamente i modelli in models/)")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Addestra modelli ML su dati reali OpenF1")
    parser.add_argument("--data", type=str, default=None, help="Path al CSV (default: data/laps_raw.csv)")
    args = parser.parse_args()

    run(csv_path=Path(args.data) if args.data else None)
