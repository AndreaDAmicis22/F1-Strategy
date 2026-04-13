"""
train_models.py
===============
Addestra i modelli ML sui dati REALI scaricati da OpenF1 (via collect_training_data.py).

Uso:
    python src/f1_strategy/train_models.py --data data/all_circuits_laps.csv
    python src/f1_strategy/train_models.py --data data/all_circuits_laps.csv --test-circuit Monza

Output:
    models/lap_time_model.pkl        — GradientBoostingRegressor tempi giro
    models/degradation_model.pkl     — Ridge polinomiale degrado per compound
    models/compound_recommender.pkl  — RandomForest classificatore compound
    models/training_report.json      — metriche, feature importance, distribuzione dati
"""

import argparse
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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

COMPOUND_ENC = {"soft": 0, "medium": 1, "hard": 2, "intermediate": 3, "wet": 4}

LAP_TIME_MIN = 60.0
LAP_TIME_MAX = 200.0


# ── Carica e pulisce il dataset ────────────────────────────────────────────────
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Caricamento dati: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    logger.info(f"  Righe raw: {len(df)}")

    df = df[df["lap_duration"].between(LAP_TIME_MIN, LAP_TIME_MAX)]
    df = df[df["compound"].isin(COMPOUND_ENC.keys())]
    df = df[df["is_pit_out_lap"] != True]  # noqa: E712
    df = df[df["is_sc_lap"] != True]  # noqa: E712

    df["compound_enc"] = df["compound"].map(COMPOUND_ENC)

    df["rainfall"] = df["rainfall"].fillna(0.0)
    df["weather_enc"] = df["rainfall"].apply(lambda r: 2 if r > 2.0 else (1 if r > 0.1 else 0))

    df["air_temp"] = df.groupby("session_key")["air_temp"].transform(lambda x: x.fillna(x.median()))
    df["track_temp"] = df.groupby("session_key")["track_temp"].transform(lambda x: x.fillna(x.median()))
    df["air_temp"] = df["air_temp"].fillna(25.0)
    df["track_temp"] = df["track_temp"].fillna(38.0)

    df["stint_lap_sq"] = df["stint_lap"] ** 2

    # Normalizza il tempo giro rispetto al best pace della sessione.
    # Rimuove l'effetto "circuito diverso" (Monaco 75s vs Monza 83s vs Spa 107s) e lascia solo il segnale di degrado relativo.
    df["lap_duration_norm"] = df.groupby("session_key")["lap_duration"].transform(lambda x: x - x.quantile(0.05))

    # Velocità medie: proxy diretto del grip disponibile
    df["speed_mean"] = df[["i1_speed", "i2_speed", "st_speed"]].mean(axis=1)
    # Differenza velocità tra tratti: indica bilanciamento aerodinamico/gomme
    df["speed_range"] = df["st_speed"] - df[["i1_speed", "i2_speed"]].min(axis=1)

    # Settori normalizzati per sessione (stesso motivo di lap_duration_norm)
    for sec in ["duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        df[f"{sec}_norm"] = df.groupby("session_key")[sec].transform(lambda x: x - x.quantile(0.05))

    # Imputa NaN nelle colonne nuove
    speed_cols = ["i1_speed", "i2_speed", "st_speed", "speed_mean", "speed_range"]
    sector_cols = ["duration_sector_1_norm", "duration_sector_2_norm", "duration_sector_3_norm"]
    for col in speed_cols + sector_cols:
        df[col] = df.groupby(["session_key", "compound"])[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    for col in sector_cols + speed_cols:
        null_pct = df[col].isna().mean()
        if null_pct > 0.3:
            logger.warning(f"  {col}: {null_pct:.0%} NaN dopo imputazione, esclusa dalle features")
            LAP_FEATURES.remove(col) if col in LAP_FEATURES else None

    df["humidity"] = df["humidity"].fillna(df["humidity"].median())
    df["wind_speed"] = df["wind_speed"].fillna(0.0)

    logger.info(f"  Righe dopo pulizia: {len(df)}")
    logger.info(f"  Sessioni: {df['session_key'].nunique()}")
    logger.info(f"  Circuiti: {df['location'].nunique() if 'location' in df.columns else 'N/A'}")
    logger.info(f"  Compound distribution:\n{df['compound'].value_counts().to_string()}")
    return df


def _temporal_split(df: pd.DataFrame, test_circuit: str | None = None, test_ratio: float = 0.15):
    """
    Divide train/test in modo temporalmente corretto.

    Se test_circuit è specificato (es. "Monza"):
        train = tutte le gare tranne Monza
        test  = solo Monza
    Altrimenti:
        train = sessioni più vecchie
        test  = sessioni più recenti

    In entrambi i casi non c'è mai data leakage: il test
    contiene solo gare che il modello non ha mai visto.
    """
    if test_circuit:
        loc_col = "location" if "location" in df.columns else None
        if loc_col and df[loc_col].str.lower().str.contains(test_circuit.lower()).any():
            train_df = df[~df[loc_col].str.lower().str.contains(test_circuit.lower())]
            test_df = df[df[loc_col].str.lower().str.contains(test_circuit.lower())]
            logger.info(f"  Split per circuito: train=tutto tranne {test_circuit}, test={test_circuit}")
            return train_df, test_df
        logger.warning(f"  Circuito '{test_circuit}' non trovato, uso split temporale")

    # Split temporale: ordina sessioni per session_key (proxy dell'ordine cronologico)
    sessions = sorted(df["session_key"].unique())
    n_test = max(1, int(len(sessions) * test_ratio))
    test_sess = sessions[-n_test:]
    train_sess = sessions[:-n_test]
    train_df = df[df["session_key"].isin(train_sess)]
    test_df = df[df["session_key"].isin(test_sess)]
    logger.info(f"  Split temporale ({test_ratio:.0%}): train={len(train_sess)} sessioni, test={len(test_sess)} sessioni")
    return train_df, test_df


# ── Feature engineering ────────────────────────────────────────────────────────
LAP_FEATURES = [
    "compound_enc",
    "stint_lap",
    "stint_lap_sq",
    "weather_enc",
    "air_temp",
    "track_temp",
    "lap_number",
    "duration_sector_1_norm",  # degrado settore 1
    "duration_sector_2_norm",  # degrado settore 2
    "duration_sector_3_norm",  # degrado settore 3
    "speed_mean",  # grip medio
    "speed_range",  # bilanciamento aero/gomme
    "humidity",  # condizioni aria
    "wind_speed",  # vento (impatta downforce a Monza)
]


def build_lap_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[LAP_FEATURES].values.astype(np.float32)
    y = df["lap_duration"].values.astype(np.float32)
    return X, y


# ── Modello 1: LapTimePredictor ────────────────────────────────────────────────
def train_lap_time_model(df: pd.DataFrame, test_circuit: str | None = None) -> dict:
    logger.info("\n=== Training LapTimePredictor (GradientBoosting) ===")

    train_df, test_df = _temporal_split(df, test_circuit)
    logger.info(f"  Train: {len(train_df)} giri  |  Test: {len(test_df)} giri")

    X_train, y_train = build_lap_features(train_df)
    X_test, y_test = build_lap_features(test_df)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.07,
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

    # Cross-validation solo sul train set (corretto: non tocca il test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    logger.info(f"  MAE  test: {mae:.4f}s")
    logger.info(f"  RMSE test: {rmse:.4f}s")
    logger.info(f"  R²   test: {r2:.4f}")
    logger.info(f"  CV MAE (5-fold, solo train): {cv_mae:.4f}s")

    gbr = model.named_steps["gbr"]
    importance = dict(zip(LAP_FEATURES, gbr.feature_importances_.tolist(), strict=False))
    logger.info("  Feature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        logger.info(f"    {feat:<16} {imp:.3f}  {bar}")

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


# ── Modello 2: DegradationModel ───────────────────────────────────────────────
def train_degradation_model(df: pd.DataFrame, test_circuit: str | None = None) -> dict:
    logger.info("\n=== Training DegradationModel (Enhanced Ridge) ===")

    train_df, test_df = _temporal_split(df, test_circuit)

    DEGR_FEATURES = ["stint_lap", "track_temp", "speed_mean", "lap_number"]

    results = {}
    models = {}

    for compound in COMPOUND_ENC:
        sub_train = train_df[train_df["compound"] == compound]
        sub_test = test_df[test_df["compound"] == compound]

        if len(sub_train) < 50:
            continue

        # Target = delta rispetto al best pace (5° percentile) della sessione
        y_train = sub_train["lap_duration_norm"].values.astype(np.float32)
        X_train = sub_train[DEGR_FEATURES].values.astype(np.float32)

        X_test = sub_test[DEGR_FEATURES].values.astype(np.float32) if len(sub_test) > 0 else None
        y_test = sub_test["lap_duration_norm"].values.astype(np.float32) if len(sub_test) > 0 else None

        # Pipeline con StandardScaler (fondamentale ora che le features hanno scale diverse)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=10.0)),
            ]
        )

        pipe.fit(X_train, y_train)

        test_mae = None
        if X_test is not None:
            preds = pipe.predict(X_test)
            test_mae = round(float(mean_absolute_error(y_test, preds)), 4)

        # Salvataggio
        models[compound] = pipe
        results[compound] = {"n_train": len(sub_train), "test_mae": test_mae, "features": DEGR_FEATURES}

        mae_str = f" test_MAE={test_mae:.4f}s" if test_mae else ""
        logger.info(f"  {compound:<14} {len(sub_train)} giri {mae_str}")

    path = MODEL_DIR / "degradation_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"models": models, "metadata": results}, f)

    return {"type": "Multi-feature Ridge Polynomial", "compounds": results}


# ── Modello 3: CompoundRecommender ────────────────────────────────────────────
def train_compound_recommender(df: pd.DataFrame, test_circuit: str | None = None) -> dict:
    logger.info("\n=== Training CompoundRecommender (RandomForest) ===")

    group_cols = ["session_key", "driver_number", "compound"]
    if "location" in df.columns:
        group_cols.append("location")

    # Aggrega per stint prima di splittare
    stint_groups = (
        df.groupby(group_cols)
        .agg(
            n_laps=("lap_duration", "count"),
            weather_enc=("weather_enc", "median"),
            air_temp=("air_temp", "median"),
            track_temp=("track_temp", "median"),
            lap_start=("lap_number", "min"),
            speed_avg=("speed_mean", "mean"),
            sec1_deg=("duration_sector_1_norm", "mean"),
            sec2_deg=("duration_sector_2_norm", "mean"),
            sec3_deg=("duration_sector_3_norm", "mean"),
        )
        .reset_index()
    )

    if len(stint_groups) < 50:
        logger.warning("  Troppo pochi stint aggregati, il recommender potrebbe essere inaccurato")

    # Split temporale sugli stint (stesso criterio degli altri modelli)
    train_stint, test_stint = _temporal_split(stint_groups, test_circuit)

    def make_X(sdf):
        feats = [
            "n_laps",
            "weather_enc",
            "air_temp",
            "track_temp",
            "lap_start",
            "speed_avg",
            "sec1_deg",
            "sec2_deg",
            "sec3_deg",
        ]
        X = sdf[feats].values.astype(np.float32)
        return np.hstack([X, np.sqrt(X[:, 0:1])])

    X_train = make_X(train_stint)
    X_test = make_X(test_stint)

    le = LabelEncoder()
    le.fit(stint_groups["compound"].values)
    y_train = le.transform(train_stint["compound"].values)
    y_test = le.transform(test_stint["compound"].values)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    logger.info(f"  Accuracy test: {acc:.3f}  (train={len(X_train)}, test={len(X_test)})")
    logger.info(f"  Classi: {list(le.classes_)}")

    feat_names = [
        "n_laps",
        "weather_enc",
        "air_temp",
        "track_temp",
        "lap_start",
        "speed_avg",
        "sec1_deg",
        "sec2_deg",
        "sec3_deg",
        "sqrt_n_laps",
    ]
    importance = dict(zip(feat_names, model.feature_importances_.tolist(), strict=False))
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
        "n_test": len(X_test),
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

    if len(df) < 100:
        logger.error(f"Dataset troppo piccolo ({len(df)} giri). Scarica più dati.")
        return None

    reports = {}
    reports["lap_time_model"] = train_lap_time_model(df, test_circuit)
    reports["degradation_model"] = train_degradation_model(df, test_circuit)
    reports["compound_recommender"] = train_compound_recommender(df, test_circuit)

    report = save_training_report(reports, df)

    logger.info("\n=== Training completato ===")
    logger.info(f"Modelli salvati in: {MODEL_DIR}/")
    logger.info("Prossimo step: python src/f1_strategy/main.py")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestra modelli ML su dati reali OpenF1")
    parser.add_argument("--data", type=str, default=None, help="Path al CSV")
    parser.add_argument(
        "--test-circuit",
        type=str,
        default=None,
        help="Circuito da usare come test set (es. 'Monza'). Default: split temporale.",
    )
    args = parser.parse_args()

    run(
        csv_path=Path(args.data) if args.data else None,
        test_circuit=args.test_circuit,
    )
