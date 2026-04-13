"""
ml_predictor.py
===============
Carica i modelli ML pre-addestrati su dati reali OpenF1 e li espone
come interfaccia compatibile con il race_simulator.

WORKFLOW:
  1. python collect_training_data.py   ← scarica dati reali da api.openf1.org
  2. python train_models.py            ← addestra e salva i modelli in models/
  3. python main.py                    ← usa i modelli qui sotto

I modelli vengono cercati in: models/ (radice del progetto)
Se non trovati, viene sollevato ModelNotTrainedError con istruzioni chiare.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

COMPOUND_ENC = {"soft": 0, "medium": 1, "hard": 2, "intermediate": 3, "wet": 4}
WEATHER_ENC = {"dry": 0, "light_rain": 1, "heavy_rain": 2}


class ModelNotTrainedError(RuntimeError):
    """Sollevato quando i modelli non sono stati ancora addestrati."""

    def __init__(self, model_name: str):
        super().__init__(f"\n[ML] Modello '{model_name}' non trovato...")


def _load_pickle(filename: str) -> dict:
    path = MODEL_DIR / filename
    if not path.exists():
        raise ModelNotTrainedError(filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ── LapTimePredictor ──────────────────────────────────────────────────────────
class LapTimePredictor:
    """
    Wrapper attorno al GradientBoostingRegressor addestrato su dati reali.
    Predice il tempo giro dato compound, stint_lap, meteo, temperatura.
    """

    def __init__(self):
        payload = _load_pickle("lap_time_model.pkl")
        self.model = payload["model"]
        self.features = payload["features"]
        logger.info("LapTimePredictor caricato da dati reali ✓")

    def predict(
        self,
        compound: str,
        stint_lap: int,
        weather: str,
        track_temp: float = 38.0,
        air_temp: float = 25.0,
        lap_number: int = 1,
        speed_range: float = 50.0,
        speed_mean: float = 210.0,
        sec1_norm: float = 0.0,
        sec2_norm: float = 0.0,
        sec3_norm: float = 0.0,
        humidity: float = 50.0,
        wind_speed: float = 5.0,
    ) -> float:
        input_data = {
            "compound_enc": COMPOUND_ENC.get(compound.lower(), 1),
            "stint_lap": stint_lap,
            "stint_lap_sq": stint_lap**2,
            "weather_enc": WEATHER_ENC.get(weather, 0),
            "air_temp": air_temp,
            "track_temp": track_temp,
            "lap_number": lap_number,
            "duration_sector_1_norm": sec1_norm,
            "duration_sector_2_norm": sec2_norm,
            "duration_sector_3_norm": sec3_norm,
            "speed_mean": speed_mean,
            "speed_range": speed_range,
            "humidity": humidity,
            "wind_speed": wind_speed,
        }

        X_ordered = np.array([[input_data[f] for f in self.features]], dtype=np.float32)
        return float(self.model.predict(X_ordered)[0])

    def evaluate(self, X_test=None, y_test=None) -> dict:
        """Carica metriche dal training report."""
        report_path = MODEL_DIR / "training_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            return report.get("models", {}).get("lap_time_model", {}).get("metrics", {})
        return {}

    def feature_importance(self) -> dict:
        report_path = MODEL_DIR / "training_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            return report.get("models", {}).get("lap_time_model", {}).get("feature_importance", {})
        gbr = self.model.named_steps["gbr"]
        return dict(zip(self.features, gbr.feature_importances_.tolist(), strict=False))


# ── DegradationModel ──────────────────────────────────────────────────────────
class DegradationModel:
    """
    Wrapper attorno ai modelli Ridge polynomial addestrati per compound.
    Stima il degrado (secondi aggiuntivi) dato il giro di stint.
    """

    def __init__(self):
        payload = _load_pickle("degradation_model.pkl")
        self.models = payload["models"]
        self.coefficients = payload.get("metadata", payload.get("coefficients", {}))
        logger.info(f"DegradationModel caricato per: {list(self.models.keys())} ✓")

    def predict_degradation(
        self, compound: str, stint_lap: int, track_temp: float = 38.0, speed_mean: float = 210.0, lap_number: int = 1
    ) -> float:
        compound = compound.lower()
        if compound not in self.models:
            logger.warning(f"Compound '{compound}' non nel modello, uso 0")
            return 0.0
        X = np.array([[stint_lap, track_temp, speed_mean, lap_number]], dtype=np.float32)
        delta = float(self.models[compound].predict(X)[0])
        return max(0.0, delta)

    def get_optimal_stint_length(self, compound: str, threshold_sec: float = 2.0) -> int:
        coefs = self.coefficients.get(compound.lower(), {})
        return coefs.get("estimated_optimal_stint_laps", 25)

    def summary(self) -> dict:
        return {
            "coefficients": self.coefficients,
            "optimal_stint_lengths": {c: self.get_optimal_stint_length(c) for c in self.coefficients},
        }


# ── CompoundRecommender ───────────────────────────────────────────────────────
class CompoundRecommender:
    """
    Wrapper attorno al RandomForestClassifier addestrato su stint reali.
    Raccomanda il compound ottimale dati: n_laps, meteo, temperatura.
    """

    def __init__(self):
        payload = _load_pickle("compound_recommender.pkl")
        self.model = payload["model"]
        self.label_encoder = payload["label_encoder"]
        self.features = payload["features"]
        logger.info("CompoundRecommender caricato da dati reali ✓")

    def recommend(
        self,
        n_laps: int,
        weather: str,
        stint_position: int = 1,
        track_temp: float = 38.0,
        air_temp: float = 25.0,
        lap_start: int = 1,
        speed_avg: float = 210.0,
        sec1_deg: float = 0.0,
        sec2_deg: float = 0.0,
        sec3_deg: float = 0.0,
    ) -> dict:
        weather_enc = WEATHER_ENC.get(weather, 0)
        X_base = np.array(
            [[n_laps, weather_enc, air_temp, track_temp, lap_start, speed_avg, sec1_deg, sec2_deg, sec3_deg]],
            dtype=np.float32,
        )

        sqrt_n_laps = np.sqrt(X_base[:, 0:1])
        X_final = np.hstack([X_base, sqrt_n_laps])

        pred_enc = self.model.predict(X_final)[0]
        proba = self.model.predict_proba(X_final)[0]
        compound = self.label_encoder.inverse_transform([pred_enc])[0]

        proba_dict = {self.label_encoder.inverse_transform([i])[0]: round(float(p), 3) for i, p in enumerate(proba)}

        return {
            "best_compound": compound,
            "probabilities": dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)),
            "confidence": round(float(proba.max()), 3),
        }


# ── StrategyEvaluator ─────────────────────────────────────────────────────────
class StrategyEvaluator:
    """
    Combina LapTimePredictor + DegradationModel + CompoundRecommender
    per valutare strategie usando stime ML su dati reali.
    """

    def __init__(self):
        self.lap_model = LapTimePredictor()
        self.deg_model = DegradationModel()
        self.recommender = CompoundRecommender()

    @classmethod
    def load(cls) -> StrategyEvaluator:
        """Carica tutti i modelli. Lancia ModelNotTrainedError se mancano."""
        return cls()

    def evaluate_strategy(
        self,
        strategy: list,
        conditions: dict,
        track_temp: float = 38.0,
        air_temp: float = 25.0,
    ) -> dict:
        """
        Simula giro per giro usando il LapTimePredictor reale.
        """
        total_laps = conditions.get("total_laps", 53)
        pit_loss = conditions.get("pit_lane_time_loss_seconds", 22.5)
        sc_info = conditions.get("safety_car", {})
        sc_active = sc_info.get("active", False)
        sc_lap = sc_info.get("lap", -1)
        sc_duration = sc_info.get("duration_laps", 0)
        rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
        rain_intensity = conditions.get("weather", {}).get("rain_intensity", "none")

        sorted_strat = sorted(strategy, key=lambda x: x["start_lap"])
        stints_ext = []
        for i, s in enumerate(sorted_strat):
            end_lap = sorted_strat[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_strat) else total_laps
            stints_ext.append({**s, "end_lap": end_lap, "compound": s["compound"].lower()})

        lap_to_stint = {}
        for st in stints_ext:
            for lap in range(st["start_lap"], st["end_lap"] + 1):
                lap_to_stint[lap] = st

        total_time = 0.0
        lap_results = []
        pit_stops = 0
        per_compound = {}

        for lap in range(1, total_laps + 1):
            stint = lap_to_stint.get(lap)
            if not stint:
                total_time += 90.0
                continue

            compound = stint["compound"]
            stint_lap = lap - stint["start_lap"]

            weather = ("heavy_rain" if rain_intensity == "heavy" else "light_rain") if lap >= rain_start else "dry"

            is_sc = sc_active and sc_lap <= lap < sc_lap + sc_duration

            lap_time = 108.0 if is_sc else self.lap_model.predict(compound, stint_lap, weather, track_temp, air_temp, lap)

            is_pit = lap == stint["start_lap"] and lap > 1
            if is_pit:
                total_time += pit_loss
                pit_stops += 1

            total_time += lap_time
            per_compound[compound] = per_compound.get(compound, 0.0) + lap_time
            lap_results.append(
                {
                    "lap": lap,
                    "lap_time": round(lap_time, 3),
                    "compound": compound,
                    "stint_lap": stint_lap,
                    "weather": weather,
                    "is_pit_lap": is_pit,
                    "is_sc_lap": is_sc,
                }
            )

        return {
            "strategy": strategy,
            "total_time": round(total_time, 2),
            "pit_stops": pit_stops,
            "lap_results": lap_results,
            "breakdown": {
                "time_by_compound": {k: round(v, 2) for k, v in per_compound.items()},
                "total_pit_time": round(pit_stops * pit_loss, 2),
                "sc_laps": sum(1 for lr in lap_results if lr["is_sc_lap"]),
            },
        }

    def recommend_strategy(self, conditions: dict, track_temp: float = 38.0) -> dict:
        """Raccomanda compound per ogni stint usando il modello reale."""
        total_laps = conditions.get("total_laps", 53)
        rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
        sc_lap = conditions.get("safety_car", {}).get("lap", -1) if conditions.get("safety_car", {}).get("active") else -1

        suggestions = []
        stint_num = 1

        # Stint 1: asciutto fino a SC o pioggia
        end1 = sc_lap if sc_lap > 0 else (rain_start - 1 if rain_start < total_laps else total_laps)
        n1 = max(1, end1 - 1)
        rec1 = self.recommender.recommend(n1, "dry", stint_position=0, track_temp=track_temp)
        suggestions.append(
            {
                "stint": stint_num,
                "start_lap": 1,
                "recommended_compound": rec1["best_compound"],
                "confidence": rec1["confidence"],
                "probabilities": rec1["probabilities"],
                "rationale": f"{n1} giri su asciutto",
            }
        )
        stint_num += 1

        # Stint 2: se c'è SC prima della pioggia
        if sc_lap > 0 and rain_start < total_laps and sc_lap < rain_start:
            n2 = max(1, rain_start - sc_lap)
            rec2 = self.recommender.recommend(n2, "dry", stint_position=1, track_temp=track_temp)
            suggestions.append(
                {
                    "stint": stint_num,
                    "start_lap": sc_lap,
                    "recommended_compound": rec2["best_compound"],
                    "confidence": rec2["confidence"],
                    "probabilities": rec2["probabilities"],
                    "rationale": f"{n2} giri su asciutto (post-SC)",
                }
            )
            stint_num += 1

        # Stint finale: con pioggia
        if rain_start < total_laps:
            n_rain = total_laps - rain_start + 1
            rec_r = self.recommender.recommend(n_rain, "light_rain", stint_position=2, track_temp=track_temp)
            suggestions.append(
                {
                    "stint": stint_num,
                    "start_lap": rain_start,
                    "recommended_compound": rec_r["best_compound"],
                    "confidence": rec_r["confidence"],
                    "probabilities": rec_r["probabilities"],
                    "rationale": f"{n_rain} giri con pioggia leggera",
                }
            )

        return {"ml_recommended_stints": suggestions}

    def get_models_info(self) -> dict:
        metrics = self.lap_model.evaluate()
        return {
            "lap_time_predictor": {
                "type": "GradientBoostingRegressor",
                "trained_on": "dati reali OpenF1",
                "metrics": metrics,
                "feature_importance": self.lap_model.feature_importance(),
            },
            "degradation_model": {
                "type": "Ridge Regression (polynomial degree=2) per compound",
                "trained_on": "dati reali OpenF1",
                "summary": self.deg_model.summary(),
            },
            "compound_recommender": {
                "type": "RandomForestClassifier",
                "trained_on": "stint reali OpenF1",
                "classes": list(self.recommender.label_encoder.classes_),
            },
        }
