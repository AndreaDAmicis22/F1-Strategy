"""
ml_predictor.py
===============
Carica i 3 modelli ML e li espone tramite due classi principali:

  StrategyEvaluator  — valuta una strategia giro per giro con ML
  StrategyValidator  — valida e assegna un punteggio a una strategy.json esterna
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

NORMAL_PIT_COST_SECONDS = 22.5


class ModelNotTrainedError(RuntimeError):
    def __init__(self, name: str):
        super().__init__(
            f"\n[ML] Modello '{name}' non trovato in {MODEL_DIR}/\n"
            f"Esegui prima:\n"
            f"  1. python collect_training_data.py\n"
            f"  2. python train_models.py\n"
        )


def _load_pickle(filename: str) -> dict:
    path = MODEL_DIR / filename
    if not path.exists():
        raise ModelNotTrainedError(filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ── LapTimePredictor ──────────────────────────────────────────────────────────
class LapTimePredictor:
    """GBR che predice il tempo giro dato compound, stint_lap, meteo, temperatura."""

    def __init__(self):
        payload = _load_pickle("lap_time_model.pkl")
        self.model = payload["model"]
        self.features = payload["features"]
        logger.info(f"LapTimePredictor caricato ({len(self.features)} features) ✓")

    def predict(
        self,
        compound: str,
        stint_lap: int,
        weather: str,
        track_temp: float = 38.0,
        air_temp: float = 25.0,
        lap_number: int = 1,
        max_session_laps: int = 55,
        prev_lap_duration: float = 90.0,
        **kwargs,
    ) -> float:
        """Predice il tempo giro. kwargs accetta feature extra (speed_mean, ecc.)."""
        data = {
            "compound_enc": COMPOUND_ENC.get(compound.lower(), 1),
            "stint_lap": stint_lap,
            "stint_lap_sq": stint_lap**2,
            "weather_enc": WEATHER_ENC.get(weather, 0),
            "air_temp": air_temp,
            "track_temp": track_temp,
            "lap_number": lap_number,
            "humidity": kwargs.get("humidity", 50.0),
            "wind_speed": kwargs.get("wind_speed", 5.0),
        }

        # 2. Settori e Velocità (Normalizzati o medi)
        data.update(
            {
                "duration_sector_1_norm": kwargs.get("sec1_norm", 0.1),
                "duration_sector_2_norm": kwargs.get("sec2_norm", 0.1),
                "duration_sector_3_norm": kwargs.get("sec3_norm", 0.1),
                "speed_mean": kwargs.get("speed_mean", 210.0),
                "speed_range": kwargs.get("speed_range", 50.0),
            }
        )

        data["session_progression"] = lap_number / max_session_laps
        data["speed_efficiency"] = kwargs.get("speed_efficiency", 1.0)
        data["prev_lap_duration"] = prev_lap_duration
        data["avg_energy_session"] = kwargs.get("avg_energy_session", (data["speed_mean"] * track_temp))

        try:
            X = np.array([[data[f] for f in self.features]], dtype=np.float32)
            return float(self.model.predict(X)[0])
        except KeyError as e:
            logger.exception(f"Feature richiesta dal modello ma non calcolata nel predictor: {e}")
            raise

    def evaluate(self) -> dict:
        report_path = MODEL_DIR / "training_report.json"
        if report_path.exists():
            r = json.loads(report_path.read_text())
            return r.get("models", {}).get("lap_time_model", {}).get("metrics", {})
        return {}

    def feature_importance(self) -> dict:
        report_path = MODEL_DIR / "training_report.json"
        if report_path.exists():
            r = json.loads(report_path.read_text())
            return r.get("models", {}).get("lap_time_model", {}).get("feature_importance", {})
        return dict(zip(self.features, self.model.named_steps["gbr"].feature_importances_.tolist(), strict=False))


# ── DegradationRiskModel ──────────────────────────────────────────────────────
class DegradationRiskModel:
    """
    Ridge polynomial per compound.
    Dato uno stint (compound + n_laps) stima:
      - degrado totale accumulato (secondi extra rispetto al giro fresco)
      - se lo stint supera il cliff del compound
      - penalità in secondi per eccesso di usura
    """

    def __init__(self):
        payload = _load_pickle("degradation_model.pkl")
        self.models = payload["models"]
        self.metadata = payload.get("metadata", payload.get("coefficients", {}))
        logger.info(f"DegradationRiskModel caricato per: {list(self.models.keys())} ✓")

    def get_cliff(self, compound: str) -> int:
        """Ritorna il giro (stint_lap) dove il degrado supera +2s."""
        return self.metadata.get(compound.lower(), {}).get("cliff_lap", 30)

    def predict_step(
        self,
        compound: str,
        stint_lap: int,
        weather: str,
        track_temp: float = 38.0,
        air_temp: float = 25.0,
        lap_number: int = 1,
        max_session_laps: int = 55,
        prev_lap_duration: float = 90.0,
        **kwargs,
    ) -> float:
        """Predice l'incremento di degrado usando solo i dati disponibili nei JSON."""
        compound_key = compound.lower()
        if compound_key not in self.models:
            return 0.0

        # Mapping pulito: solo quello che passa dal simulatore o dai JSON
        data = {
            "stint_lap": stint_lap,
            "stint_lap_sq": stint_lap**2,
            "track_temp": track_temp,
            "air_temp": air_temp,
            "lap_number": lap_number,
            "session_progression": lap_number / max(max_session_laps, 1),
            "weather_enc": WEATHER_ENC.get(weather, 0),
            "compound_enc": COMPOUND_ENC.get(compound_key, 1),
            # Feature di contesto base (fallback se non presenti in race_conditions.json)
            "speed_mean": kwargs.get("speed_mean", 210.0),
            "avg_energy_session": kwargs.get("avg_energy_session", (210.0 * track_temp) / 100),
            "prev_lap_duration": prev_lap_duration,
            "humidity": kwargs.get("humidity", 50.0),
        }

        features = self.metadata.get(compound_key, {}).get("features", [])

        try:
            row = [data.get(f, 0.0) for f in features]
            X = np.array([row], dtype=np.float32)
            step_pred = float(self.models[compound_key].predict(X)[0])
            offset = self.metadata.get(compound_key, {}).get("offset_g1", 0.0)
            return max(0.0, step_pred - offset)

        except Exception as e:
            logger.exception(f"Errore predizione step {compound_key}: {e}")
            return 0.0

    def assess_stint_risk(self, compound: str, stint_length: int, **kwargs) -> dict:
        """
        Valuta il rischio basandosi sull'accumulo degli step predetti.
        """
        cliff_lap = self.get_cliff(compound)

        # Estraiamo i parametri necessari per predict_step, con fallback
        # 'weather' è ora obbligatorio in predict_step, quindi diamogli un default qui
        weather = kwargs.get("weather", "dry")

        accumulated_degr = 0.0
        degr_history = []

        for lap in range(1, stint_length + 1):
            # Passiamo esplicitamente weather e il resto tramite kwargs
            step = self.predict_step(compound, lap, weather=weather, **kwargs)
            accumulated_degr += step
            degr_history.append(accumulated_degr)

        final_degr = accumulated_degr
        exceeds = stint_length > cliff_lap

        penalty_avg = 0.0
        if exceeds:
            # Calcolo corretto della media sui giri oltre il cliff
            # degr_history è 0-indexed, quindi il giro cliff_lap è all'indice cliff_lap-1
            beyond_cliff = degr_history[cliff_lap:]
            if beyond_cliff:
                penalty_avg = sum(beyond_cliff) / len(beyond_cliff)

        return {
            "compound": compound,
            "stint_length": stint_length,
            "cliff_lap": cliff_lap,
            "total_degradation_seconds": round(final_degr, 2),
            "exceeds_cliff": exceeds,
            "penalty_seconds": round(penalty_avg, 2),
            "risk_level": "HIGH"
            if final_degr > 2.5 or (exceeds and penalty_avg > 1.5)
            else ("MEDIUM" if exceeds else "LOW"),
        }


# ── SafetyCarImpactModel ──────────────────────────────────────────────────────
class SafetyCarImpactModel:
    """
    GBR che stima il costo effettivo di un pit stop in relazione alla Safety Car.
    Un pit durante SC ha costo ≈0s (tutti rallentano).
    Un pit lontano dalla SC ha costo ≈pit_lane_time (22+ secondi).
    """

    def __init__(self):
        payload = _load_pickle("sc_impact_model.pkl")
        self.model = payload.get("model")
        self.features = payload.get("features", [])
        self.fallback = payload.get("fallback", True)
        if not self.fallback:
            logger.info("SafetyCarImpactModel caricato ✓")
        else:
            logger.info("SafetyCarImpactModel: usando fallback analitico")

    def predict_pit_cost(
        self,
        pit_lap: int,
        sc_lap: int | None,
        compound: str,
        weather: str = "dry",
        track_temp: float = 38.0,
        pit_lane_seconds: float = NORMAL_PIT_COST_SECONDS,
        **kwargs,
    ) -> float:
        """
        Stima il costo effettivo di un pit stop in secondi.
        """
        # 1. Fallback se il modello non è stato addestrato
        if self.model is None or self.fallback:
            if sc_lap is not None and abs(pit_lap - sc_lap) <= 2:
                return 12.0  # Costo medio stimato sotto SC reale
            return pit_lane_seconds

        # 2. Preparazione dati (Mapping delle nuove features)
        sc_active = sc_lap is not None
        dist = (pit_lap - sc_lap) if sc_active else 999

        # sc_speed_ratio: 0.4 per Full SC, 0.7 per VSC, 1.0 per Green Flag
        sc_speed_ratio = kwargs.get("sc_speed_ratio")
        if sc_speed_ratio is None:
            sc_speed_ratio = 0.4 if sc_active else 1.0

        data_map = {
            "dist_to_sc": dist,
            "has_sc": 1 if sc_active else 0,
            "sc_speed_ratio": sc_speed_ratio,
            "sc_duration_step": kwargs.get("sc_duration_step", (dist if sc_active and dist > 0 else 0)),
            "compound_enc": COMPOUND_ENC.get(compound.lower(), 1),
            "lap_number": pit_lap,
            "weather_enc": WEATHER_ENC.get(weather, 0),
            "track_temp": track_temp,
            "rainfall": kwargs.get("rainfall", 0.0),
        }

        # 3. Prediction
        try:
            row = [data_map.get(f, 0.0) for f in self.features]
            X = np.array([row], dtype=np.float32)
            # Il costo predetto è il tempo extra rispetto al giro ideale
            return max(0.0, float(self.model.predict(X)[0]))
        except Exception as e:
            logger.warning(f"Errore prediction SC: {e}. Uso fallback.")
            return pit_lane_seconds if not sc_active else 12.0

    def evaluate_strategy_sc_timing(self, strategy: list, conditions: dict) -> dict:
        sc_active = conditions.get("safety_car", {}).get("active", False)
        sc_lap = conditions.get("safety_car", {}).get("lap") if sc_active else None
        sc_duration = conditions.get("safety_car", {}).get("duration_laps", 3)
        pit_lane_base = conditions.get("pit_lane_time_loss_seconds", 23.5)

        # Parametri dinamici per la SC
        sc_params = {
            "sc_speed_ratio": conditions.get("safety_car", {}).get("speed_ratio", 0.4 if sc_active else 1.0),
            "rainfall": conditions.get("weather", {}).get("rainfall", 0.0),
            "sc_duration_step": conditions.get("safety_car", {}).get("duration_step", 0),
        }

        pit_stops = []
        for stint in strategy[1:]:
            pit_lap = stint["start_lap"]

            # Calcoliamo il costo con il modello predetto
            cost = self.predict_pit_cost(pit_lap, sc_lap, stint["compound"], pit_lane_seconds=pit_lane_base, **sc_params)
            saving = max(0.0, pit_lane_base - cost)

            # FORZATURA LOGICA: Se il pit è esattamente durante la finestra SC,
            # la qualità DEVE essere EXCELLENT, a prescindere da piccole oscillazioni del modello ML
            is_during_sc = False
            if sc_active and sc_lap is not None and sc_lap <= pit_lap <= (sc_lap + sc_duration):
                is_during_sc = True

            # Assegnazione qualità migliorata
            if is_during_sc or saving > 12:
                quality = "EXCELLENT"
            elif saving > 5:
                quality = "GOOD"
            else:
                quality = "NEUTRAL"

            pit_stops.append(
                {
                    "pit_lap": pit_lap,
                    "estimated_cost": round(cost, 2),
                    "sc_saving": round(saving, 2),
                    "timing_quality": quality,
                }
            )

        return {
            "pit_stops": pit_stops,
            "total_sc_saving_seconds": round(sum(p["sc_saving"] for p in pit_stops), 2),
            "sc_active": sc_active,
        }


# ── StrategyEvaluator ─────────────────────────────────────────────────────────
# ── StrategyEvaluator ─────────────────────────────────────────────────────────
class StrategyEvaluator:
    """
    Valuta una strategia giro per giro usando LapTimePredictor.
    """

    def __init__(self):
        self.lap_model = LapTimePredictor()
        self.deg_model = DegradationRiskModel()
        self.sc_model = SafetyCarImpactModel()

    @classmethod
    def load(cls) -> StrategyEvaluator:
        return cls()

    def evaluate_strategy(self, strategy: list, conditions: dict, track_temp: float = 38.0, air_temp: float = 25.0) -> dict:
        total_laps = conditions.get("total_laps", 53)
        pit_lane_base = conditions.get("pit_lane_time_loss_seconds", NORMAL_PIT_COST_SECONDS)

        # Info Safety Car
        sc_info = conditions.get("safety_car", {})
        sc_active = sc_info.get("active", False)
        sc_lap_trigger = sc_info.get("lap", -1)
        sc_duration = sc_info.get("duration_laps", 0)
        # IMPORTANTE: In F1 il Delta SC è circa +40/60% sul tempo sul giro.
        # Usiamo un moltiplicatore, non una divisione drastica.
        sc_time_multiplier = 1.5

        # Info Meteo
        weather_info = conditions.get("weather", {})
        rain_start = weather_info.get("rain_start_lap", 999)
        rain_intens = weather_info.get("rain_intensity", "none")
        rainfall = weather_info.get("rainfall", 0.0)

        # Pre-processamento stint
        sorted_strat = sorted(strategy, key=lambda x: x["start_lap"])
        stints_ext = []
        for i, s in enumerate(sorted_strat):
            end_lap = sorted_strat[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_strat) else total_laps
            stints_ext.append({**s, "end_lap": end_lap, "compound": s["compound"].lower()})

        lap_to_stint = {lap: st for st in stints_ext for lap in range(st["start_lap"], st["end_lap"] + 1)}

        total_time = 0.0
        lap_results = []
        pit_stops = 0
        current_prev_lap_time = 90.0  # Valore di partenza realistico per Monza
        accumulated_usura = 0.0
        current_stint_start = -1

        for lap in range(1, total_laps + 1):
            stint = lap_to_stint.get(lap)
            if not stint:
                continue

            # RESET USURA AL PIT STOP
            if stint["start_lap"] != current_stint_start:
                accumulated_usura = 0.0
                current_stint_start = stint["start_lap"]

            compound = stint["compound"]
            stint_lap = lap - stint["start_lap"] + 1
            weather = ("heavy_rain" if rain_intens == "heavy" else "light_rain") if lap >= rain_start else "dry"

            # Check se siamo sotto Safety Car
            is_sc_active_this_lap = sc_active and sc_lap_trigger <= lap < (sc_lap_trigger + sc_duration)

            # 1. GESTIONE PIT STOP
            is_pit = lap == stint["start_lap"] and lap > 1
            pit_cost = 0.0
            if is_pit:
                sc_lap_val = sc_lap_trigger if sc_active else None
                pit_cost = self.sc_model.predict_pit_cost(
                    pit_lap=lap,
                    sc_lap=sc_lap_val,
                    compound=compound,
                    weather=weather,
                    track_temp=track_temp,
                    pit_lane_seconds=pit_lane_base,
                    # Parametri corretti per il modello
                    sc_speed_ratio=0.4 if is_sc_active_this_lap else 1.0,
                    rainfall=rainfall if lap >= rain_start else 0.0,
                )
                total_time += pit_cost
                pit_stops += 1

            # 2. CALCOLO LAP TIME
            if is_sc_active_this_lap:
                # SOTTO SC: Il tempo è il tempo precedente * moltiplicatore (es. 90s * 1.5 = 135s)
                lap_time = current_prev_lap_time * sc_time_multiplier
                # Sotto SC l'usura è ridotta dell'80%
                step_degr = self.deg_model.predict_step(compound=compound, stint_lap=stint_lap, weather=weather) * 0.2
                accumulated_usura += max(0.0, step_degr)
            else:
                # PASSO NORMALE (GBR)
                lap_time = self.lap_model.predict(
                    compound=compound,
                    stint_lap=stint_lap,
                    weather=weather,
                    track_temp=track_temp,
                    air_temp=air_temp,
                    lap_number=lap,
                    max_session_laps=total_laps,
                    prev_lap_duration=current_prev_lap_time,
                )

                step_degr = self.deg_model.predict_step(
                    compound=compound,
                    stint_lap=stint_lap,
                    weather=weather,
                    track_temp=track_temp,
                    air_temp=air_temp,
                    lap_number=lap,
                    max_session_laps=total_laps,
                    prev_lap_duration=current_prev_lap_time,
                )
                accumulated_usura += max(0.0, step_degr)
                lap_time += accumulated_usura

                # Guardrail Fisici (Meteo)
                is_slick = compound.lower() in {"soft", "medium", "hard"}
                if rain_intens == "light" and is_slick:
                    lap_time += 5.0
                elif rain_intens == "heavy" and is_slick:
                    lap_time += 15.0  # Più severo su heavy
                elif rain_intens == "dry" and compound.lower() in {"intermediate", "wet"}:
                    lap_time += 4.0

            total_time += lap_time
            current_prev_lap_time = lap_time

            lap_results.append(
                {
                    "lap": lap,
                    "lap_time": round(lap_time, 3),
                    "compound": compound,
                    "is_pit_lap": is_pit,
                    "is_sc_lap": is_sc_active_this_lap,
                    "cumulative_time": round(total_time, 3),
                    "tire_degradation": round(accumulated_usura, 3),
                }
            )

        return {"total_time": round(total_time, 3), "pit_stops": pit_stops, "lap_results": lap_results}

    def get_models_info(self) -> dict:
        """Ritorna i metadati e le performance dei modelli caricati."""
        metrics = self.lap_model.evaluate()

        # Recuperiamo i metadati del modello di degradazione per mostrare le soglie usate
        degr_metadata = getattr(self.deg_model, "metadata", {})

        return {
            "lap_time_predictor": {
                "type": "GradientBoostingRegressor (GBR)",
                "trained_on": "OpenF1 Real Data",
                "features": self.lap_model.features,
                "metrics": metrics,
                "feature_importance": self.lap_model.feature_importance(),
            },
            "degradation_risk_model": {
                "type": "Incremental Step GBR (per compound)",
                "trained_on": "OpenF1 Real Data",
                "compounds_stats": {
                    c: {
                        "cliff_lap": info.get("cliff_lap"),
                        "threshold_s": info.get("threshold_used"),
                        "n_train": info.get("n_train"),
                    }
                    for c, info in degr_metadata.items()
                },
            },
            "sc_impact_model": {
                "type": "GradientBoostingRegressor",
                "trained_on": "OpenF1 Pit Stop Data",
                "status": "Modello ML" if not self.sc_model.fallback else "Fallback Analitico",
                "fallback_active": self.sc_model.fallback,
            },
        }


# ── StrategyValidator ─────────────────────────────────────────────────────────
class StrategyValidator:
    """
    Valida e assegna un punteggio a una strategy.json esterna.

    PRIMA del calcolo ML vengono applicati i constraint fisici F1:
    violazioni gravi (slick su pioggia pesante, wet su asciutto, ecc.)
    abbattono il punteggio in modo proporzionale alla gravità,
    indipendentemente dal tempo stimato dal modello.
    """

    W_TIME = 0.50
    W_DEGR = 0.25
    W_SC = 0.20
    W_FORMAT = 0.05

    # Compound considerati "slick" (non adatti alla pioggia)
    SLICK_COMPOUNDS = {"soft", "medium", "hard"}
    # Compound adatti alla pioggia leggera
    WET_COMPOUNDS = {"intermediate", "wet"}

    def __init__(self, evaluator: StrategyEvaluator):
        self.evaluator = evaluator

    # ── Constraint fisici ──────────────────────────────────────────────────────
    def _check_weather_constraints(self, strategy: list, conditions: dict) -> tuple[list[str], list[str], float]:
        """
        Verifica che i compound siano compatibili con le condizioni meteo
        per ogni giro della strategia.

        Regole F1:
          - Pioggia PESANTE + slick  → ILLEGALE / pericoloso (penalità -60)
          - Pioggia LEGGERA + slick  → subottimale ma legale  (penalità -20)
          - Asciutto + wet           → inutilizzabili          (penalità -30)
          - Asciutto + intermediate  → molto lenti             (penalità -15)
          - Pioggia PESANTE senza wet → fortemente sconsigliato (penalità -40)

        Ritorna (issues, warnings, weather_penalty).
        """
        issues = []
        warnings = []
        penalty = 0.0

        rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
        rain_intensity = conditions.get("weather", {}).get("rain_intensity", "none")
        total_laps = conditions.get("total_laps", 53)

        sorted_strat = sorted(strategy, key=lambda x: x.get("start_lap", 0))

        for i, s in enumerate(sorted_strat):
            compound = s.get("compound", "").lower()
            start_lap = s.get("start_lap", 1)
            end_lap = sorted_strat[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_strat) else total_laps

            # Giri di questo stint in fase di pioggia
            rain_laps_in_stint = max(0, end_lap - max(start_lap, rain_start) + 1) if rain_start <= end_lap else 0
            dry_laps_in_stint = max(0, min(end_lap, rain_start - 1) - start_lap + 1)
            total_stint = end_lap - start_lap + 1

            # ── Caso 1: Slick su pioggia pesante ──────────────────────────
            if rain_intensity == "heavy" and compound in self.SLICK_COMPOUNDS and rain_laps_in_stint > 0:
                msg = (
                    f"Stint {i + 1} ({compound.upper()}, G.{start_lap}–{end_lap}): "
                    f"GOMME SLICK CON PIOGGIA PESANTE per {rain_laps_in_stint} giri — "
                    f"pericoloso e illegale. Necessarie WET."
                )
                issues.append(msg)
                penalty += 60.0

            # ── Caso 2: Slick su pioggia leggera ──────────────────────────
            elif rain_intensity in ("light", "moderate") and compound in self.SLICK_COMPOUNDS and rain_laps_in_stint > 0:
                # Non illegale ma fortemente subottimale
                frac = rain_laps_in_stint / max(total_stint, 1)
                p = 20.0 * frac
                penalty += p
                warnings.append(
                    f"Stint {i + 1} ({compound.upper()}): {rain_laps_in_stint} giri su pioggia leggera "
                    f"con slick — intermedie sarebbero più rapide (~3-5s/giro più lente)"
                )

            # ── Caso 3: Pioggia pesante senza wet ─────────────────────────
            elif rain_intensity == "heavy" and compound == "intermediate" and rain_laps_in_stint > 0:
                frac = rain_laps_in_stint / max(total_stint, 1)
                p = 40.0 * frac
                penalty += p
                warnings.append(
                    f"Stint {i + 1} (INTERMEDIATE): {rain_laps_in_stint} giri con pioggia PESANTE — "
                    f"le WET sarebbero obbligatorie/molto più veloci"
                )

            # ── Caso 4: Wet su asciutto ────────────────────────────────────
            elif compound == "wet" and dry_laps_in_stint > 0:
                frac = dry_laps_in_stint / max(total_stint, 1)
                p = 30.0 * frac
                penalty += p
                warnings.append(
                    f"Stint {i + 1} (WET): {dry_laps_in_stint} giri su asciutto con gomme da pioggia — "
                    f"inutilizzabili su asciutto (~8s/giro più lente)"
                )

            # ── Caso 5: Intermediate su asciutto ──────────────────────────
            elif compound == "intermediate" and dry_laps_in_stint > 0:
                frac = dry_laps_in_stint / max(total_stint, 1)
                p = 15.0 * frac
                penalty += p
                warnings.append(
                    f"Stint {i + 1} (INTERMEDIATE): {dry_laps_in_stint} giri su asciutto — ~2s/giro più lento degli slick"
                )

        return issues, warnings, penalty

    def _check_rain_coverage(self, strategy: list, conditions: dict) -> tuple[list[str], float]:
        """
        Verifica che la strategia preveda un compound da pioggia
        per tutta la fase di pioggia dichiarata nelle condizioni.
        Se la pioggia è prevista e nessuno stint la copre con wet/intermediate,
        è un errore strategico grave.
        """
        rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
        rain_intensity = conditions.get("weather", {}).get("rain_intensity", "none")
        total_laps = conditions.get("total_laps", 53)

        if rain_intensity == "none" or rain_start >= total_laps:
            return [], 0.0

        # Cerca se esiste almeno uno stint wet/intermediate che copre il rain_start
        sorted_strat = sorted(strategy, key=lambda x: x.get("start_lap", 0))
        rain_covered = False
        for i, s in enumerate(sorted_strat):
            compound = s.get("compound", "").lower()
            start_lap = s.get("start_lap", 1)
            end_lap = sorted_strat[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_strat) else total_laps
            if compound in self.WET_COMPOUNDS and start_lap <= rain_start + 3 and end_lap >= rain_start:
                rain_covered = True
                break

        if not rain_covered:
            severity = "PESANTE" if rain_intensity == "heavy" else "LEGGERA"
            msg = (
                f"Nessun compound da pioggia (intermediate/wet) previsto "
                f"per la fase di pioggia {severity} (dal giro {rain_start})"
            )
            penalty = 35.0 if rain_intensity == "heavy" else 15.0
            return [msg], penalty

        return [], 0.0

    # ── validate ──────────────────────────────────────────────────────────────
    def validate(self, strategy_json: dict, conditions: dict, reference_time: float | None = None) -> dict:
        total_laps = conditions.get("total_laps", 53)
        strategy = strategy_json.get("strategy", [])
        team_name = strategy_json.get("team_name", "Unknown")

        issues = []
        warnings = []

        # ── 1. Validazione formato ─────────────────────────────────────────
        format_score = 100.0
        valid_compounds = {"soft", "medium", "hard", "intermediate", "wet"}

        if not strategy:
            issues.append("strategy vuota")
            format_score = 0.0
        else:
            if strategy[0].get("start_lap") != 1:
                issues.append("Il primo stint deve iniziare al giro 1")
                format_score -= 30

            sorted_strat = sorted(strategy, key=lambda x: x.get("start_lap", 0))
            for i, s in enumerate(sorted_strat):
                c = s.get("compound", "").lower()
                if c not in valid_compounds:
                    issues.append(f"Stint {i + 1}: compound non valido '{c}'")
                    format_score -= 20
                if "start_lap" not in s:
                    issues.append(f"Stint {i + 1}: manca start_lap")
                    format_score -= 20
                if i > 0 and s.get("start_lap", 0) <= sorted_strat[i - 1].get("start_lap", 0):
                    issues.append(f"Stint {i + 1}: start_lap non monotono crescente")
                    format_score -= 20

            if sorted_strat[-1].get("start_lap", 1) > total_laps:
                issues.append(f"L'ultimo stint inizia oltre il giro {total_laps}")
                format_score -= 20

        format_score = max(0.0, format_score)

        # ── 2. Constraint fisici meteo ─────────────────────────────────────
        # Questi vengono valutati SEMPRE, anche se il formato è invalido,
        # perché sono constraint di sicurezza indipendenti.
        weather_issues, weather_warnings, weather_penalty = [], [], 0.0
        rain_issues, rain_penalty = [], 0.0

        if strategy:
            weather_issues, weather_warnings, weather_penalty = self._check_weather_constraints(strategy, conditions)
            rain_issues, rain_penalty = self._check_rain_coverage(strategy, conditions)

        # Issues meteo gravi (slick + pioggia pesante) bloccano la validità
        issues += weather_issues
        warnings += weather_warnings
        issues += rain_issues

        # ── 3. Stima tempo totale con ML ───────────────────────────────────
        ml_result = None
        total_time = None
        time_score = 50.0

        if not [i for i in issues if "ILLEGALE" in i or "SLICK CON PIOGGIA PESANTE" in i]:
            # Calcola il tempo ML solo se non ci sono violazioni bloccanti
            try:
                ml_result = self.evaluator.evaluate_strategy(strategy, conditions)
                total_time = ml_result["total_time"]

                if reference_time is not None:
                    delta_seconds = total_time - reference_time
                    # Invece di 0.5 (molto severo), usa un divisore più morbido
                    # o una percentuale (es. 100 punti se entro il 2% del tempo reference)
                    time_score = max(0.0, 100.0 - (delta_seconds / 5.0))
                else:
                    time_score = max(0.0, 100.0 - max(0.0, total_time - 4500) * 0.1)

                # Applica penalità meteo al time_score (non al tempo stimato,
                # che il modello ML potrebbe non catturare correttamente
                # per condizioni mai viste nel training)
                time_score = max(0.0, time_score - weather_penalty - rain_penalty)

            except Exception as e:
                warnings.append(f"Errore stima tempo ML: {e}")
                time_score = max(0.0, 50.0 - weather_penalty - rain_penalty)
        else:
            # Strategia con violazione grave: time_score minimo
            time_score = max(0.0, 10.0 - weather_penalty)

        # ── 4. Rischio degrado ─────────────────────────────────────────────
        degradation_report = []
        degr_score = 100.0

        if strategy and not [i for i in issues if "start_lap" in i or "vuota" in i]:
            sorted_strat = sorted(strategy, key=lambda x: x.get("start_lap", 0))
            for i, s in enumerate(sorted_strat):
                compound = s.get("compound", "medium").lower()
                start_lap = s.get("start_lap", 1)
                end_lap = sorted_strat[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_strat) else total_laps
                stint_len = end_lap - start_lap + 1

                risk = self.evaluator.deg_model.assess_stint_risk(compound, stint_len)
                degradation_report.append(risk)

                cliff_lap = risk["cliff_lap"]
                extra_laps = max(0, stint_len - cliff_lap)

                if risk["risk_level"] == "HIGH":
                    degr_score -= 25
                    warnings.append(
                        f"Stint {i + 1} ({compound.upper()}): Raggiunto limite fisico della gomma al giro {cliff_lap}. "
                        f"Percorsi {extra_laps} giri in over-stretch con una perdita media di +{risk['penalty_seconds']:.1f}s/giro."
                    )
                elif risk["risk_level"] == "MEDIUM":
                    degr_score -= 10
                    # Qui usiamo stint_len/cliff_lap per mostrare quanto siamo al limite
                    warnings.append(
                        f"Stint {i + 1} ({compound.upper()}): Stint al limite ({stint_len}/{cliff_lap} giri). "
                        "Rischio crollo prestazioni elevato per i giri finali."
                    )

        degr_score = max(0.0, degr_score)

        # ── 5. Qualità timing SC ───────────────────────────────────────────
        sc_report = {}
        sc_score = 50.0

        if conditions.get("safety_car", {}).get("active") and strategy:
            sc_report = self.evaluator.sc_model.evaluate_strategy_sc_timing(strategy, conditions)
            saving = sc_report.get("total_sc_saving_seconds", 0)
            sc_score = min(100.0, 50.0 + saving * 2)

            for pit in sc_report.get("pit_stops", []):
                if pit["timing_quality"] == "NEUTRAL":
                    sc_lap = conditions.get("safety_car", {}).get("lap")
                    dist = abs(pit["pit_lap"] - sc_lap) if sc_lap else None
                    if dist is not None and dist <= 8:
                        warnings.append(
                            f"Pit G.{pit['pit_lap']}: SC al G.{sc_lap} — distanza {dist} giri, opportunità non sfruttata"
                        )

        # ── 6. Punteggio composito ─────────────────────────────────────────
        composite_score = (
            self.W_TIME * time_score + self.W_DEGR * degr_score + self.W_SC * sc_score + self.W_FORMAT * format_score
        )

        return {
            "team_name": team_name,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "scores": {
                "composite": round(composite_score, 1),
                "time": round(time_score, 1),
                "degradation": round(degr_score, 1),
                "sc_timing": round(sc_score, 1),
                "format": round(format_score, 1),
            },
            "estimated_total_time": total_time,
            "ml_lap_detail": ml_result,
            "degradation_report": degradation_report,
            "sc_report": sc_report,
        }

    def rank_strategies(self, strategies: list[dict], conditions: dict) -> list[dict]:
        """
        Valida e classifica più strategy.json.
        Il reference_time è il miglior tempo stimato tra le strategie VALIDE
        (senza violazioni fisiche gravi), così le strategie illegali non
        abbassano artificialmente il reference.
        """
        # Prima passata: stima i tempi delle sole strategie valide per il reference
        valid_times = []
        for strat in strategies:
            if not strat.get("strategy"):
                continue
            # Check rapido violazioni gravi prima di chiamare ML
            rain_intensity = conditions.get("weather", {}).get("rain_intensity", "none")
            rain_start = conditions.get("weather", {}).get("rain_start_lap", 999)
            total_laps = conditions.get("total_laps", 53)
            has_critical = False
            if rain_intensity == "heavy":
                sorted_s = sorted(strat["strategy"], key=lambda x: x.get("start_lap", 0))
                for i, s in enumerate(sorted_s):
                    end_lap = sorted_s[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_s) else total_laps
                    rain_laps = max(0, end_lap - max(s.get("start_lap", 1), rain_start) + 1) if rain_start <= end_lap else 0
                    if s.get("compound", "").lower() in self.SLICK_COMPOUNDS and rain_laps > 0:
                        has_critical = True
                        break
            if not has_critical:
                try:
                    r = self.evaluator.evaluate_strategy(strat["strategy"], conditions)
                    valid_times.append(r["total_time"])
                except Exception:
                    pass

        reference_time = min(valid_times) if valid_times else None

        # Seconda passata: validazione completa
        results = []
        for strat in strategies:
            report = self.validate(strat, conditions, reference_time=reference_time)
            results.append(report)

        results.sort(key=lambda x: (x["valid"], x["scores"]["composite"]), reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results
