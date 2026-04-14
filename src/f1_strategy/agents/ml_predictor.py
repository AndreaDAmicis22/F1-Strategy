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

    def predict_delta(
        self,
        compound: str,
        stint_lap: int,
        track_temp: float = 38.0,
        air_temp: float = 25.0,
        speed_mean: float = 210.0,
        lap_number: int = 27,
        total_session_laps: int = 55,
        weather: int = 0,
        **kwargs,
    ) -> float:
        """
        Calcola i secondi extra rispetto al best pace dovuti al degrado.
        """
        compound = compound.lower()
        if compound not in self.models:
            return 0.0

        features = self.metadata[compound].get("features", ["stint_lap"])

        # Dizionario per mappare i valori correnti alle feature attive nel modello
        data_map = {
            "stint_lap": stint_lap,
            "track_temp": track_temp,
            "air_temp": air_temp,
            "speed_mean": speed_mean,
            "lap_number": lap_number,
            "session_progression": lap_number / total_session_laps,
            "weather_enc": weather,
            "avg_energy_session": kwargs.get("avg_energy_session", (speed_mean * track_temp)),
        }

        # Costruiamo la riga di input nell'ordine corretto salvato nei metadati
        row = [data_map.get(f, 0.0) for f in features]

        X = np.array([row], dtype=np.float32)
        # Il degrado non può essere negativo (la gomma non rigenera tempo)
        return max(0.0, float(self.models[compound].predict(X)[0]))

    def assess_stint_risk(self, compound: str, stint_length: int, track_temp: float = 38.0, **kwargs) -> dict:
        """
        Valuta il rischio degrado di uno stint completo.
        """
        cliff = self.get_cliff(compound)

        # Somma del degrado giro per giro
        total_deg = sum(
            self.predict_delta(compound, lap, track_temp=track_temp, **kwargs) for lap in range(1, stint_length + 1)
        )

        exceeds = stint_length > cliff
        penalty = 0.0

        if exceeds:
            # Calcolo della penalità: media del degrado nei giri oltre il cliff
            extra_laps = stint_length - cliff
            penalty = sum(
                self.predict_delta(compound, cliff + l, track_temp=track_temp, **kwargs) for l in range(1, extra_laps + 1)
            ) / max(extra_laps, 1)

        return {
            "compound": compound,
            "stint_length": stint_length,
            "cliff_lap": cliff,
            "total_degradation_seconds": round(total_deg, 2),
            "exceeds_cliff": exceeds,
            "penalty_seconds": round(penalty, 2),
            "risk_level": "HIGH" if exceeds and penalty > 4 else ("MEDIUM" if exceeds else "LOW"),
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
        """
        Valuta il guadagno tattico dei pit stop pianificati in caso di SC.
        """
        sc_active = conditions.get("safety_car", {}).get("active", False)
        sc_lap = conditions.get("safety_car", {}).get("lap") if sc_active else None
        pit_lane_base = conditions.get("pit_lane_time_loss_seconds", NORMAL_PIT_COST_SECONDS)

        # Parametri dinamici per la SC
        sc_params = {
            "sc_speed_ratio": conditions.get("safety_car", {}).get("speed_ratio", 0.4 if sc_active else 1.0),
            "rainfall": conditions.get("weather", {}).get("rainfall", 0.0),
            "sc_duration_step": conditions.get("safety_car", {}).get("duration_step", 0),
        }

        pit_stops = []
        for stint in strategy[1:]:
            pit_lap = stint["start_lap"]
            cost = self.predict_pit_cost(pit_lap, sc_lap, stint["compound"], pit_lane_seconds=pit_lane_base, **sc_params)
            saving = max(0.0, pit_lane_base - cost)

            pit_stops.append(
                {
                    "pit_lap": pit_lap,
                    "estimated_cost": round(cost, 2),
                    "sc_saving": round(saving, 2),
                    "timing_quality": "EXCELLENT" if saving > 12 else "GOOD" if saving > 5 else "NEUTRAL",
                }
            )

        return {
            "pit_stops": pit_stops,
            "total_sc_saving_seconds": round(sum(p["sc_saving"] for p in pit_stops), 2),
            "sc_active": sc_active,
        }


# ── StrategyEvaluator ─────────────────────────────────────────────────────────


class StrategyEvaluator:
    """
    Valuta una strategia giro per giro usando LapTimePredictor.
    Usato internamente dal StrategyValidator per calcolare il tempo totale.
    """

    def __init__(self):
        self.lap_model = LapTimePredictor()
        self.deg_model = DegradationRiskModel()
        self.sc_model = SafetyCarImpactModel()

    @classmethod
    def load(cls) -> StrategyEvaluator:
        return cls()

    def evaluate_strategy(self, strategy: list, conditions: dict, track_temp: float = 38.0, air_temp: float = 25.0) -> dict:
        """Simula giro per giro con ML e ritorna tempo totale + lap results."""
        total_laps = conditions.get("total_laps", 53)
        pit_lane_base = conditions.get("pit_lane_time_loss_seconds", NORMAL_PIT_COST_SECONDS)

        # Info Safety Car
        sc_info = conditions.get("safety_car", {})
        sc_active = sc_info.get("active", False)
        sc_lap_trigger = sc_info.get("lap", -1)
        sc_duration = sc_info.get("duration_laps", 0)
        sc_speed_ratio = sc_info.get("speed_ratio", 0.4)  # Intensità neutralizzazione

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

        lap_to_stint = {}
        for st in stints_ext:
            for lap in range(st["start_lap"], st["end_lap"] + 1):
                lap_to_stint[lap] = st

        # --- Variabili di Stato della Simulazione ---
        total_time = 0.0
        lap_results = []
        pit_stops = 0
        current_prev_lap_time = 92.0  # Valore di fallback per il giro 1
        sc_current_step = 0

        for lap in range(1, total_laps + 1):
            stint = lap_to_stint.get(lap)
            if not stint:
                continue

            compound = stint["compound"]
            stint_lap = lap - stint["start_lap"]
            weather = ("heavy_rain" if rain_intens == "heavy" else "light_rain") if lap >= rain_start else "dry"

            # Gestione Safety Car
            is_sc = sc_active and sc_lap_trigger <= lap < sc_lap_trigger + sc_duration
            if is_sc:
                sc_current_step += 1
            else:
                sc_current_step = 0

            # 1. Calcolo del PIT COST (se è un giro di sosta)
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
                    sc_speed_ratio=sc_speed_ratio if is_sc else 1.0,
                    sc_duration_step=sc_current_step,
                    rainfall=rainfall if lap >= rain_start else 0.0,
                )
                total_time += pit_cost
                pit_stops += 1

            # 2. Predizione LAP TIME
            if is_sc:
                # Sotto SC il tempo è dettato dalla direzione gara (es. 140% del tempo normale)
                lap_time = current_prev_lap_time / sc_speed_ratio if sc_speed_ratio > 0 else 140.0
            else:
                lap_time = self.lap_model.predict(
                    compound=compound,
                    stint_lap=stint_lap,
                    weather=weather,
                    track_temp=track_temp,
                    air_temp=air_temp,
                    lap_number=lap,
                    max_session_laps=total_laps,
                    prev_lap_duration=current_prev_lap_time,
                    humidity=conditions.get("weather", {}).get("humidity", 50.0),
                )

            # Aggiornamento stato per il giro successivo
            current_prev_lap_time = lap_time
            total_time += lap_time

            lap_results.append(
                {
                    "lap": lap,
                    "lap_time": round(lap_time, 3),
                    "compound": compound,
                    "stint_lap": stint_lap,
                    "weather": weather,
                    "is_pit_lap": is_pit,
                    "is_sc_lap": is_sc,
                    "pit_cost": round(pit_cost, 2) if is_pit else 0.0,
                    "cumulative_time": round(total_time, 3),
                }
            )

        return {
            "total_time": round(total_time, 3),
            "pit_stops": pit_stops,
            "lap_results": lap_results,
            "strategy_summary": stints_ext,
        }

    def get_models_info(self) -> dict:
        metrics = self.lap_model.evaluate()
        return {
            "lap_time_predictor": {
                "type": "GradientBoostingRegressor",
                "trained_on": "dati reali OpenF1",
                "metrics": metrics,
                "feature_importance": self.lap_model.feature_importance(),
            },
            "degradation_risk_model": {
                "type": "Ridge(poly=2) per compound",
                "trained_on": "dati reali OpenF1",
                "cliffs": {c: self.deg_model.get_cliff(c) for c in self.deg_model.models},
            },
            "sc_impact_model": {
                "type": "GradientBoostingRegressor",
                "trained_on": "pit stop reali OpenF1",
                "fallback": self.sc_model.fallback,
            },
        }


# ── StrategyValidator ─────────────────────────────────────────────────────────


class StrategyValidator:
    """
    Valida e assegna un punteggio a una strategy.json esterna.
    Produce un report strutturato con:
      - tempo totale stimato (ML)
      - rischio degrado per ogni stint
      - qualità timing pit stop rispetto a SC
      - punteggio composito (0–100)
      - lista warning/issues
    """

    # Pesi del punteggio composito
    W_TIME = 0.50  # 50% — tempo totale (quanto è veloce)
    W_DEGR = 0.25  # 25% — rischio degrado (stint troppo lunghi)
    W_SC = 0.15  # 15% — sincronizzazione SC
    W_FORMAT = 0.10  # 10% — correttezza formato

    def __init__(self, evaluator: StrategyEvaluator):
        self.evaluator = evaluator

    def validate(self, strategy_json: dict, conditions: dict, reference_time: float | None = None) -> dict:
        """
        Valida una strategy.json e ritorna il report completo.

        Args:
            strategy_json:  dict con "team_name", "strategy", "rationale"
            conditions:     race_conditions.json
            reference_time: tempo di riferimento per il punteggio relativo
                            (es. miglior tempo tra tutti i team).
                            Se None usa il tempo stimato come riferimento.
        """
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
                if i > 0:
                    prev_start = sorted_strat[i - 1].get("start_lap", 0)
                    curr_start = s.get("start_lap", 0)
                    if curr_start <= prev_start:
                        issues.append(f"Stint {i + 1}: start_lap non monotono crescente")
                        format_score -= 20

            # Copertura giri
            last_start = sorted_strat[-1].get("start_lap", 1)
            if last_start > total_laps:
                issues.append(f"L'ultimo stint inizia oltre il giro {total_laps}")
                format_score -= 20

        format_score = max(0.0, format_score)

        # ── 2. Stima tempo totale con ML ───────────────────────────────────
        ml_result = None
        time_score = 50.0  # default neutro se non calcolabile

        if not issues:  # solo se il formato è valido
            try:
                ml_result = self.evaluator.evaluate_strategy(strategy, conditions)
                total_time = ml_result["total_time"]

                if reference_time is not None and reference_time > 0:
                    # Punteggio relativo: 100 se uguale al reference, -1pt ogni secondo in più
                    delta = total_time - reference_time
                    time_score = max(0.0, 100.0 - delta * 0.5)
                else:
                    # Senza reference: punteggio basato su range atteso
                    # Monza 53 giri: range tipico 4500–4800s
                    time_score = max(0.0, 100.0 - max(0.0, total_time - 4500) * 0.1)
            except Exception as e:
                warnings.append(f"Errore stima tempo ML: {e}")
                total_time = None
                time_score = 0.0
        else:
            total_time = None

        # ── 3. Rischio degrado ─────────────────────────────────────────────
        degradation_report = []
        degr_score = 100.0

        if not issues and strategy:
            sorted_strat = sorted(strategy, key=lambda x: x.get("start_lap", 0))
            for i, s in enumerate(sorted_strat):
                compound = s.get("compound", "medium").lower()
                start_lap = s.get("start_lap", 1)
                end_lap = sorted_strat[i + 1]["start_lap"] - 1 if i + 1 < len(sorted_strat) else total_laps
                stint_len = end_lap - start_lap + 1

                risk = self.evaluator.deg_model.assess_stint_risk(compound, stint_len)
                degradation_report.append(risk)

                if risk["risk_level"] == "HIGH":
                    degr_score -= 25
                    warnings.append(
                        f"Stint {i + 1} ({compound.upper()}): {stint_len} giri supera "
                        f"il cliff di {risk['cliff_lap']} giri "
                        f"(+{risk['penalty_seconds']:.1f}s di degrado extra)"
                    )
                elif risk["risk_level"] == "MEDIUM":
                    degr_score -= 10
                    warnings.append(
                        f"Stint {i + 1} ({compound.upper()}): {stint_len} giri vicino al cliff ({risk['cliff_lap']} giri)"
                    )

        degr_score = max(0.0, degr_score)

        # ── 4. Qualità timing SC ───────────────────────────────────────────
        sc_report = {}
        sc_score = 50.0  # neutro se non c'è SC

        if conditions.get("safety_car", {}).get("active") and not issues:
            sc_report = self.evaluator.sc_model.evaluate_strategy_sc_timing(strategy, conditions)
            saving = sc_report.get("total_sc_saving_seconds", 0)
            # Ogni secondo guadagnato dalla SC vale +2 punti, max 100
            sc_score = min(100.0, 50.0 + saving * 2)

            for pit in sc_report.get("pit_stops", []):
                if pit["timing_quality"] == "EXCELLENT":
                    pass  # già reflesso nel saving
                elif pit["timing_quality"] == "NEUTRAL" and pit.get("sc_distance") is not None:
                    if pit["sc_distance"] < 10:
                        warnings.append(
                            f"Pit G.{pit['pit_lap']}: SC al G.{sc_report['sc_lap']}, "
                            f"distanza {pit['sc_distance']} giri — opportunità persa"
                        )

        # ── 5. Punteggio composito ─────────────────────────────────────────
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
        Input: lista di dict {"team_name": ..., "strategy": [...], ...}
        Output: lista ordinata per punteggio composito (decrescente).
        """
        # Prima passata: stima tutti i tempi per calcolare il reference
        times = []
        for strat in strategies:
            if strat.get("strategy"):
                try:
                    r = self.evaluator.evaluate_strategy(strat["strategy"], conditions)
                    times.append(r["total_time"])
                except Exception:
                    pass

        reference_time = min(times) if times else None

        # Seconda passata: valutazione completa con reference
        results = []
        for strat in strategies:
            report = self.validate(strat, conditions, reference_time=reference_time)
            results.append(report)

        results.sort(key=lambda x: x["scores"]["composite"], reverse=True)

        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results
