import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    name: str
    strategy: list[dict]
    total_time: float
    pit_stops: int
    rationale: str
    warnings: list[str] = field(default_factory=list)
    lap_results: list[dict] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


class StrategyArchitect:
    def __init__(self, conditions: dict):
        self.conditions = conditions
        self.total_laps = conditions.get("total_laps", 53)
        self.rain_lap = conditions.get("weather", {}).get("rain_start_lap", 999)
        self.rain_intensity = conditions.get("weather", {}).get("rain_intensity", "none")
        self.sc_lap = (
            conditions.get("safety_car", {}).get("lap", -1) if conditions.get("safety_car", {}).get("active") else -1
        )

    def propose_strategies(self) -> list[dict]:
        scenarios = []
        # Strategia base
        scenarios.append(
            {
                "name": "Standard 1-Stop (M-H)",
                "rationale": "Strategia bilanciata per Monza.",
                "strategy": [
                    {"stint": 1, "compound": "medium", "start_lap": 1},
                    {"stint": 2, "compound": "hard", "start_lap": 26},
                ],
            }
        )
        # Se c'è pioggia pesante
        if self.rain_intensity == "heavy":
            scenarios.append(
                {
                    "name": "Full Wet Safety",
                    "rationale": "Massima protezione sotto pioggia pesante.",
                    "strategy": [
                        {"stint": 1, "compound": "medium", "start_lap": 1},
                        {"stint": 2, "compound": "wet", "start_lap": self.rain_lap},
                    ],
                }
            )
        elif self.rain_lap < self.total_laps:
            scenarios.append(
                {
                    "name": "Intermediate Pivot",
                    "rationale": "Switch su intermedie all'inizio della pioggia.",
                    "strategy": [
                        {"stint": 1, "compound": "medium", "start_lap": 1},
                        {"stint": 2, "compound": "intermediate", "start_lap": self.rain_lap},
                    ],
                }
            )
        return scenarios


# QUESTA FUNZIONE DEVE ESSERE A BORDO RIGA
def find_optimal_strategy(conditions: dict, historical_patterns: dict | None = None, ml_evaluator=None) -> dict:
    if ml_evaluator is None:
        msg = "ml_evaluator non può essere None"
        raise RuntimeError(msg)

    architect = StrategyArchitect(conditions)
    candidates = architect.propose_strategies()

    results = []
    for cand in candidates:
        try:
            evaluated = ml_evaluator.evaluate_strategy(cand["strategy"], conditions)

            # Applichiamo i guardrail
            final_time = evaluated["total_time"]
            local_warnings = []

            if conditions.get("weather", {}).get("rain_intensity") == "heavy":
                for lap in evaluated["lap_results"]:
                    if lap["weather"] == "heavy_rain" and lap["compound"] in ["soft", "medium", "hard"]:
                        final_time += 40.0
                        local_warnings.append("PERICOLO: Slick su pioggia pesante!")

            res = SimulationResult(
                name=cand["name"],
                strategy=cand["strategy"],
                total_time=final_time,
                pit_stops=evaluated["pit_stops"],
                rationale=cand["rationale"],
                warnings=local_warnings,
                lap_results=evaluated["lap_results"],
                breakdown=evaluated["breakdown"],
            )
            results.append(res)
        except Exception as e:
            logger.exception(f"Errore nello scenario {cand['name']}: {e}")

    if not results:
        return {"optimal": {"name": "Errore", "total_time": 0}, "ranking": []}

    results.sort(key=lambda x: x.total_time)

    return {
        "optimal": vars(results[0]),
        "ranking": [
            {
                "rank": i + 1,
                "name": r.name,
                "total_time": r.total_time,
                "delta": round(r.total_time - results[0].total_time, 2),
                "rationale": r.rationale,
            }
            for i, r in enumerate(results)
        ],
    }
