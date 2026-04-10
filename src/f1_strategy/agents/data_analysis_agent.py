"""
Agent 1: Analisi Dati Storici
Analizza dati reali OpenF1 per estrarre pattern strategici da Monza
"""

import json
import logging
import statistics

import openf1_client as api

logger = logging.getLogger(__name__)

# Circuit key Monza sull'API OpenF1
MONZA_CIRCUIT_KEY = "monza"


def find_monza_sessions(years: list | None = None) -> list:
    """Trova le sessioni Race di Monza negli anni specificati."""
    if years is None:
        years = [2023, 2024]
    sessions = []
    for year in years:
        all_sessions = api.fetch("sessions", {"year": year, "session_type": "Race"})
        for s in all_sessions:
            name = (s.get("circuit_short_name") or "").lower()
            location = (s.get("location") or "").lower()
            if "monza" in name or "monza" in location or "italy" in location.lower():
                sessions.append(s)
                logger.info(f"Trovata sessione Monza {year}: session_key={s.get('session_key')}")
    return sessions


def analyze_stints(session_key: int) -> dict:
    """Analizza stint e compound per una sessione."""
    stints = api.get_stints(session_key)
    if not stints:
        return {}

    compound_stats = {}
    for stint in stints:
        compound = (stint.get("compound") or "UNKNOWN").lower()
        lap_start = stint.get("lap_start", 0)
        lap_end = stint.get("lap_end", 0)
        length = max(0, (lap_end or 0) - (lap_start or 0))

        if compound not in compound_stats:
            compound_stats[compound] = {"count": 0, "lengths": [], "positions": []}

        compound_stats[compound]["count"] += 1
        if length > 0:
            compound_stats[compound]["lengths"].append(length)

    result = {}
    for compound, stats in compound_stats.items():
        avg_len = statistics.mean(stats["lengths"]) if stats["lengths"] else 0
        result[compound] = {
            "count": stats["count"],
            "avg_stint_length": round(avg_len, 1),
            "max_stint_length": max(stats["lengths"], default=0),
            "min_stint_length": min(stats["lengths"], default=0),
        }
    return result


def analyze_pit_stops(session_key: int) -> dict:
    """Analizza pit stop: timing, numero stop per driver."""
    pits = api.get_pit_stops(session_key)
    if not pits:
        return {"avg_pit_duration": 22.5, "stops_distribution": {}}

    # Raggruppa per driver
    driver_stops = {}
    pit_durations = []
    for p in pits:
        drv = p.get("driver_number")
        dur = p.get("pit_duration")
        lap = p.get("lap_number")
        if dur and dur > 0:
            pit_durations.append(dur)
        if drv:
            if drv not in driver_stops:
                driver_stops[drv] = []
            driver_stops[drv].append({"lap": lap, "duration": dur})

    stops_count = [len(v) for v in driver_stops.values()]
    stop_dist = {}
    for c in stops_count:
        stop_dist[str(c)] = stop_dist.get(str(c), 0) + 1

    avg_dur = statistics.mean(pit_durations) if pit_durations else 22.5

    # Identifica giri comuni per pit stop
    pit_laps = [p.get("lap_number") for p in pits if p.get("lap_number")]
    lap_frequency = {}
    for l in pit_laps:
        lap_frequency[l] = lap_frequency.get(l, 0) + 1

    top_pit_laps = sorted(lap_frequency.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "avg_pit_duration": round(avg_dur, 2),
        "stops_distribution": stop_dist,
        "most_common_pit_laps": top_pit_laps,
        "total_pit_stops": len(pits),
    }


def analyze_lap_times(session_key: int) -> dict:
    """Analizza tempi sul giro per stimare degrado gomme."""
    laps = api.get_laps(session_key)
    if not laps:
        return {}

    # Filtra giri validi (no pit in lap, no safety car)
    valid_laps = [
        l
        for l in laps
        if l.get("lap_duration")
        and l.get("lap_duration", 0) > 60
        and not l.get("is_pit_out_lap")
        and l.get("lap_duration", 999) < 200  # Monza < 200s
    ]

    if not valid_laps:
        return {}

    durations = [l["lap_duration"] for l in valid_laps]

    # Stima degrado: confronta primi 10 vs ultimi 10 giri validi
    if len(durations) >= 20:
        early = statistics.mean(durations[:10])
        late = statistics.mean(durations[-10:])
        degradation_per_lap = max(0, (late - early) / max(len(durations) - 10, 1))
    else:
        degradation_per_lap = 0.05

    return {
        "avg_lap_time": round(statistics.mean(durations), 3),
        "best_lap_time": round(min(durations), 3),
        "worst_lap_time": round(max(durations), 3),
        "estimated_degradation_per_lap": round(degradation_per_lap, 4),
        "valid_laps_analyzed": len(valid_laps),
    }


def analyze_weather(session_key: int) -> dict:
    """Analizza condizioni meteo durante la gara."""
    weather_data = api.get_weather(session_key)
    if not weather_data:
        return {}

    rain_laps = [w for w in weather_data if w.get("rainfall", 0) > 0]
    rain_pct = len(rain_laps) / len(weather_data) * 100 if weather_data else 0

    temps = [w.get("air_temperature") for w in weather_data if w.get("air_temperature")]
    track_temps = [w.get("track_temperature") for w in weather_data if w.get("track_temperature")]

    return {
        "rain_percentage": round(rain_pct, 1),
        "had_rain": rain_pct > 5,
        "avg_air_temp": round(statistics.mean(temps), 1) if temps else None,
        "avg_track_temp": round(statistics.mean(track_temps), 1) if track_temps else None,
    }


def analyze_race_control(session_key: int) -> dict:
    """Analizza eventi race control: SC, VSC, bandiere."""
    events = api.get_race_control(session_key)
    if not events:
        return {}

    sc_events = [e for e in events if "SAFETY CAR" in (e.get("message") or "").upper()]
    vsc_events = [e for e in events if "VIRTUAL" in (e.get("message") or "").upper()]
    flags = [e for e in events if e.get("flag")]

    return {
        "safety_car_deployments": len(sc_events),
        "vsc_deployments": len(vsc_events),
        "safety_car_laps": [e.get("lap_number") for e in sc_events],
        "flag_events": len(flags),
    }


def identify_winning_patterns(sessions_data: list) -> dict:
    """Identifica pattern strategici vincenti dall'analisi multi-sessione."""
    patterns = {
        "most_used_strategy": "1-stop",
        "optimal_compounds": [],
        "typical_pit_window": [],
        "degradation_profile": "medium",
        "rain_impact": False,
        "notes": [],
    }

    # Analizza distribuzione stop
    total_one_stop = 0
    total_two_stop = 0
    all_pit_laps = []
    all_compounds = []

    for session in sessions_data:
        pits = session.get("pit_analysis", {})
        dist = pits.get("stops_distribution", {})
        total_one_stop += dist.get("1", 0)
        total_two_stop += dist.get("2", 0)

        for lap, _ in pits.get("most_common_pit_laps", []):
            all_pit_laps.append(lap)

        stints = session.get("stint_analysis", {})
        all_compounds.extend(stints.keys())

        if session.get("weather_analysis", {}).get("had_rain"):
            patterns["rain_impact"] = True

    if total_one_stop >= total_two_stop:
        patterns["most_used_strategy"] = "1-stop"
        patterns["notes"].append("Monza favorisce strategia 1-stop per basso degrado gomme")
    else:
        patterns["most_used_strategy"] = "2-stop"
        patterns["notes"].append("Strategia 2-stop risultata più competitiva nei dati storici")

    if all_pit_laps:
        all_pit_laps.sort()
        mid = len(all_pit_laps) // 2
        patterns["typical_pit_window"] = [all_pit_laps[max(0, mid - 3)], all_pit_laps[min(len(all_pit_laps) - 1, mid + 3)]]

    # Compound più usati a Monza (hard/medium per bassa usura)
    compound_freq = {}
    for c in all_compounds:
        compound_freq[c] = compound_freq.get(c, 0) + 1

    patterns["optimal_compounds"] = sorted(compound_freq, key=lambda x: compound_freq[x], reverse=True)[:3]

    if not patterns["optimal_compounds"]:
        patterns["optimal_compounds"] = ["medium", "hard", "soft"]
        patterns["notes"].append("Dati compound non disponibili, usando defaults Monza")

    return patterns


def run_analysis(circuit: str = "Monza") -> dict:
    """
    Entry point principale: analisi completa dei dati storici.
    Returns dizionario con tutti i risultati dell'analisi.
    """
    logger.info(f"=== Avvio analisi storica per {circuit} ===")

    sessions = find_monza_sessions(years=[2023, 2024])

    if not sessions:
        logger.warning("Nessuna sessione trovata, uso dati di fallback")
        sessions = []

    sessions_data = []

    for session in sessions[:3]:  # Max 3 sessioni per velocità
        sk = session.get("session_key")
        if not sk:
            continue

        logger.info(f"Analisi session_key={sk} ({session.get('date_start', 'N/A')})")

        session_result = {
            "session_key": sk,
            "year": session.get("year"),
            "date": session.get("date_start"),
            "stint_analysis": analyze_stints(sk),
            "pit_analysis": analyze_pit_stops(sk),
            "lap_analysis": analyze_lap_times(sk),
            "weather_analysis": analyze_weather(sk),
            "race_control": analyze_race_control(sk),
        }
        sessions_data.append(session_result)

    # Fallback con dati noti di Monza se API non restituisce dati
    if not sessions_data:
        logger.warning("Nessun dato API disponibile. Uso knowledge base Monza.")
        sessions_data = [
            {
                "session_key": "fallback",
                "year": 2024,
                "date": "2024-09-01",
                "stint_analysis": {
                    "medium": {"count": 15, "avg_stint_length": 28, "max_stint_length": 38, "min_stint_length": 18},
                    "hard": {"count": 18, "avg_stint_length": 32, "max_stint_length": 45, "min_stint_length": 22},
                    "soft": {"count": 8, "avg_stint_length": 18, "max_stint_length": 25, "min_stint_length": 12},
                },
                "pit_analysis": {
                    "avg_pit_duration": 22.5,
                    "stops_distribution": {"1": 14, "2": 6},
                    "most_common_pit_laps": [(27, 8), (29, 7), (31, 5), (25, 4), (33, 3)],
                    "total_pit_stops": 26,
                },
                "lap_analysis": {
                    "avg_lap_time": 84.2,
                    "best_lap_time": 83.1,
                    "worst_lap_time": 92.0,
                    "estimated_degradation_per_lap": 0.035,
                    "valid_laps_analyzed": 53,
                },
                "weather_analysis": {"rain_percentage": 0, "had_rain": False, "avg_air_temp": 26, "avg_track_temp": 42},
                "race_control": {"safety_car_deployments": 0, "vsc_deployments": 0, "safety_car_laps": [], "flag_events": 2},
            }
        ]

    patterns = identify_winning_patterns(sessions_data)

    result = {
        "circuit": circuit,
        "sessions_analyzed": len(sessions_data),
        "sessions_data": sessions_data,
        "winning_patterns": patterns,
        "key_insights": _generate_insights(sessions_data, patterns),
    }

    logger.info(f"Analisi completata: {len(sessions_data)} sessioni, pattern={patterns['most_used_strategy']}")
    return result


def _generate_insights(sessions_data: list, patterns: dict) -> list:
    """Genera insight leggibili dall'analisi."""
    insights = []

    insights.append(f"Strategia dominante a Monza: {patterns['most_used_strategy'].upper()}")

    if patterns.get("typical_pit_window"):
        w = patterns["typical_pit_window"]
        insights.append(f"Finestra pit stop ottimale: giro {w[0]}–{w[1]}")

    # Analisi degrado
    all_deg = [
        s.get("lap_analysis", {}).get("estimated_degradation_per_lap", 0) for s in sessions_data if s.get("lap_analysis")
    ]
    if all_deg:
        avg_deg = statistics.mean(all_deg)
        if avg_deg < 0.05:
            insights.append("Degrado gomme BASSO: favorisce stint lunghi su Hard/Medium")
        elif avg_deg < 0.1:
            insights.append("Degrado gomme MODERATO: strategia 2-stop competitiva")
        else:
            insights.append("Degrado gomme ALTO: considerare 2-stop")

    if patterns.get("rain_impact"):
        insights.append("Storico: pioggia ha impattato significativamente la strategia")

    insights.append("Monza: circuito ad alta velocità, basso carico aerodinamico, poche curve lente")
    insights.append("Sorpasso possibile su DRS: posizione griglia meno critica rispetto ad altri circuiti")

    return insights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_analysis("Monza")
    print(json.dumps(result, indent=2, default=str))
