"""Benchmark: the main evaluation plus ablations, summarised for docs/results.md."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from balltracker.evaluate import DeliveryResult, EvalConfig, evaluate_many

FIRST_SEED = 1000


def ablation_configs() -> list[EvalConfig]:
    return [
        EvalConfig(name="main"),
        EvalConfig(fps=120, name="120 fps"),
        EvalConfig(fps=60, name="60 fps"),
        EvalConfig(fps=30, name="30 fps"),
        EvalConfig(click_noise_px=0.0, name="true camera (no click error)"),
        EvalConfig(click_noise_px=3.0, name="3 px click error"),
        EvalConfig(fixed=(("drag", 0.0),), name="no drag in model"),
        EvalConfig(fixed=(("swing", 0.0), ("seam", 0.0)), name="no swing or seam in model"),
    ]


def run(deliveries: int, ablation_deliveries: int, workers: int) -> dict:
    out = {"machine": f"{platform.machine()} {platform.system()}, {workers} workers", "runs": {}}
    for config in ablation_configs():
        n = deliveries if config.name == "main" else ablation_deliveries
        seeds = list(range(FIRST_SEED, FIRST_SEED + n))
        started = time.perf_counter()
        results = evaluate_many(seeds, config, workers)
        out["runs"][config.name] = {
            "config": asdict(config),
            "seconds": round(time.perf_counter() - started, 1),
            "summary": summarise(results),
            "results": [asdict(r) for r in results],
        }
    main = out["runs"]["main"]
    main["by_ball"] = {
        ball: summarise([DeliveryResult(**r) for r in main["results"] if r["ball"] == ball])
        for ball in ("white", "red")
    }
    return out


def summarise(results: list[DeliveryResult]) -> dict:
    ok = [r for r in results if r.status == "ok"]
    calls = [r for r in ok if np.isfinite(r.stumps_error_cm)]
    return {
        "deliveries": len(results),
        "ok": len(ok),
        "status": {
            s: sum(r.status == s for r in results) for s in sorted({r.status for r in results})
        },
        "detection_recall": _ratio(
            sum(r.detected_frames for r in results), sum(r.visible_frames for r in results)
        ),
        "track_error_px": _q([r.track_error_px for r in ok]),
        "calibration_rms_px": _q([r.calibration_rms_px for r in results]),
        "fit_rms_px": _q([r.fit_rms_px for r in ok]),
        "rmse_3d_cm": _q([r.rmse_3d_cm for r in ok]),
        "speed_abs_error_kmh": _q([abs(r.speed_error_kmh) for r in ok]),
        "bounce_error_cm": _q([r.bounce_error_cm for r in ok]),
        "stumps_error_cm": _q([r.stumps_error_cm for r in calls]),
        "hit_miss_accuracy": _ratio(
            sum(r.hitting_true == r.hitting_predicted for r in calls), len(calls)
        ),
        "hitting_deliveries": sum(r.hitting_true for r in calls),
        "drag_abs_error": _q([abs(r.drag_error) for r in ok]),
        "restitution_abs_error": _q([abs(r.restitution_error) for r in ok]),
    }


def worst_cases(run_data: dict, count: int = 8) -> list[dict]:
    rows = [r for r in run_data["results"] if r["status"] != "ok" or r["stumps_error_cm"] > 15]
    rows.sort(key=lambda r: -(r["stumps_error_cm"] if r["status"] == "ok" else 1e9))
    return rows[:count]


def save(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1, default=float))


def _q(values) -> dict:
    values = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return {}
    return {
        "median": round(float(np.median(values)), 3),
        "mean": round(float(values.mean()), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "max": round(float(values.max()), 3),
    }


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None
