"""Render benchmark output as Markdown."""

from __future__ import annotations

from balltracker.bench import worst_cases

ABLATION_HEADER = [
    "Run", "Deliveries fitted", "Stumps error median (cm)", "p90 (cm)",
    "Bounce error median (cm)", "Speed error median (km/h)", "Hit/miss accuracy",
]  # fmt: skip
WORST_HEADER = [
    "Seed", "Ball", "Status", "Stumps error (cm)", "Detected / visible frames",
    "Calibration RMS (px)",
]  # fmt: skip


def to_markdown(data: dict, command: str) -> str:
    main = data["runs"]["main"]
    s = main["summary"]
    metrics = [
        ("Stumps-line prediction error, from frames before the pad (cm)", s["stumps_error_cm"]),
        ("Bounce point error (cm)", s["bounce_error_cm"]),
        ("Speed error (km/h)", s["speed_abs_error_kmh"]),
        ("3D path RMSE over tracked frames (cm)", s["rmse_3d_cm"]),
        ("Fit reprojection RMS (px)", s["fit_rms_px"]),
        ("Tracked position error vs truth (px)", s["track_error_px"]),
        ("Calibration reprojection RMS (px)", s["calibration_rms_px"]),
        ("Restitution error", s["restitution_abs_error"]),
        ("Drag coefficient error (1/m)", s["drag_abs_error"]),
    ]
    lines = [
        "# Benchmark results",
        "",
        f"Reproduce with `{command}` ({data['machine']}).",
        "",
        "Every delivery is rendered from a seed, so the true 3D path, camera and ball",
        "parameters are known exactly. The camera pose is not taken from the truth: it is",
        "calibrated from pitch landmarks clicked with Gaussian click error, as a user would.",
        "",
        f"## Main run: {s['deliveries']} deliveries at 240 fps, 1.5 px click error",
        "",
        *_table(["Metric", "Median", "p90"],
                [[name, q.get("median", "n/a"), q.get("p90", "n/a")] for name, q in metrics]),
        "",
        f"- Deliveries fitted: {s['ok']} of {s['deliveries']} ({_status(s)})",
        f"- Ball detected in {_pct(s['detection_recall'])} of frames where it was visible",
        f"- Hitting / missing called correctly: {_pct(s['hit_miss_accuracy'])} "
        f"({s['hitting_deliveries']} of the deliveries would have hit)",
        "",
        "### By ball colour",
        "",
        *_table(
            ["Ball", "Deliveries", "Detection recall", "Stumps error median (cm)",
             "Hit/miss accuracy"],
            [[ball, b["deliveries"], _pct(b["detection_recall"]),
              b["stumps_error_cm"].get("median", "n/a"), _pct(b["hit_miss_accuracy"])]
             for ball, b in main["by_ball"].items()],
        ),
        "",
        "## Ablations",
        "",
        "Each row runs the same seeded deliveries with one thing changed.",
        "",
        *_table(ABLATION_HEADER, [_ablation_row(name, run["summary"])
                                  for name, run in data["runs"].items()]),
        "",
        "## Worst cases in the main run",
        "",
        *_table(WORST_HEADER, [
            [r["seed"], r["ball"], r["status"], _num(r["stumps_error_cm"]),
             f"{r['detected_frames']} / {r['visible_frames']}", f"{r['calibration_rms_px']:.2f}"]
            for r in worst_cases(main)
        ]),
        "",
    ]  # fmt: skip
    return "\n".join(lines)


def _ablation_row(name: str, r: dict) -> list:
    return [
        name, f"{r['ok']}/{r['deliveries']}",
        r["stumps_error_cm"].get("median", "n/a"), r["stumps_error_cm"].get("p90", "n/a"),
        r["bounce_error_cm"].get("median", "n/a"), r["speed_abs_error_kmh"].get("median", "n/a"),
        _pct(r["hit_miss_accuracy"]),
    ]  # fmt: skip


def _table(header: list, rows: list[list]) -> list[str]:
    align = ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return out


def _status(s: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in s["status"].items())


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value) -> str:
    return "n/a" if value != value else f"{value:.1f}"
