"""Command line: balltracker bench | demo | track | calibrate-intrinsics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="balltracker", description=__doc__)
    commands = parser.add_subparsers(required=True)

    bench = commands.add_parser("bench", help="evaluate on rendered deliveries, with ablations")
    bench.add_argument("--deliveries", type=int, default=150)
    bench.add_argument("--ablation-deliveries", type=int, default=60)
    bench.add_argument("--workers", type=int, default=8)
    bench.add_argument("--json", type=Path, default=Path("data/bench.json"))
    bench.add_argument("--markdown", type=Path, default=Path("docs/results.md"))
    bench.set_defaults(handler=_bench)

    demo = commands.add_parser("demo", help="render one delivery and show what the tracker sees")
    demo.add_argument("--seed", type=int, default=1000)
    demo.add_argument("--out", type=Path, default=Path("out/demo"))
    demo.set_defaults(handler=_demo)

    track = commands.add_parser("track", help="track a ball in a real video")
    track.add_argument("video", type=Path)
    track.add_argument("--clicks", type=Path, required=True, help="landmark clicks JSON")
    track.add_argument("--intrinsics", type=Path, required=True, help='{"K": ..., "dist": ...}')
    track.add_argument("--start", type=int, default=0)
    track.add_argument("--stop", type=int)
    track.add_argument("--fps", type=float, help="override the frame rate in the file")
    track.add_argument("--out", type=Path, default=Path("out/track"))
    track.set_defaults(handler=_track)

    intrinsics = commands.add_parser(
        "calibrate-intrinsics", help="K and distortion from checkerboard photos"
    )
    intrinsics.add_argument("images", nargs="+", type=Path)
    intrinsics.add_argument("--pattern", default="9x6", help="inner corners, columns x rows")
    intrinsics.add_argument("--square", type=float, required=True, help="square size in metres")
    intrinsics.add_argument("--out", type=Path, default=Path("intrinsics.json"))
    intrinsics.set_defaults(handler=_intrinsics)
    return parser


def _bench(args) -> int:
    from balltracker import bench
    from balltracker.report import to_markdown

    data = bench.run(args.deliveries, args.ablation_deliveries, args.workers)
    bench.save(data, args.json)
    command = (
        f"balltracker bench --deliveries {args.deliveries} "
        f"--ablation-deliveries {args.ablation_deliveries}"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(to_markdown(data, command))
    print(json.dumps(data["runs"]["main"]["summary"], indent=2))
    return 0


def _demo(args) -> int:
    from balltracker.evaluate import EvalConfig, evaluate_delivery
    from balltracker.render import Renderer
    from balltracker.scenario import random_delivery, random_setup

    rng = np.random.default_rng(args.seed)
    delivery, setup = random_delivery(rng), random_setup(rng)
    renderer = Renderer(setup, delivery, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    w, h = setup.camera.size
    writer = cv2.VideoWriter(
        str(args.out / "delivery.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
    )
    trail = renderer.background.copy()
    for k in range(len(renderer)):
        frame = renderer.frame(k)
        writer.write(frame)
        trail = np.maximum(trail, frame)
    writer.release()
    cv2.imwrite(str(args.out / "trail.png"), trail)
    result = evaluate_delivery(args.seed, EvalConfig())
    (args.out / "result.json").write_text(json.dumps(vars(result), indent=2, default=float))
    print(f"wrote {args.out}/delivery.mp4 (slowed to 30 fps), trail.png, result.json")
    return 0


def _track(args) -> int:
    from balltracker.calibrate_scene import calibration_from_file
    from balltracker.video import analyse, load_frames, load_intrinsics, overlay, to_json

    frames, fps = load_frames(args.video, args.start, args.stop)
    K, dist = load_intrinsics(args.intrinsics)
    h, w = frames[0].shape[:2]
    calibration = calibration_from_file(args.clicks, K, dist, (w, h))
    print(f"calibration: {calibration.points} landmarks, {calibration.rms_px:.2f} px RMS")
    result = analyse(frames, args.fps or fps, calibration.camera)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(to_json(result), indent=2, default=float))
    cv2.imwrite(
        str(args.out / "overlay.png"),
        overlay(frames[result.track.frames[-1]], calibration.camera, result),
    )
    print(json.dumps(to_json(result), indent=2, default=float))
    return 0


def _intrinsics(args) -> int:
    from balltracker.camera import calibrate_intrinsics

    columns, rows = map(int, args.pattern.lower().split("x"))
    images = [cv2.imread(str(p)) for p in args.images]
    K, dist, rms = calibrate_intrinsics(images, (columns, rows), args.square)
    args.out.write_text(
        json.dumps({"K": K.tolist(), "dist": dist.tolist(), "rms_px": rms}, indent=2)
    )
    print(f"RMS {rms:.3f} px -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
