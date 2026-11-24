import json

import cv2
import numpy as np

from balltracker.cli import main
from balltracker.pitch import landmarks
from balltracker.render import Renderer
from balltracker.scenario import random_delivery, random_setup


def test_track_command_on_a_video_file(tmp_path):
    rng = np.random.default_rng(21)
    delivery, setup = random_delivery(rng), random_setup(rng, fps=120)
    renderer = Renderer(setup, delivery, 21)
    w, h = setup.camera.size
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 120, (w, h))
    for k in range(len(renderer)):
        writer.write(renderer.frame(k))
    writer.release()
    in_view = {n: p for n, p in landmarks().items() if setup.camera.in_view(p[None, :])[0]}
    clicks = {n: setup.camera.project(p).tolist() for n, p in in_view.items()}
    (tmp_path / "clicks.json").write_text(json.dumps(clicks))
    intrinsics = {"K": setup.camera.K.tolist(), "dist": [0, 0, 0, 0, 0]}
    (tmp_path / "intrinsics.json").write_text(json.dumps(intrinsics))
    out = tmp_path / "out"
    code = main(
        [
            "track",
            str(video),
            "--clicks",
            str(tmp_path / "clicks.json"),
            "--intrinsics",
            str(tmp_path / "intrinsics.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    result = json.loads((out / "result.json").read_text())
    assert abs(result["speed_kmh"] - delivery.speed_kmh) < 15
    assert (out / "overlay.png").exists()
