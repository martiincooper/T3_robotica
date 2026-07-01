"""Record a video of the G2T episode from CoppeliaSim (off-screen render).

Replays the ground-truth trajectory stored in dataset.h5, placing the three
bodies each step, and captures frames from a camera overlooking the scene.
Works even when the interactive 3-D viewport does not draw (macOS Edu), because
it uses the same off-screen vision-sensor renderer as the LiDAR.

    python coppelia/record_video.py                 # -> report/figures/episode.gif
    python coppelia/record_video.py --mp4           # -> episode.mp4 (needs imageio-ffmpeg)

Deliverable: satisfies the "videos que evidencien el funcionamiento" requirement.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coppelia._client import connect, version_string
from coppelia.scenario import Scenario
from coppelia.vehicle import make_kinematics, find_bodies, place
from coppelia.capture_scene import _look_at
from g2t_core.simulation.g2t_sim.kinematics import G2TState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="coppelia/g3_scene.ttt")
    ap.add_argument("--dataset", default="datasets/g3_run/dataset.h5")
    ap.add_argument("--out", default="report/figures/episode")
    ap.add_argument("--stride", type=int, default=8, help="log every Nth step")
    ap.add_argument("--res", type=int, nargs=2, default=[720, 540])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--top", action="store_true")
    args = ap.parse_args()

    sc = Scenario.load(ROOT / "config/scenario_g3.yaml")
    with h5py.File(ROOT / args.dataset, "r") as f:
        gt = f["gt/state"][:]

    client, sim = connect()
    print(version_string(sim))
    sim.loadScene(str(ROOT / args.scene))
    try:
        sim.setArrayParam(sim.arrayparam_ambient_light, [0.6, 0.6, 0.6])
    except Exception:
        pass

    bodies = find_bodies(sim)
    kin = make_kinematics(sc.cfg)
    w, h = sc.size
    cam = sim.createVisionSensor(1, [args.res[0], args.res[1], 0, 0],
                                 [0.05, 200.0, math.radians(60), 0.2,
                                  0, 0, 0, 0, 0, 0, 0])
    for fn in (
        lambda: sim.setObjectInt32Param(cam, sim.visionintparam_perspective_operation, 1),
        lambda: sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, math.radians(60)),
        lambda: sim.setObjectFloatParam(cam, sim.visionfloatparam_far_clipping, 200.0),
    ):
        try:
            fn()
        except Exception:
            pass
    if args.top:
        pose = _look_at([0, 0, max(w, h) * 1.3], [0, 0, 0], up=(0, 1, 0))
    else:
        pose = _look_at([-w * 0.75, -h * 1.1, max(w, h) * 0.7], [0, 0, 0])
    sim.setObjectPose(cam, -1, pose)

    sim.setStepping(True)
    sim.startSimulation()
    frames = []
    for k in range(0, len(gt), args.stride):
        s = gt[k]
        place(sim, bodies, kin, G2TState(s[0], s[1], s[2], s[3], s[4]))
        sim.step()
        sim.handleVisionSensor(cam)
        img, res = sim.getVisionSensorImg(cam)
        a = (np.frombuffer(img, np.uint8) if isinstance(img, (bytes, bytearray))
             else np.array(img, np.uint8)).reshape(res[1], res[0], 3)[::-1]
        frames.append(a)
    sim.stopSimulation()
    sim.removeObjects([cam])
    print(f"captured {len(frames)} frames")

    outbase = ROOT / args.out
    outbase.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        if args.mp4:
            path = str(outbase) + ".mp4"
            imageio.mimsave(path, frames, fps=args.fps)
        else:
            path = str(outbase) + ".gif"
            imageio.mimsave(path, frames, fps=args.fps)
        print(f"wrote {path}")
    except Exception as exc:
        # fallback: dump PNG frames + ffmpeg hint
        fdir = outbase.parent / "episode_frames"
        fdir.mkdir(exist_ok=True)
        import matplotlib.pyplot as plt
        for i, fr in enumerate(frames):
            plt.imsave(str(fdir / f"f{i:04d}.png"), fr)
        print(f"[record_video] imageio unavailable ({exc}); wrote PNG frames to "
              f"{fdir}. Assemble with:\n"
              f"  ffmpeg -framerate {args.fps} -i {fdir}/f%04d.png "
              f"-pix_fmt yuv420p {outbase}.mp4")


if __name__ == "__main__":
    main()
