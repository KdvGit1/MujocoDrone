"""
Progressive test suite: EASY → MEDIUM → HARD
Runs headless (rgb_array) so video can be captured on any machine.

Usage:
    python test_progressive.py

Outputs:
    videos/test_easy.mp4    – static hover, 5 episodes
    videos/test_medium.mp4  – challenge radius 0.5 m, 3 episodes
    videos/test_hard.mp4    – challenge radius 1.5 m, short interval, 3 episodes
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "training", "evaluate.py")

BEST_MODEL   = os.path.join(ROOT, "models", "trained", "best", "best_model.zip")
VEC_NORM     = os.path.join(ROOT, "models", "trained", "best", "vec_normalize.pkl")

VIDEO_DIR = os.path.join(ROOT, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

LEVELS = [
    {
        "name":    "EASY  – Static Hover",
        "video":   os.path.join(VIDEO_DIR, "test_easy.mp4"),
        "args": [
            "--episodes", "5",
            "--no-render",           # headless → rgb_array handled by --video
            "--vec-normalize", VEC_NORM,
            # no --challenge  → drone just holds position at [0,0,1]
        ],
    },
    {
        "name":    "MEDIUM – Nav Challenge (radius 0.5 m, interval 200)",
        "video":   os.path.join(VIDEO_DIR, "test_medium.mp4"),
        "args": [
            "--episodes", "3",
            "--no-render",
            "--vec-normalize", VEC_NORM,
            "--challenge",
            "--cmd-radius",   "0.5",
            "--cmd-interval", "200",
        ],
    },
    {
        "name":    "HARD  – Nav Challenge (radius 1.5 m, interval 80)",
        "video":   os.path.join(VIDEO_DIR, "test_hard.mp4"),
        "args": [
            "--episodes", "3",
            "--no-render",
            "--vec-normalize", VEC_NORM,
            "--challenge",
            "--cmd-radius",   "1.5",
            "--cmd-interval", "80",
        ],
    },
]


def _separator(title: str):
    line = "═" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


if __name__ == "__main__":
    for i, level in enumerate(LEVELS, 1):
        _separator(f"LEVEL {i}/3 | {level['name']}")
        cmd = [
            sys.executable, EVAL,
            BEST_MODEL,
            "--video", level["video"],
        ] + level["args"]

        print("Command:", " ".join(cmd), "\n")
        result = subprocess.run(cmd, cwd=ROOT)

        if result.returncode != 0:
            print(f"\n[!] Level {i} finished with non-zero exit code {result.returncode}")
        else:
            print(f"\n[✓] Video saved → {level['video']}")

    _separator("ALL TESTS COMPLETE")
    print("Videos written to:")
    for level in LEVELS:
        exists = "✓" if os.path.exists(level["video"]) else "✗"
        print(f"  [{exists}] {level['video']}")
    print()
