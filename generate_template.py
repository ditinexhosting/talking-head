# coding: utf-8
"""
Generate a .pkl motion template from a named pattern in api/nod_patterns.py.

Usage:
    python generate_template.py welcome_interview --duration 25 --fps 25
    python generate_template.py yes --duration 2
"""

import argparse
import pickle
from pathlib import Path

from api.nod_patterns import NOD_PATTERNS, build_motion_template

OUTPUT_DIR = Path("assets/examples/driving")


def main():
    parser = argparse.ArgumentParser(description="Bake a nod/pose pattern into a .pkl driving template")
    parser.add_argument("pattern", choices=list(NOD_PATTERNS.keys()), help="Pattern name")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds (default: 2.0)")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second (default: 25)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: assets/examples/driving/<pattern>.pkl)")
    args = parser.parse_args()

    template = build_motion_template(args.pattern, fps=args.fps, duration_s=args.duration)

    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.pattern}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(template, f)

    print(f"Saved: {out_path}  ({template['n_frames']} frames @ {args.fps} fps = {args.duration:.1f}s)")
    print(f"Pattern: {NOD_PATTERNS[args.pattern]['description']}")


if __name__ == "__main__":
    main()
