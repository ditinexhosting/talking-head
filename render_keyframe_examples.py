#!/usr/bin/env python3
# coding: utf-8
"""
CLI tool to render example animations locally.

Usage:
    python render_keyframe_examples.py <example_name> [--source <image>] [--output <dir>]
    python render_keyframe_examples.py --list
    python render_keyframe_examples.py emotions --source assets/my_photo.png

Available examples:
    - head_movements      Head nod, turn left/right, and tilt
    - eye_movements       Look left, right, up, down, and blink
    - eyebrows            Raise, lower, and angle eyebrows
    - mouth_expressions   Smile, frown, talk, pucker, shift mouth
    - talking             Talking animation (mouth opening/closing)
    - emotions            Emotion sequence (surprise → confusion → smile → sadness)
    - nose                Nostril flare and nose wrinkle
    - scale_and_translation  Zoom and shift position
    - asymmetric_eyes     Wink and control each eye independently
"""

import sys
import pickle
import argparse
from pathlib import Path

# Add root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from api.motion import build_motion_from_keyframes
from api import keyframe_examples


EXAMPLES = {
    "head_movements": {
        "func": keyframe_examples.example_head_movements,
        "description": "Head nod, turn left/right, and tilt"
    },
    "eye_movements": {
        "func": keyframe_examples.example_eye_movements,
        "description": "Look left, right, up, down, and blink"
    },
    "eyebrows": {
        "func": keyframe_examples.example_eyebrows,
        "description": "Raise, lower, and angle eyebrows (surprise, angry, confused)"
    },
    "mouth_expressions": {
        "func": keyframe_examples.example_mouth_expressions,
        "description": "Smile, frown, talk, pucker, and shift mouth"
    },
    "talking": {
        "func": keyframe_examples.example_talking,
        "description": "Simple talking animation with mouth opening/closing cycles"
    },
    "emotions": {
        "func": keyframe_examples.example_emotions,
        "description": "Emotion sequence: surprise → confusion → smile → sadness"
    },
    "nose": {
        "func": keyframe_examples.example_nose,
        "description": "Nostril flare and nose wrinkle"
    },
    "scale_and_translation": {
        "func": keyframe_examples.example_scale_and_translation,
        "description": "Move face closer/farther and shift position"
    },
    "asymmetric_eyes": {
        "func": keyframe_examples.example_asymmetric_eyes,
        "description": "Wink and control each eye independently"
    },
}


def list_examples():
    """Print all available examples."""
    print("\nAvailable Examples:")
    print("=" * 60)
    for name, info in EXAMPLES.items():
        print(f"\n  {name}")
        print(f"    {info['description']}")


def render_example(example_name, source_image, output_dir):
    """Render an example animation."""
    if example_name not in EXAMPLES:
        print(f"Error: Unknown example '{example_name}'")
        print(f"Available: {', '.join(EXAMPLES.keys())}")
        return False

    if not Path(source_image).exists():
        print(f"Error: Source image not found: {source_image}")
        return False

    print(f"Loading pipeline...")
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

    pipeline = LivePortraitPipeline(
        inference_cfg=InferenceConfig(),
        crop_cfg=CropConfig(),
    )

    print(f"Rendering example: {example_name}")
    template_func = EXAMPLES[example_name]["func"]
    template_dct = template_func()

    n_frames = template_dct["n_frames"]
    fps = template_dct["output_fps"]
    print(f"  Frames: {n_frames}, FPS: {fps}, Duration: {n_frames/fps:.1f}s")

    # Save template to temp pkl
    import tempfile
    tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    pickle.dump(template_dct, tmp_pkl)
    tmp_pkl.close()

    try:
        from src.config.argument_config import ArgumentConfig
        args = ArgumentConfig(
            source=source_image,
            driving=tmp_pkl.name,
            output_dir=output_dir,
            flag_save_concat=False,
        )

        print(f"Executing pipeline...")
        wfp, _ = pipeline.execute(args)
        print(f"✓ Video saved: {wfp}")
        return True

    finally:
        import os
        os.unlink(tmp_pkl.name)


def main():
    parser = argparse.ArgumentParser(
        description="Render example animations with keyframe control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "example",
        nargs="?",
        help="Example name to render"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available examples"
    )

    parser.add_argument(
        "-s", "--source",
        default="assets/my_photo.png",
        help="Source image path (default: assets/my_photo.png)"
    )

    parser.add_argument(
        "-o", "--output",
        default="animations",
        help="Output directory (default: animations)"
    )

    args = parser.parse_args()

    if args.list:
        list_examples()
        return

    if not args.example:
        parser.print_help()
        return

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    success = render_example(args.example, args.source, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
