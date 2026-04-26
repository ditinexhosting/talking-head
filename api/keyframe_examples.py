# coding: utf-8
"""
Examples of using the Keyframe-based animation API with detailed facial control.

Each parameter controls specific facial movements:
  - Eyeballs: gaze_x (left/right), gaze_y (up/down)
  - Eye opening: eye_open_left, eye_open_right (0=closed, 1=wide)
  - Eyebrows: raise (up/down), angle (tilt), inner_raise
  - Mouth: open (0=closed, 1=open), smile (-1=frown, 1=smile), pucker, x/y shift
  - Nose: x (flare), y (wrinkle)
  - Head: pitch (nod), yaw (turn), roll (tilt)
"""

import numpy as np
from api.motion import Keyframe, build_motion_from_keyframes


# ======================
# Listening movements
# ======================

# Listening nod with smile
def listening_nod_with_smile():
    """Head nod with smile."""
    keyframes = [
        Keyframe(frame_idx=0, pitch_deg=0, yaw_deg=0, roll_deg=0),      # neutral
        Keyframe(frame_idx=10, pitch_deg=5, yaw_deg=0, roll_deg=0, mouth_smile=0.3),   # nod down
        Keyframe(frame_idx=20, pitch_deg=0, yaw_deg=0, roll_deg=0, mouth_smile=0.3),   # neutral
        Keyframe(frame_idx=30, pitch_deg=4, yaw_deg=0, roll_deg=0, mouth_smile=0.3),   # nod down
        Keyframe(frame_idx=40, pitch_deg=0, yaw_deg=0, roll_deg=0),   # neutral
    ]
    return build_motion_from_keyframes(keyframes, n_frames=40, fps=25)


# Listening nod with curiocity






# ======================
# Talking Head Movements
# ======================

def talking_left_side_head_movement(start_frame=0):
    """Gesture: Move head sideways while talking."""
    keyframes = [
        Keyframe(frame_idx=start_frame + 0, yaw_deg=0),
        Keyframe(frame_idx=start_frame + 10, yaw_deg=6),
        Keyframe(frame_idx=start_frame + 20, yaw_deg=0)
    ]
    return start_frame + 20, keyframes

def talking_right_side_head_movement(start_frame=0):
    """Gesture: Move head sideways while talking."""
    keyframes = [
        Keyframe(frame_idx=start_frame + 0, yaw_deg=0),
        Keyframe(frame_idx=start_frame + 10, yaw_deg=-6),
        Keyframe(frame_idx=start_frame + 20, yaw_deg=0)
    ]
    return start_frame + 20, keyframes


def talking_tilt_top_right_head_movement(start_frame=0):
    """Gesture: Tilt head while talking."""
    keyframes = [
        Keyframe(frame_idx=start_frame + 0, yaw_deg=0, roll_deg=0, pitch_deg=0),
        Keyframe(frame_idx=start_frame + 10, yaw_deg=-2 , roll_deg=-5, pitch_deg=-1),
        Keyframe(frame_idx=start_frame + 20, yaw_deg=0, roll_deg=0, pitch_deg=0)
    ]
    return start_frame + 20, keyframes


def talking_tilt_top_left_head_movement(start_frame=0):
    """Gesture: Tilt head while talking."""
    keyframes = [
        Keyframe(frame_idx=start_frame + 0, yaw_deg=0, roll_deg=0, pitch_deg=0),
        Keyframe(frame_idx=start_frame + 10, yaw_deg=2 , roll_deg=5, pitch_deg=-1),
        Keyframe(frame_idx=start_frame + 20, yaw_deg=0, roll_deg=0, pitch_deg=0)
    ]
    return start_frame + 20, keyframes

def talking_chin_up_movement(start_frame=0):
    """Gesture: Chin up while talking."""
    keyframes = [
        Keyframe(frame_idx=start_frame + 0, yaw_deg=0, roll_deg=0, pitch_deg=0),
        Keyframe(frame_idx=start_frame + 10, yaw_deg=0 , roll_deg=0, pitch_deg=-3),
        Keyframe(frame_idx=start_frame + 15, yaw_deg=0 , roll_deg=0, pitch_deg=-7),
        Keyframe(frame_idx=start_frame + 20, yaw_deg=0 , roll_deg=0, pitch_deg=-3),
        Keyframe(frame_idx=start_frame + 25, yaw_deg=0, roll_deg=0, pitch_deg=0)
    ]
    return start_frame + 25, keyframes

# ============================================================================
# Example 1: Simple head movements
# ============================================================================
def example_head_movements():
    """Head nod, turn left, turn right, and tilt."""
    keyframes = [
        Keyframe(frame_idx=0, pitch_deg=0, yaw_deg=0, roll_deg=0),      # neutral
        Keyframe(frame_idx=30, pitch_deg=-10, yaw_deg=0, roll_deg=0),   # nod down
        Keyframe(frame_idx=60, pitch_deg=0, yaw_deg=20, roll_deg=0),    # turn left
        Keyframe(frame_idx=90, pitch_deg=0, yaw_deg=-20, roll_deg=0),   # turn right
        Keyframe(frame_idx=120, pitch_deg=0, yaw_deg=0, roll_deg=15),   # tilt right
        Keyframe(frame_idx=150, pitch_deg=0, yaw_deg=0, roll_deg=0),    # back to neutral
    ]
    return build_motion_from_keyframes(keyframes, n_frames=150, fps=25)


# ============================================================================
# Example 2: Eye movements and gaze
# ============================================================================
def example_eye_movements():
    """Look left, right, up, down, and blink."""
    keyframes = [
        Keyframe(frame_idx=0, eye_gaze_x=0, eye_gaze_y=0,
                 eye_open_left=0.30, eye_open_right=0.30),                    # center gaze
        Keyframe(frame_idx=25, eye_gaze_x=-1.0, eye_gaze_y=0,
                 eye_open_left=0.30, eye_open_right=0.30),                    # look left
        Keyframe(frame_idx=50, eye_gaze_x=1.0, eye_gaze_y=0,
                 eye_open_left=0.30, eye_open_right=0.30),                    # look right
        Keyframe(frame_idx=75, eye_gaze_x=0, eye_gaze_y=1.0,
                 eye_open_left=0.30, eye_open_right=0.30),                    # look up
        Keyframe(frame_idx=100, eye_gaze_x=0, eye_gaze_y=-1.0,
                 eye_open_left=0.30, eye_open_right=0.30),                    # look down
        Keyframe(frame_idx=125, eye_gaze_x=0, eye_gaze_y=0,
                 eye_open_left=0.05, eye_open_right=0.05),                    # blink
        Keyframe(frame_idx=150, eye_gaze_x=0, eye_gaze_y=0,
                 eye_open_left=0.30, eye_open_right=0.30),                    # open eyes again
    ]
    return build_motion_from_keyframes(keyframes, n_frames=160, fps=25)


# ============================================================================
# Example 3: Eyebrow control
# ============================================================================
def example_eyebrows():
    """Raise, lower, and angle eyebrows."""
    keyframes = [
        Keyframe(frame_idx=0,
                 brow_raise_left=0, brow_raise_right=0,
                 brow_angle_left=0, brow_angle_right=0),                      # neutral
        Keyframe(frame_idx=40,
                 brow_raise_left=1.0, brow_raise_right=1.0,
                 brow_angle_left=0, brow_angle_right=0),                      # both raised (surprised)
        Keyframe(frame_idx=80,
                 brow_raise_left=-1.0, brow_raise_right=-1.0,
                 brow_angle_left=0, brow_angle_right=0),                      # both lowered (angry)
        Keyframe(frame_idx=120,
                 brow_raise_left=0.5, brow_raise_right=-0.5,
                 brow_angle_left=-0.5, brow_angle_right=0.5),                 # asymmetric (confused)
        Keyframe(frame_idx=150,
                 brow_raise_left=0, brow_raise_right=0,
                 brow_angle_left=0, brow_angle_right=0),                      # back to neutral
    ]
    return build_motion_from_keyframes(keyframes, n_frames=160, fps=25)


# ============================================================================
# Example 4: Mouth expressions
# ============================================================================
def example_mouth_expressions():
    """Smile, frown, talk, pucker, and shift mouth."""
    keyframes = [
        Keyframe(frame_idx=0, mouth_open=0, mouth_smile=0, mouth_pucker=0),  # closed, neutral
        Keyframe(frame_idx=30, mouth_open=0, mouth_smile=1.0, mouth_pucker=0), # big smile
        Keyframe(frame_idx=60, mouth_open=0, mouth_smile=-1.0, mouth_pucker=0), # frown
        Keyframe(frame_idx=90, mouth_open=0.5, mouth_smile=0, mouth_pucker=0), # open mouth
        Keyframe(frame_idx=120, mouth_open=0, mouth_smile=0, mouth_pucker=1.0), # pucker/kiss
        Keyframe(frame_idx=150, mouth_x=0.3, mouth_y=0.2),                   # shift mouth right & up
        Keyframe(frame_idx=180, mouth_open=0, mouth_smile=0, mouth_pucker=0, mouth_x=0, mouth_y=0), # reset
    ]
    return build_motion_from_keyframes(keyframes, n_frames=200, fps=25)


# ============================================================================
# Example 5: Speaking animation (simple)
# ============================================================================
def example_talking():
    """Simple talking animation with mouth opening & closing."""
    keyframes = []

    # Start neutral
    keyframes.append(Keyframe(frame_idx=0, mouth_open=0))

    # Simulate 3 cycles of talking (opening/closing mouth)
    for cycle in range(3):
        base_frame = 25 + cycle * 40
        keyframes.append(Keyframe(frame_idx=base_frame, mouth_open=0.6))         # open
        keyframes.append(Keyframe(frame_idx=base_frame + 15, mouth_open=0.1))    # close
        keyframes.append(Keyframe(frame_idx=base_frame + 30, mouth_open=0.0))    # closed

    # Add some head nod while talking
    keyframes.append(Keyframe(frame_idx=25, pitch_deg=-3))
    keyframes.append(Keyframe(frame_idx=65, pitch_deg=2))
    keyframes.append(Keyframe(frame_idx=105, pitch_deg=-3))
    keyframes.append(Keyframe(frame_idx=145, pitch_deg=0))

    keyframes.sort(key=lambda k: k.frame_idx)
    return build_motion_from_keyframes(keyframes, n_frames=160, fps=25)


# ============================================================================
# Example 6: Complex emotion sequence
# ============================================================================
def example_emotions():
    """Sequence showing: surprise → confusion → smile → sadness."""
    keyframes = [
        # Surprise (wide eyes, raised brows, open mouth)
        Keyframe(
            frame_idx=0,
            eye_open_left=0.37, eye_open_right=0.37,
            brow_raise_left=1.0, brow_raise_right=1.0,
            mouth_open=0.3,
        ),
        # Confusion (furrowed brows, gaze down, head tilt)
        Keyframe(
            frame_idx=40,
            brow_raise_left=-0.5, brow_raise_right=-0.5,
            brow_angle_left=-0.3, brow_angle_right=0.3,
            eye_gaze_y=-0.5,
            roll_deg=10,
        ),
        # Big smile
        Keyframe(
            frame_idx=80,
            mouth_smile=1.0,
            eye_open_left=0.20, eye_open_right=0.20,  # eyes squint
            brow_raise_left=0.3, brow_raise_right=0.3,
        ),
        # Sadness (lowered eyes, frown, inner brow raise)
        Keyframe(
            frame_idx=120,
            eye_gaze_y=-0.8,
            eye_open_left=0.25, eye_open_right=0.25,
            mouth_smile=-0.7,
            brow_inner_raise=1.0,
        ),
        # Back to neutral
        Keyframe(
            frame_idx=160,
            eye_open_left=0.30, eye_open_right=0.30,
            eye_gaze_x=0, eye_gaze_y=0,
            brow_raise_left=0, brow_raise_right=0,
            brow_inner_raise=0,
            mouth_smile=0,
            roll_deg=0,
        ),
    ]
    return build_motion_from_keyframes(keyframes, n_frames=170, fps=25)


# ============================================================================
# Example 7: Nose control (flare & wrinkle)
# ============================================================================
def example_nose():
    """Nostril flare and nose wrinkle."""
    keyframes = [
        Keyframe(frame_idx=0, nose_x=0, nose_y=0),              # neutral
        Keyframe(frame_idx=30, nose_x=1.0, nose_y=0),           # flare left nostril
        Keyframe(frame_idx=60, nose_x=-1.0, nose_y=0),          # flare right nostril
        Keyframe(frame_idx=90, nose_x=0, nose_y=1.0),           # wrinkle nose
        Keyframe(frame_idx=120, nose_x=0, nose_y=0),            # back to neutral
    ]
    return build_motion_from_keyframes(keyframes, n_frames=130, fps=25)


# ============================================================================
# Example 8: Scale/distance control
# ============================================================================
def example_scale_and_translation():
    """Move face closer/farther and shift position."""
    keyframes = [
        Keyframe(frame_idx=0, scale=1.393, tx=0.0, ty=0.10, tz=0.0),  # normal
        Keyframe(frame_idx=40, scale=1.6, tx=0.0, ty=0.10, tz=0.0),   # zoom in
        Keyframe(frame_idx=80, scale=1.2, tx=0.0, ty=0.10, tz=0.0),   # zoom out
        Keyframe(frame_idx=120, scale=1.393, tx=0.1, ty=0.20, tz=0.0), # shift right & down
        Keyframe(frame_idx=160, scale=1.393, tx=0.0, ty=0.10, tz=0.0), # back to normal
    ]
    return build_motion_from_keyframes(keyframes, n_frames=170, fps=25)


# ============================================================================
# Example 9: Asymmetric left/right eye control
# ============================================================================
def example_asymmetric_eyes():
    """Wink and control each eye independently."""
    keyframes = [
        Keyframe(frame_idx=0,
                 eye_open_left=0.30, eye_open_right=0.30),      # both open
        Keyframe(frame_idx=25,
                 eye_open_left=0.05, eye_open_right=0.30),      # wink left
        Keyframe(frame_idx=50,
                 eye_open_left=0.30, eye_open_right=0.05),      # wink right
        Keyframe(frame_idx=75,
                 eye_open_left=0.30, eye_open_right=0.30),      # both open
        Keyframe(frame_idx=100,
                 eye_open_left=0.05, eye_open_right=0.05),      # double blink
        Keyframe(frame_idx=125,
                 eye_open_left=0.30, eye_open_right=0.30),      # open
    ]
    return build_motion_from_keyframes(keyframes, n_frames=135, fps=25)


# ============================================================================
# Parameter Ranges Reference
# ============================================================================
PARAMETER_REFERENCE = {
    "Head pose (degrees)": {
        "pitch_deg": "[-90, 90] — nod up/down",
        "yaw_deg": "[-180, 180] — turn left/right",
        "roll_deg": "[-90, 90] — tilt left/right",
    },
    "Scale & Translation": {
        "scale": "[0.5, 2.0] — default 1.393",
        "tx": "[-1, 1] — horizontal shift",
        "ty": "[-1, 1] — vertical shift (default 0.10)",
        "tz": "[-1, 1] — depth shift",
    },
    "Eye control": {
        "eye_open_left": "[0.01, 1.0] — 0=closed, 1=wide (default 0.30)",
        "eye_open_right": "[0.01, 1.0] — same for right eye",
        "eye_gaze_x": "[-1, 1] — -1=left, 0=center, 1=right",
        "eye_gaze_y": "[-1, 1] — -1=down, 0=center, 1=up",
    },
    "Eyebrow control": {
        "brow_raise_left": "[-1, 1] — -1=down, 0=neutral, 1=up",
        "brow_raise_right": "[-1, 1] — same for right brow",
        "brow_inner_raise": "[-1, 1] — inner brow raise (sadness)",
        "brow_angle_left": "[-1, 1] — tilt left brow",
        "brow_angle_right": "[-1, 1] — tilt right brow",
    },
    "Mouth control": {
        "mouth_open": "[0, 1] — 0=closed, 1=wide open",
        "mouth_smile": "[-1, 1] — -1=frown, 0=neutral, 1=smile",
        "mouth_pucker": "[-1, 1] — -1=pucker, 0=neutral, 1=unpucker",
        "mouth_x": "[-1, 1] — horizontal shift",
        "mouth_y": "[-1, 1] — vertical shift",
    },
    "Nose control": {
        "nose_x": "[-1, 1] — nostril flare",
        "nose_y": "[-1, 1] — nose wrinkle",
    },
}


if __name__ == "__main__":
    # Run examples
    print("Example animations available:")
    print("  - example_head_movements()")
    print("  - example_eye_movements()")
    print("  - example_eyebrows()")
    print("  - example_mouth_expressions()")
    print("  - example_talking()")
    print("  - example_emotions()")
    print("  - example_nose()")
    print("  - example_scale_and_translation()")
    print("  - example_asymmetric_eyes()")
