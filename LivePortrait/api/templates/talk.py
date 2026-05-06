from api.utils.motion import Keyframe, _EXP_DIM


# Lip keypoint indices within the 21-kp set (from pipeline animation_region="lip")
# [6, 12, 14, 17, 19, 20] — each occupies 3 consecutive slots in the flat 63-d exp vector.
#
# Coordinate convention (LivePortrait implicit keypoint space):
#   x — horizontal  (negative = left,  positive = right)
#   y — vertical    (negative = up,    positive = down)
#   z — depth       (positive = forward / toward camera)
#
# All values are empirical deltas from the neutral (closed-mouth) pose.
# Tune the magnitudes if the generated animation looks over- or under-stated.

def _make_exp(kp_deltas: dict) -> list[float]:
    """Build a 63-d exp array from {kp_index: (dx, dy, dz)}."""
    exp = [0.0] * _EXP_DIM
    for kp_idx, (dx, dy, dz) in kp_deltas.items():
        b = kp_idx * 3
        exp[b], exp[b + 1], exp[b + 2] = dx, dy, dz
    return exp


# ---------------------------------------------------------------------------
# Per-viseme exp arrays
# ---------------------------------------------------------------------------
# Rhubarb Lip Sync mouth-shape reference:
#   A — Closed mouth for “P”, “B”, and “M” sounds. Almost identical to X,
#       but with ever-so-slight pressure between the lips.
#   B — Slightly open mouth with clenched teeth. Used for most consonants
#       (“K”, “S”, “T”, etc.) and some vowels such as “EE” in “bee”.
#   C — Open mouth. Used for vowels like “EH” in “men” and “AE” in “bat”,
#       and some consonants depending on context. Also an in-between from
#       A/B to D, so A-C-D and B-C-D should animate smoothly.
#   D — Wide open mouth. Used for vowels like “AA” in “father”.
#   E — Slightly rounded mouth. Used for vowels like “AO” in “off” and
#       “ER” in “bird”. Also an in-between from C/D to F; it should not be
#       wider open than C, and C-E-F / D-E-F should animate smoothly.
#   F — Puckered lips. Used for “UW” in “you”, “OW” in “show”, and “W” in
#       “way”.
#   G — Upper teeth touching the lower lip for “F” in “for” and “V” in
#       “very”. Optional extended shape.
#   H — Long “L” sounds, with the tongue raised behind the upper teeth.
#       The mouth should be at least as open as C, but not quite as open as D.
#       Optional extended shape.
#   X — Idle/rest position for pauses in speech. Almost identical to A, but
#       with closed, relaxed lips and slightly less pressure. Optional extended
#       shape.

# X — Silence / rest
_EXP_X = [0.0] * _EXP_DIM

# A — Closed mouth (m, b, p): slight lip press, near-neutral
_EXP_A = {
    6:  (0.00,  -0.00, 0.00),
    12: (0.00,  -0.00, 0.00),
    14: (0.00,  0.003, 0.00), # Left smile
    17: (0.00,  -0.003, 0.00), # Right smile
    19: (0.00, -0.003, 0.00), # Bottom lip
    20: (0.00, 0.003, 0.00), # Top lip
}

# B — Slightly open mouth with clenched teeth.
_EXP_B = {
    6:  (0.00,  0.02, 0.00),
    12: (0.00,  0.02, 0.00),
    14: (0.00,  -0.00, 0.00),
    17: (0.00,  0.00, 0.00),
    19: (0.00, 0.01, 0.00), # Bottom lip
    20: (0.00, -0.006, 0.00), # Top lip
}

# C — Open mouth
_EXP_C = {
    6:  (0.00,  0.02, 0.00),
    12: (0.00,  0.02, 0.00),
    14: (0.00,  -0.00, 0.00),
    17: (0.00,  0.00, 0.00),
    19: (0.00, 0.02, 0.00), # Bottom lip
    20: (0.00, -0.008, 0.00), # Top lip
}

# D — Wide open mouth
_EXP_D = {
    6:  (0.00,  0.02, 0.00),
    12: (0.00,  0.02, 0.00),
    14: (0.00,  -0.003, 0.00),
    17: (0.00,  0.003, 0.00),
    19: (0.00, 0.04, 0.00), # Bottom lip
    20: (0.00, -0.015, 0.00), # Top lip
}

# E — Tongue up  (l): slight open, similar to C
_EXP_E = _make_exp({
    12: ( 0.00,  0.04,  0.00),
    14: ( 0.00, -0.02,  0.00),
    17: ( 0.00,  0.04,  0.00),
    20: ( 0.00,  0.03,  0.00),
})

# F — Rounded / puckered lips  (w, q): corners pull inward, lips protrude
_EXP_F = {
     6: ( 0.02,  0.00,  0.02),
    12: ( 0.00,  0.01,  0.04),
    14: ( 0.00, -0.01,  0.04),
    17: ( 0.00,  0.01,  0.03),
    19: (-0.02,  0.00,  0.02),
    20: ( 0.00,  0.01,  0.03),
}


# G — Relaxed open mouth  (a, i): medium jaw drop
_EXP_G = _make_exp({
     6: ( 0.00,  0.02,  0.00),
    12: ( 0.00,  0.07,  0.00),
    14: ( 0.00, -0.03,  0.00),
    17: ( 0.00,  0.07,  0.00),
    19: ( 0.00,  0.02,  0.00),
    20: ( 0.00,  0.05,  0.00),
})

# H — Wide open mouth  (short a, e): maximum jaw drop
_EXP_H = {
     6: ( 0.00,  0.05,  0.00),
    12: ( 0.00,  0.11,  0.00),
    # 14: ( 0.00, -0.05,  0.00),
    # 17: ( 0.00,  0.11,  0.00),
    # 19: ( 0.00,  0.03,  0.00),
    # 20: ( 0.00,  0.08,  0.00),
}



def mouth_shape(start_frame: int):
    keyframes = [
        Keyframe(frame_idx=start_frame),
        Keyframe(frame_idx=start_frame + 2,  mouth=_EXP_H),
        Keyframe(frame_idx=start_frame + 8,  mouth=_EXP_H),
        Keyframe(frame_idx=start_frame + 10),
    ]
    end_frame = keyframes[-1].frame
    return keyframes, end_frame
