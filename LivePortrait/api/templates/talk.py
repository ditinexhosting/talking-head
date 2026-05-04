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

# X — Silence / rest
_EXP_X = [0.0] * _EXP_DIM

# A — Closed lips  (m, b, p): slight lip press, near-neutral
_EXP_A = _make_exp({
    12: ( 0.00,  0.01,  0.00),
    14: ( 0.00, -0.01,  0.00),
    17: ( 0.00,  0.01,  0.00),
    20: ( 0.00,  0.01,  0.00),
})

# B — Upper teeth on lower lip  (f, v): lower lip curls up under upper teeth
_EXP_B = _make_exp({
     6: ( 0.00,  0.01,  0.00),
    12: ( 0.00,  0.04,  0.00),
    17: ( 0.00,  0.04,  0.00),
    19: ( 0.00,  0.01,  0.00),
    20: ( 0.00,  0.03,  0.00),
})

# C — Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)
_EXP_C = _make_exp({
    12: ( 0.00,  0.04,  0.00),
    14: ( 0.00, -0.02,  0.00),
    17: ( 0.00,  0.04,  0.00),
    20: ( 0.00,  0.03,  0.00),
})

# D — Tongue behind upper teeth  (th): slightly open, upper lip raised
_EXP_D = _make_exp({
     6: ( 0.00,  0.01,  0.00),
    12: ( 0.00,  0.04,  0.00),
    14: ( 0.00, -0.03,  0.00),
    17: ( 0.00,  0.04,  0.00),
    19: ( 0.00,  0.01,  0.00),
    20: ( 0.00,  0.03,  0.00),
})

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
