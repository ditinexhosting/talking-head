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
_EXP_F = _make_exp({
     6: ( 0.02,  0.00,  0.02),
    12: ( 0.00,  0.01,  0.04),
    14: ( 0.00, -0.01,  0.04),
    17: ( 0.00,  0.01,  0.03),
    19: (-0.02,  0.00,  0.02),
    20: ( 0.00,  0.01,  0.03),
})

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
_EXP_H = _make_exp({
     6: ( 0.00,  0.03,  0.00),
    12: ( 0.00,  0.11,  0.00),
    14: ( 0.00, -0.05,  0.00),
    17: ( 0.00,  0.11,  0.00),
    19: ( 0.00,  0.03,  0.00),
    20: ( 0.00,  0.08,  0.00),
})


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

# Lip-open ratio per viseme (0.0 = closed, ~0.8 = wide open).
# Passed to the lip-retargeting network (R_lip) as c_lip_lst.
VISEME_LIP: dict[str, float] = {
    "X": 0.00,  # silence / rest
    "A": 0.00,  # closed lips (m, b, p)
    "B": 0.20,  # upper teeth on lower lip (f, v)
    "C": 0.30,  # slightly open (d, g, k, n, r, s, t, y, z)
    "D": 0.30,  # tongue behind upper teeth (th)
    "E": 0.30,  # tongue up (l)
    "F": 0.25,  # rounded / puckered (w, q)
    "G": 0.60,  # relaxed open (a, i)
    "H": 0.80,  # wide open (short a, e)
}

VISEME_EXP: dict[str, list[float]] = {
    "X": _EXP_X,
    "A": _EXP_A,
    "B": _EXP_B,
    "C": _EXP_C,
    "D": _EXP_D,
    "E": _EXP_E,
    "F": _EXP_F,
    "G": _EXP_G,
    "H": _EXP_H,
}


def viseme_exp(viseme: str) -> list[float]:
    """Return the 63-d exp array for the given viseme label (falls back to X)."""
    return VISEME_EXP.get(viseme, _EXP_X)


def viseme_lip(viseme: str) -> float:
    """Return the lip-open ratio for the given viseme label (falls back to 0.0)."""
    return VISEME_LIP.get(viseme, 0.0)


# =====================================
# Mouth shape for speaking
# =====================================

def mouth(
    viseme_sequence: list[dict],
    fps: int = 25,
) -> list[Keyframe]:
    """
    Convert a timed viseme sequence into a list of Keyframes ready for build_template().

    Each entry in viseme_sequence must have "start" (seconds), "end" (seconds),
    and "viseme" (one of X A B C D E F G H).
    """
    kf_map: dict[int, Keyframe] = {}

    for seg in viseme_sequence:
        start_frame = round(seg["start"] * fps)
        end_frame   = round(seg["end"]   * fps)
        exp         = viseme_exp(seg["viseme"])
        lip         = viseme_lip(seg["viseme"])

        for frame in (start_frame, end_frame):
            if frame not in kf_map:
                kf_map[frame] = Keyframe(frame_idx=frame, exp=exp, lip=lip)

    return sorted(kf_map.values(), key=lambda k: k.frame)
