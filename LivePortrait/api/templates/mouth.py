from api.utils.motion import Keyframe, _EXP_DIM

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

# E — Slightly rounded mouth
_EXP_E = {
    6:  (0.00,  0.02, 0.00),
    12: (0.00,  0.02, 0.00),
    14: (0.00,  0.008, 0.00),
    17: (0.00,  -0.008, 0.00),
    19: (0.00, 0.025, 0.008), # Bottom lip
    20: (0.00, -0.008, 0.008), # Top lip
}

# F — Puckered lips
_EXP_F = {
    6:  (0.00,  0.02, 0.00),
    12: (0.00,  0.02, 0.00),
    14: (0.00,  0.01, 0.00),
    17: (0.00,  -0.01, 0.00),
    19: (0.00, 0.025, 0.02), # Bottom lip
    20: (0.00, -0.008, 0.02), # Top lip
}


# G — Upper teeth touching the lower lip
_EXP_G = {
    6:  (0.00,  0.02, 0.00),
    12: (0.00,  0.02, 0.00),
    14: (0.00,  0.007, 0.00),
    17: (0.00,  -0.007, 0.00),
    19: (0.00, -0.012, -0.008), # Bottom lip
    20: (0.00, -0.01, 0.00), # Top lip
}

# H — This shape is used for long “L” sounds, with the tongue raised behind the upper teeth.
_EXP_H = {
    6:  (0.00,  0.02, 0.00),
    12: (0.00,  0.02, 0.00),
    14: (0.00,  -0.00, 0.00),
    17: (0.00,  0.00, 0.00),
    19: (0.00, 0.02, -0.010), # Bottom lip
    20: (0.00, -0.008, 0.00), # Top lip
}

# X — Silence / rest
_EXP_X = {
    6:  (0.00,  0.00, 0.00),
    12: (0.00,  0.00, 0.00),
    14: (0.00,  0.00, 0.00),
    17: (0.00,  0.00, 0.00),
    19: (0.00,  0.00, 0.00), # Bottom lip
    20: (0.00,  0.00, 0.00), # Top lip
}



_VISEME_MAP = {
    'A': _EXP_A,
    'B': _EXP_B,
    'C': _EXP_C,
    'D': _EXP_D,
    'E': _EXP_E,
    'F': _EXP_F,
    'G': _EXP_G,
    'H': _EXP_H,
    'X': _EXP_X,
}


def mouth_shape(viseme_sequence: list[dict]) -> tuple[list[Keyframe], int]:
    frame_idx = 1
    keyframes = [
        Keyframe(frame_idx=0, mouth=_EXP_X),
    ]
    for item in viseme_sequence:
        exp = _VISEME_MAP.get(item['viseme'], _EXP_X)
        keyframes.append(Keyframe(frame_idx=frame_idx, mouth=exp))
        if item['viseme'] == 'X' and frame_idx + item['frames'] - 2 > frame_idx:
            rest_end_frame = frame_idx + item['frames'] - 2
            keyframes.append(Keyframe(frame_idx=rest_end_frame, mouth=_EXP_X))

        frame_idx += item['frames']

    keyframes.append(Keyframe(frame_idx=frame_idx, mouth=_EXP_X))

    end_frame = frame_idx-1
    return keyframes, end_frame
