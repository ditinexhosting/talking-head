from api.utils.motion import Keyframe


#---- Neutral -----#
natural_keyframes = [
    (0,  [0.4, 0.4]),
    (2,  [0.4, 0.4]),
    (5,  [0.0, 0.0]),
    (8, [0.4, 0.4]),
    (10, [0.4, 0.4]),
]

#---- Thinking -----#
thinking_keyframes = [
    (0,  [0.4, 0.4]),
    (2,  [0.4, 0.4]),
    (5,  [0.0, 0.0]),
    (7,  [0.0, 0.0]),
    (10, [0.4, 0.4]),
    (12, [0.4, 0.4]),
]

# =====================================
# Blink
# =====================================
def blink(start_frame: int, type: str = "neutral"):
    match type:
        case "thinking":
            keyframes = thinking_keyframes
        case _:
            keyframes = natural_keyframes

    animated_keyframes = [
        Keyframe(frame_idx=start_frame + frame, eye=eye) for frame, eye in keyframes
    ]
    end_frame = start_frame + keyframes[-1][0]

    return animated_keyframes, end_frame
