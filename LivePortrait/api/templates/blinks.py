from api.utils.motion import Keyframe


#---- Neutral -----#
natural_keyframes = [
    (0,  {11: 0, 15: 0}),
    (3,  {11: 0.016, 15: 0.016}),
    (6,  {11: 0, 15: 0})
]

#---- Thinking -----#
thinking_keyframes = [
    (0,  {11: 0, 15: 0}),
    (3,  {11: 0.016, 15: 0.016}),
    (4,  {11: 0.016, 15: 0.016}),
    (7,  {11: 0, 15: 0})
]

# =====================================
# Blink
# Eye point indices :
# 11 - Left eye
# 15 - Right eye
# =====================================
def blink(start_frame: int, type: str = "neutral"):
    match type:
        case "thinking":
            keyframes = thinking_keyframes
        case _:
            keyframes = natural_keyframes

    animated_keyframes = [
        Keyframe(frame_idx=start_frame + frame, blink=blink) for frame, blink in keyframes
    ]
    end_frame = start_frame + keyframes[-1][0]

    return animated_keyframes, end_frame
