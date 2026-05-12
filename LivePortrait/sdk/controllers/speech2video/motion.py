import json
import random
import numpy as np

_EXP_DIM = 63  # 21 keypoints × 3 coords


class Keyframe:
    def __init__(self, frame_idx: int, blink: dict = None, mouth: dict = None):
        self.frame = frame_idx
        self.exp   = [0.0] * _EXP_DIM
        if blink:
            for eye_idx, value in blink.items():
                self.exp[eye_idx * 3 + 1] = value
        if mouth:
            for kp_idx, (dx, dy, dz) in mouth.items():
                b = kp_idx * 3
                self.exp[b], self.exp[b + 1], self.exp[b + 2] = dx, dy, dz

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "tx": 0.0, "ty": 0.0, "scale": 1.0,
            "exp": self.exp,
            "eyes": [],
            "lip": 0.0,
        }


def neutral_keyframes(seconds: float, fps: int = 25) -> list["Keyframe"]:
    total_frames = int(seconds * fps)
    return [Keyframe(frame_idx=i) for i in range(total_frames)]


def add_blinks(keyframes: list["Keyframe"], fps: int = 25) -> list["Keyframe"]:
    from sdk.controllers.speech2video.templates.blinks import blink  # local import — blinks.py imports Keyframe from here

    total_frames = len(keyframes)
    start_frame  = int(0.5 * fps)

    while True:
        blink_kfs, end_frame = blink(start_frame)
        if end_frame >= total_frames:
            break

        kf_frames = np.array([kf.frame for kf in blink_kfs], dtype=float)
        for eye_idx in [11, 15]:
            exp_col = eye_idx * 3 + 1
            kf_vals = np.array([kf.exp[exp_col] for kf in blink_kfs], dtype=float)
            for i in range(start_frame, end_frame + 1):
                keyframes[i].exp[exp_col] = float(np.interp(i, kf_frames, kf_vals))

        start_frame += random.randint(4, 6) * fps

    return keyframes



_MOUTH_KP_INDICES = [6, 12, 14, 17, 19, 20]


def add_talks(keyframes: list["Keyframe"], viseme, fps: int = 25) -> list["Keyframe"]:
    from sdk.controllers.speech2video.templates.mouth import mouth_shape

    formatted_viseme = visemes_to_frames(viseme)
    mouth_kfs, _end = mouth_shape(formatted_viseme)

    kf_frames   = np.array([kf.frame for kf in mouth_kfs], dtype=float)
    start_frame = int(kf_frames[0])
    end_frame   = min(_end, len(keyframes) - 1)

    for kp_idx in _MOUTH_KP_INDICES:
        for coord in range(3):
            exp_col = kp_idx * 3 + coord
            kf_vals = np.array([kf.exp[exp_col] for kf in mouth_kfs], dtype=float)
            for i in range(start_frame, end_frame + 1):
                keyframes[i].exp[exp_col] = float(np.interp(i, kf_frames, kf_vals))

    return keyframes


def euler_to_R(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    p, y, r = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg), np.deg2rad(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p),  np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T


def build_template(keyframes: list[dict], fps: int) -> dict:
    keyframes = sorted(keyframes, key=lambda k: k["frame"])
    src_t = np.array([k["frame"] for k in keyframes], dtype=float)
    n     = int(src_t[-1]) + 1
    dst_t = np.arange(n, dtype=float)

    def dense(field, default):
        vals = np.array([k.get(field, default) for k in keyframes], dtype=float)
        return np.interp(dst_t, src_t, vals)

    pitch = dense("pitch", 0.0)
    yaw   = dense("yaw",   0.0)
    roll  = dense("roll",  0.0)
    t_x   = dense("tx",    0.0)
    t_y   = dense("ty",    0.0)
    scale = dense("scale", 1.0)
    lip   = dense("lip",   0.0)

    exp = np.array([k.get("exp", [0.0] * _EXP_DIM) for k in keyframes], dtype=np.float32)

    eyes = np.full((n, 2), 0.4, dtype=float)

    motion = [
        {
            "scale": np.array([[scale[i]]], dtype=np.float32),
            "R":     euler_to_R(pitch[i], yaw[i], roll[i])[np.newaxis].astype(np.float32),
            "exp":   exp[i].reshape(1, 21, 3).astype(np.float32),
            "t":     np.array([[t_x[i], t_y[i], 0.0]], dtype=np.float32),
            "kp":    np.zeros((1, 21, 3), dtype=np.float32),
            "x_s":   np.zeros((1, 21, 3), dtype=np.float32),
        }
        for i in range(n)
    ]

    return {
        "n_frames":   n,
        "output_fps": fps,
        "motion":     motion,
        "c_eyes_lst": [eyes[i : i + 1].astype(np.float32) for i in range(n)],
        "c_lip_lst":  [np.array([[lip[i]]], dtype=np.float32) for i in range(n)],
    }


def visemes_to_frames(visemes: list[dict], fps: int = 25) -> list[dict]:
    result = []
    for v in visemes:
        n_frames = max(1, round(v["duration"] * fps))
        result.append({"viseme": v["viseme"], "frames": n_frames})
    return result


def template_to_json(template: dict) -> dict:
    def _serial(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    return json.loads(json.dumps({
        "n_frames":   template["n_frames"],
        "output_fps": template["output_fps"],
        "motion": [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()}
            for m in template["motion"]
        ],
        "c_eyes_lst": [e.tolist() for e in template["c_eyes_lst"]],
        "c_lip_lst":  [l.tolist() for l in template["c_lip_lst"]],
    }, default=_serial))
