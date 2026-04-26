# coding: utf-8

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field


def _rotation_matrix_np(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Compute (1,3,3) rotation matrix from Euler angles in degrees (mirrors camera.py)."""
    p, y, r = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p),  np.cos(p)]], dtype=np.float32)
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]], dtype=np.float32)
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r),  np.cos(r), 0],
                   [0, 0, 1]], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return R.T[np.newaxis]  # transpose then add batch dim, matching camera.py


def build_motion_template(n_frames: int = 150, fps: int = 25) -> dict:
    """
    Build a synthetic n_frames motion template with hardcoded per-frame animation.

    Animation timeline (t = frame / (n_frames-1), range 0..1):
      0.00–0.20  Neutral hold
      0.20–0.40  Head nod down (-5°) and back
      0.35–0.40  Eye blink #1
      0.40–0.60  Head turn left then right (yaw ±8°)
      0.60–0.85  Talking (3 open/close mouth cycles)
      0.72–0.77  Eye blink #2
      0.85–1.00  Return to neutral
    """
    SCALE = np.array([[1.393]], dtype=np.float32)
    T_NEUTRAL = np.array([[0.0, 0.10, 0.0]], dtype=np.float32)
    EYE_NEUTRAL = 0.30
    LIP_IDX = [6, 12, 14, 17, 19, 20]
    EYE_IDX = [11, 13, 15, 16, 18]

    def blink_amount(t, center, half_width=0.025):
        d = abs(t - center)
        return float(max(0.0, 1.0 - d / half_width))

    motion = []
    c_eyes_lst = []
    c_lip_lst = []

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)

        # ---- head pose ----
        pitch = yaw = roll = 0.0

        if 0.20 <= t < 0.40:
            local = (t - 0.20) / 0.20
            pitch = -5.0 * float(np.sin(local * np.pi))

        elif 0.40 <= t < 0.60:
            local = (t - 0.40) / 0.20
            yaw = -8.0 * float(np.sin(local * np.pi * 1.5)) * (1.0 - local * 0.4)

        pitch += 0.8 * float(np.sin(t * 2 * np.pi * 0.6))

        R = _rotation_matrix_np(pitch, yaw, roll)

        # ---- expression deltas ----
        exp = np.zeros((1, 21, 3), dtype=np.float32)

        lip_open = 0.0
        if 0.60 <= t < 0.85:
            local = (t - 0.60) / 0.25
            lip_open = float(max(0.0, np.sin(local * np.pi * 6.0))) * 0.020

        weights = [0.5, 1.0, 0.7, 0.4, -0.6, -0.6]
        for idx, w in zip(LIP_IDX, weights):
            exp[0, idx, 1] += lip_open * w

        blink = max(blink_amount(t, 0.37), blink_amount(t, 0.74))
        eye_weights = [0.9, 0.7, 0.9, 0.7, 0.6]
        for idx, w in zip(EYE_IDX, eye_weights):
            exp[0, idx, 1] -= blink * 0.012 * w

        # ---- eye/lip openness ratios ----
        eye_open = float(np.clip(EYE_NEUTRAL - blink * 0.28, 0.01, 0.37))
        c_eyes = np.array([[eye_open, eye_open]], dtype=np.float32)
        c_lip = np.array([[lip_open * 20.0]], dtype=np.float32)

        motion.append({
            'scale': SCALE.copy(),
            'R': R,
            'exp': exp,
            't': T_NEUTRAL.copy(),
            'kp': np.zeros((1, 21, 3), dtype=np.float32),
            'x_s': np.zeros((1, 21, 3), dtype=np.float32),
        })
        c_eyes_lst.append(c_eyes)
        c_lip_lst.append(c_lip)

    return {
        'n_frames': n_frames,
        'output_fps': fps,
        'motion': motion,
        'c_eyes_lst': c_eyes_lst,
        'c_lip_lst': c_lip_lst,
    }


@dataclass
class Keyframe:
    """Single keyframe specification for animation with detailed facial control.

    Keypoint indices (0-20 of the 21-point model):
      Eyes: 11, 13, 15, 16, 18
      Eyebrows: 0, 1, 2, 3, 4, 5 (left & right eyebrows)
      Mouth: 6, 12, 14, 17, 19, 20
      Nose: 7, 8, 9, 10

    Head pose & scale:
        pitch_deg: Head nod rotation (-90 to 90)
        yaw_deg: Head left/right rotation (-180 to 180)
        roll_deg: Head tilt rotation (-90 to 90)
        scale: Face size (default 1.393)
        tx, ty, tz: Translation offsets (default 0.0, 0.10, 0.0)

    Eye control:
        eye_open_left: Left eye openness, 0.0-1.0 (0=closed, 1=wide open)
        eye_open_right: Right eye openness, 0.0-1.0
        eye_gaze_x: Horizontal gaze direction, -1.0 (left) to 1.0 (right)
        eye_gaze_y: Vertical gaze direction, -1.0 (down) to 1.0 (up)

    Eyebrow control:
        brow_raise_left: Left eyebrow raise, -1.0 (down) to 1.0 (up)
        brow_raise_right: Right eyebrow raise, -1.0 (down) to 1.0 (up)
        brow_inner_raise: Inner brow raise, -1.0 to 1.0

    Mouth control:
        mouth_open: Mouth openness, 0.0-1.0 (0=closed, 1=wide open)
        mouth_smile: Smile intensity, -1.0 (frown) to 1.0 (big smile)
        mouth_pucker: Lip pucker amount, -1.0 to 1.0
        mouth_x: Mouth horizontal shift, -1.0 to 1.0
        mouth_y: Mouth vertical shift, -1.0 to 1.0

    Nose control:
        nose_x: Nostril flare, -1.0 to 1.0
        nose_y: Nose wrinkle, -1.0 to 1.0

    Raw expression control:
        exp_deltas: Optional direct 21x3 keypoint displacement array
                    Overrides all facial parameters if provided
    """
    frame_idx: int

    # Head pose & scale
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    scale: float = 1.393
    tx: float = 0.0
    ty: float = 0.10
    tz: float = 0.0

    # Eye control
    eye_open_left: float = 0.30
    eye_open_right: float = 0.30
    eye_gaze_x: float = 0.0  # -1=left, 0=center, 1=right
    eye_gaze_y: float = 0.0  # -1=down, 0=center, 1=up

    # Eyebrow control
    brow_raise_left: float = 0.0    # -1=down, 0=neutral, 1=up
    brow_raise_right: float = 0.0
    brow_inner_raise: float = 0.0
    brow_angle_left: float = 0.0    # -1=down, 1=up
    brow_angle_right: float = 0.0

    # Mouth control
    mouth_open: float = 0.0         # 0=closed, 1=wide open
    mouth_smile: float = 0.0        # -1=frown, 0=neutral, 1=smile
    mouth_pucker: float = 0.0       # -1=pucker, 0=neutral, 1=unpucker
    mouth_x: float = 0.0            # horizontal shift
    mouth_y: float = 0.0            # vertical shift

    # Nose control
    nose_x: float = 0.0             # nostril flare
    nose_y: float = 0.0             # wrinkle

    # Raw keypoint override
    exp_deltas: np.ndarray = field(default_factory=lambda: np.zeros((21, 3), dtype=np.float32))


def _build_expression_from_params(kf: Keyframe) -> np.ndarray:
    """Convert semantic facial parameters to 21x3 expression deltas.

    Args:
        kf: Keyframe with facial parameters

    Returns:
        Expression array of shape (21, 3) with keypoint displacements
    """
    exp = kf.exp_deltas.copy() if kf.exp_deltas.any() else np.zeros((21, 3), dtype=np.float32)

    # If raw exp_deltas are provided, use them directly
    if exp.any():
        return exp

    # Eye indices: 11, 13, 15, 16, 18
    EYE_L_TOP, EYE_R_TOP, EYE_L_BOT, EYE_R_BOT, EYE_INNER = [11, 13, 15, 16, 18]

    # Eyebrow indices: 0, 1, 2, 3, 4, 5 (left: 0-2, right: 3-5)
    BROW_L_INNER, BROW_L_MID, BROW_L_OUTER = [0, 1, 2]
    BROW_R_INNER, BROW_R_MID, BROW_R_OUTER = [3, 4, 5]

    # Mouth indices: 6, 12, 14, 17, 19, 20
    MOUTH_TOP, MOUTH_BOT, MOUTH_L_CORNER, MOUTH_R_CORNER, MOUTH_L, MOUTH_R = [6, 12, 14, 17, 19, 20]

    # Nose indices: 7, 8, 9, 10
    NOSE_TIP, NOSE_L, NOSE_R, NOSE_BOT = [7, 8, 9, 10]

    # --- Eye movement (gaze) ---
    if kf.eye_gaze_x != 0.0 or kf.eye_gaze_y != 0.0:
        gaze_x = kf.eye_gaze_x * 0.015
        gaze_y = kf.eye_gaze_y * 0.010

        # Both eyes move together for gaze
        exp[[EYE_L_TOP, EYE_R_TOP, EYE_L_BOT, EYE_R_BOT], 0] += gaze_x
        exp[[EYE_L_TOP, EYE_R_TOP, EYE_L_BOT, EYE_R_BOT], 1] += gaze_y

    # --- Eye opening/closing ---
    # Left eye
    if kf.eye_open_left < 0.30:
        blink_left = (0.30 - kf.eye_open_left) * 0.015
        exp[[EYE_L_TOP], 1] -= blink_left
        exp[[EYE_L_BOT], 1] += blink_left

    # Right eye
    if kf.eye_open_right < 0.30:
        blink_right = (0.30 - kf.eye_open_right) * 0.015
        exp[[EYE_R_TOP], 1] -= blink_right
        exp[[EYE_R_BOT], 1] += blink_right

    # --- Eyebrow control ---
    # Left eyebrow raise
    if kf.brow_raise_left != 0.0:
        brow_raise = kf.brow_raise_left * 0.012
        exp[[BROW_L_INNER, BROW_L_MID, BROW_L_OUTER], 1] += brow_raise

    # Right eyebrow raise
    if kf.brow_raise_right != 0.0:
        brow_raise = kf.brow_raise_right * 0.012
        exp[[BROW_R_INNER, BROW_R_MID, BROW_R_OUTER], 1] += brow_raise

    # Eyebrow angle/tilt
    if kf.brow_angle_left != 0.0:
        angle = kf.brow_angle_left * 0.010
        exp[BROW_L_OUTER, 0] += angle
        exp[BROW_L_INNER, 0] -= angle

    if kf.brow_angle_right != 0.0:
        angle = kf.brow_angle_right * 0.010
        exp[BROW_R_INNER, 0] -= angle
        exp[BROW_R_OUTER, 0] += angle

    # --- Mouth control ---
    # Mouth opening
    if kf.mouth_open > 0.0:
        mouth_open = kf.mouth_open * 0.020
        exp[MOUTH_TOP, 1] -= mouth_open
        exp[MOUTH_BOT, 1] += mouth_open

    # Smile
    if kf.mouth_smile > 0.0:
        smile = kf.mouth_smile * 0.015
        exp[[MOUTH_L_CORNER, MOUTH_R_CORNER], 1] += smile * 0.5
        exp[[MOUTH_L_CORNER, MOUTH_R_CORNER], 0] += np.array([smile, -smile])
    elif kf.mouth_smile < 0.0:
        frown = -kf.mouth_smile * 0.015
        exp[[MOUTH_L_CORNER, MOUTH_R_CORNER], 1] -= frown * 0.3

    # Mouth pucker
    if kf.mouth_pucker > 0.0:
        pucker = kf.mouth_pucker * 0.010
        exp[[MOUTH_TOP, MOUTH_BOT, MOUTH_L, MOUTH_R], :] -= pucker

    # Mouth horizontal/vertical shift
    if kf.mouth_x != 0.0:
        mouth_x = kf.mouth_x * 0.012
        exp[[MOUTH_TOP, MOUTH_BOT, MOUTH_L, MOUTH_R, MOUTH_L_CORNER, MOUTH_R_CORNER], 0] += mouth_x

    if kf.mouth_y != 0.0:
        mouth_y = kf.mouth_y * 0.012
        exp[[MOUTH_TOP, MOUTH_BOT, MOUTH_L, MOUTH_R, MOUTH_L_CORNER, MOUTH_R_CORNER], 1] += mouth_y

    # --- Nose control ---
    if kf.nose_x != 0.0:
        nostril = kf.nose_x * 0.008
        exp[NOSE_L, 0] -= nostril
        exp[NOSE_R, 0] += nostril

    if kf.nose_y != 0.0:
        wrinkle = kf.nose_y * 0.008
        exp[[NOSE_L, NOSE_R], 1] += wrinkle

    return exp


def _interpolate_keyframes(keyframes: List[Keyframe], n_frames: int) -> List[Dict[str, Any]]:
    """Interpolate between keyframes to generate per-frame motion data.

    Args:
        keyframes: List of Keyframe objects in ascending frame_idx order
        n_frames: Total number of frames in output

    Returns:
        List of motion dicts, one per frame
    """
    if not keyframes:
        raise ValueError("At least one keyframe required")

    keyframes = sorted(keyframes, key=lambda k: k.frame_idx)
    motion = []

    for frame in range(n_frames):
        # Find surrounding keyframes
        kf_before = keyframes[0]
        kf_after = keyframes[-1]

        for i, kf in enumerate(keyframes):
            if kf.frame_idx <= frame:
                kf_before = kf
            if kf.frame_idx >= frame and kf_after == keyframes[-1]:
                kf_after = kf

        # Linear interpolation weight
        if kf_before.frame_idx == kf_after.frame_idx:
            alpha = 0.0
        else:
            alpha = (frame - kf_before.frame_idx) / (kf_after.frame_idx - kf_before.frame_idx)
            alpha = np.clip(alpha, 0.0, 1.0)

        # Interpolate all parameters
        pitch = kf_before.pitch_deg + alpha * (kf_after.pitch_deg - kf_before.pitch_deg)
        yaw = kf_before.yaw_deg + alpha * (kf_after.yaw_deg - kf_before.yaw_deg)
        roll = kf_before.roll_deg + alpha * (kf_after.roll_deg - kf_before.roll_deg)

        scale = kf_before.scale + alpha * (kf_after.scale - kf_before.scale)
        tx = kf_before.tx + alpha * (kf_after.tx - kf_before.tx)
        ty = kf_before.ty + alpha * (kf_after.ty - kf_before.ty)
        tz = kf_before.tz + alpha * (kf_after.tz - kf_before.tz)

        # Interpolate eye parameters
        eye_open_left = kf_before.eye_open_left + alpha * (kf_after.eye_open_left - kf_before.eye_open_left)
        eye_open_left = np.clip(eye_open_left, 0.01, 1.0)

        eye_open_right = kf_before.eye_open_right + alpha * (kf_after.eye_open_right - kf_before.eye_open_right)
        eye_open_right = np.clip(eye_open_right, 0.01, 1.0)

        eye_gaze_x = kf_before.eye_gaze_x + alpha * (kf_after.eye_gaze_x - kf_before.eye_gaze_x)
        eye_gaze_y = kf_before.eye_gaze_y + alpha * (kf_after.eye_gaze_y - kf_before.eye_gaze_y)

        # Interpolate eyebrow parameters
        brow_raise_left = kf_before.brow_raise_left + alpha * (kf_after.brow_raise_left - kf_before.brow_raise_left)
        brow_raise_right = kf_before.brow_raise_right + alpha * (kf_after.brow_raise_right - kf_before.brow_raise_right)
        brow_inner_raise = kf_before.brow_inner_raise + alpha * (kf_after.brow_inner_raise - kf_before.brow_inner_raise)
        brow_angle_left = kf_before.brow_angle_left + alpha * (kf_after.brow_angle_left - kf_before.brow_angle_left)
        brow_angle_right = kf_before.brow_angle_right + alpha * (kf_after.brow_angle_right - kf_before.brow_angle_right)

        # Interpolate mouth parameters
        mouth_open = kf_before.mouth_open + alpha * (kf_after.mouth_open - kf_before.mouth_open)
        mouth_open = np.clip(mouth_open, 0.0, 1.0)

        mouth_smile = kf_before.mouth_smile + alpha * (kf_after.mouth_smile - kf_before.mouth_smile)
        mouth_smile = np.clip(mouth_smile, -1.0, 1.0)

        mouth_pucker = kf_before.mouth_pucker + alpha * (kf_after.mouth_pucker - kf_before.mouth_pucker)
        mouth_x = kf_before.mouth_x + alpha * (kf_after.mouth_x - kf_before.mouth_x)
        mouth_y = kf_before.mouth_y + alpha * (kf_after.mouth_y - kf_before.mouth_y)

        # Interpolate nose parameters
        nose_x = kf_before.nose_x + alpha * (kf_after.nose_x - kf_before.nose_x)
        nose_y = kf_before.nose_y + alpha * (kf_after.nose_y - kf_before.nose_y)

        # Create interpolated keyframe for expression building
        interp_kf = Keyframe(
            frame_idx=frame,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            scale=scale,
            tx=tx,
            ty=ty,
            tz=tz,
            eye_open_left=eye_open_left,
            eye_open_right=eye_open_right,
            eye_gaze_x=eye_gaze_x,
            eye_gaze_y=eye_gaze_y,
            brow_raise_left=brow_raise_left,
            brow_raise_right=brow_raise_right,
            brow_inner_raise=brow_inner_raise,
            brow_angle_left=brow_angle_left,
            brow_angle_right=brow_angle_right,
            mouth_open=mouth_open,
            mouth_smile=mouth_smile,
            mouth_pucker=mouth_pucker,
            mouth_x=mouth_x,
            mouth_y=mouth_y,
            nose_x=nose_x,
            nose_y=nose_y,
        )

        # Build expression from interpolated parameters
        exp_deltas = _build_expression_from_params(interp_kf)
        exp = np.zeros((1, 21, 3), dtype=np.float32)
        exp[0] = exp_deltas

        R = _rotation_matrix_np(pitch, yaw, roll)

        motion.append({
            'scale': np.array([[scale]], dtype=np.float32),
            'R': R,
            'exp': exp,
            't': np.array([[tx, ty, tz]], dtype=np.float32),
            'kp': np.zeros((1, 21, 3), dtype=np.float32),
            'x_s': np.zeros((1, 21, 3), dtype=np.float32),
        })

    return motion


def build_motion_from_keyframes(keyframes: List[Keyframe], n_frames: int, fps: int = 25) -> dict:
    """Build a motion template from keyframes with smooth interpolation.

    Args:
        keyframes: List of Keyframe objects specifying animation at key points
        n_frames: Total number of frames to generate
        fps: Output frames per second

    Returns:
        Motion template dict compatible with LivePortraitPipeline

    Example:
        keyframes = [
            Keyframe(frame_idx=0, pitch_deg=0, yaw_deg=0, eye_open=0.30, lip_open=0.0),
            Keyframe(frame_idx=25, pitch_deg=-5, yaw_deg=0, eye_open=0.10, lip_open=0.0),  # nod
            Keyframe(frame_idx=50, pitch_deg=0, yaw_deg=8, eye_open=0.30, lip_open=0.3),   # turn & smile
            Keyframe(frame_idx=75, pitch_deg=0, yaw_deg=0, eye_open=0.30, lip_open=0.0),
        ]
        motion_dict = build_motion_from_keyframes(keyframes, n_frames=100, fps=25)
    """
    motion = _interpolate_keyframes(keyframes, n_frames)

    c_eyes_lst = []
    c_lip_lst = []

    for frame_motion in motion:
        # Extract eye/lip openness from motion data
        eye_open = np.clip(frame_motion['t'][0, 0], 0.01, 0.37)  # Recompute from interpolation
        c_eyes_lst.append(np.array([[eye_open, eye_open]], dtype=np.float32))

        # Extract lip open amount (approximated from exp deltas on lip keypoints)
        lip_open = np.clip(np.abs(frame_motion['exp'][0, 6:7, 1]).max(), 0.0, 1.0) * 20.0
        c_lip_lst.append(np.array([[lip_open]], dtype=np.float32))

    return {
        'n_frames': n_frames,
        'output_fps': fps,
        'motion': motion,
        'c_eyes_lst': c_eyes_lst,
        'c_lip_lst': c_lip_lst,
    }
