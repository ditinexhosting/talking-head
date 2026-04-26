# coding: utf-8
"""
MuseTalk lip-sync integration for LivePortrait.

Uses face_alignment instead of mmpose/DWPose to avoid heavy OpenMMLab
dependencies. The landmark indexing follows the same 68-point convention so
results are equivalent.
"""

import copy
import glob
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

MUSETALK_ROOT = Path("/workspace/MuseTalk")
MUSETALK_MODELS = MUSETALK_ROOT / "models"
sys.path.insert(0, str(MUSETALK_ROOT))

# Lazy singletons — loaded once on first call to apply_lipsync
_fa = None          # face_alignment.FaceAlignment
_vae = None
_unet = None
_pe = None
_audio_processor = None
_whisper = None
_fp = None          # FaceParsing

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


def _get_face_alignment():
    global _fa
    if _fa is None:
        import face_alignment
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=_DEVICE,
        )
    return _fa


def _load_musetalk_models():
    """Load VAE, UNet, PositionalEncoding, Whisper, FaceParsing once."""
    global _vae, _unet, _pe, _audio_processor, _whisper, _fp

    if _vae is not None:
        return

    from musetalk.models.vae import VAE
    from musetalk.models.unet import UNet, PositionalEncoding
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from transformers import WhisperModel

    vae_path = str(MUSETALK_MODELS / "sd-vae")
    unet_cfg = str(MUSETALK_MODELS / "musetalkV15/musetalk.json")
    unet_pth = str(MUSETALK_MODELS / "musetalkV15/unet.pth")
    whisper_dir = str(MUSETALK_MODELS / "whisper")

    print("[MuseTalk] Loading VAE...")
    _vae = VAE(model_path=vae_path)

    print("[MuseTalk] Loading UNet...")
    device = torch.device(_DEVICE)
    _unet = UNet(unet_config=unet_cfg, model_path=unet_pth, device=device)
    _unet.model = _unet.model.half().to(device)

    _pe = PositionalEncoding(d_model=384).half().to(device)

    print("[MuseTalk] Loading Whisper...")
    _audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    _whisper = WhisperModel.from_pretrained(whisper_dir)
    _whisper = _whisper.to(device=device, dtype=torch.float16).eval()
    _whisper.requires_grad_(False)

    print("[MuseTalk] Loading FaceParsing...")
    resnet_path = str(MUSETALK_MODELS / "face-parse-bisent/resnet18-5c106cde.pth")
    bisenet_path = str(MUSETALK_MODELS / "face-parse-bisent/79999_iter.pth")
    # FaceParsing.__init__ calls model_init() with CWD-relative defaults;
    # we restore CWD afterwards so LivePortrait's own paths are unaffected.
    _prev_cwd = os.getcwd()
    try:
        os.chdir(str(MUSETALK_ROOT))
        _fp = FaceParsing()
        _fp.net = _fp.model_init(resnet_path=resnet_path, model_pth=bisenet_path)
    finally:
        os.chdir(_prev_cwd)

    # Move VAE to device and half-precision after loading
    _vae.vae = _vae.vae.half().to(device)
    print("[MuseTalk] Models ready.")


def _get_face_bbox_for_frame(frame_bgr: np.ndarray, fa) -> tuple:
    """
    Return (x1, y1, x2, y2) face crop for MuseTalk using 68-point landmarks.

    Upper boundary y1 is set to landmark[29] (nose bridge), matching the
    DWPose-based preprocessing behaviour used by the original MuseTalk code.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    landmarks_list = fa.get_landmarks(frame_rgb)
    if not landmarks_list:
        return _COORD_PLACEHOLDER

    lm = landmarks_list[0]  # (68, 2)

    # Horizontal extent from full face width
    x1 = int(np.min(lm[:, 0]))
    x2 = int(np.max(lm[:, 0]))

    # Upper boundary: nose bridge (landmark 29, same as DWPose face_land_mark[29])
    y_upper = int(lm[29, 1])

    # Lower boundary: lowest jaw point
    y_lower = int(np.max(lm[0:17, 1]))

    h, w = frame_bgr.shape[:2]
    x1 = max(0, x1)
    x2 = min(w, x2)
    y_upper = max(0, y_upper)
    y_lower = min(h, y_lower)

    if x2 <= x1 or y_lower <= y_upper:
        return _COORD_PLACEHOLDER

    return (x1, y_upper, x2, y_lower)


def _extract_frames(video_path: str, out_dir: str) -> Tuple[list, float]:
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f"ffmpeg -y -v fatal -i {video_path} "
        f"-start_number 0 {out_dir}/%08d.png"
    )
    os.system(cmd)
    imgs = sorted(glob.glob(os.path.join(out_dir, "*.png")))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return imgs, fps


def _datagen(whisper_chunks, vae_latents, batch_size=8):
    latent_cycle = vae_latents + vae_latents[::-1]
    w_buf, l_buf = [], []
    for i, w in enumerate(whisper_chunks):
        lat = latent_cycle[i % len(latent_cycle)]
        w_buf.append(w)
        l_buf.append(lat)
        if len(w_buf) >= batch_size:
            yield torch.stack(w_buf), torch.cat(l_buf, dim=0)
            w_buf, l_buf = [], []
    if w_buf:
        yield torch.stack(w_buf), torch.cat(l_buf, dim=0)


def apply_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    extra_margin: int = 10,
    batch_size: int = 8,
) -> str:
    """
    Apply MuseTalk lip-sync to *video_path* driven by *audio_path*.

    Parameters
    ----------
    video_path:  input MP4 (generated by LivePortrait)
    audio_path:  WAV or MP3 audio file containing the speech
    output_path: where to write the final MP4
    extra_margin: pixels added to the bottom of the face crop (MuseTalk v1.5 default = 10)
    batch_size:  UNet batch size

    Returns
    -------
    output_path (str)
    """
    _load_musetalk_models()
    fa = _get_face_alignment()

    device = torch.device(_DEVICE)
    dtype = torch.float16

    # ── 1. Extract frames ──────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="musetalk_") as tmpdir:
        frames_dir = os.path.join(tmpdir, "frames")
        result_dir = os.path.join(tmpdir, "result")
        os.makedirs(result_dir, exist_ok=True)

        img_paths, fps = _extract_frames(video_path, frames_dir)
        if not img_paths:
            raise RuntimeError(f"No frames extracted from {video_path}")

        frames = [cv2.imread(p) for p in img_paths]
        print(f"[MuseTalk] {len(frames)} frames at {fps:.1f} fps")

        # ── 2. Face detection ──────────────────────────────────────────────
        print("[MuseTalk] Detecting faces...")
        coord_list = []
        for frame in tqdm(frames):
            coord_list.append(_get_face_bbox_for_frame(frame, fa))

        # ── 3. Encode face crops with VAE ──────────────────────────────────
        print("[MuseTalk] VAE encoding...")
        vae_latents = []
        for bbox, frame in tqdm(zip(coord_list, frames), total=len(frames)):
            if bbox == _COORD_PLACEHOLDER:
                # Use blank latent for frames with no detected face
                blank = torch.zeros(1, 8, 32, 32, device=device, dtype=dtype)
                vae_latents.append(blank)
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            y2 = min(y2 + extra_margin, frame.shape[0])
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            vae_latents.append(_vae.get_latents_for_unet(crop))

        # ── 4. Extract Whisper audio features ─────────────────────────────
        print("[MuseTalk] Extracting audio features...")
        whisper_features, librosa_len = _audio_processor.get_audio_feature(audio_path)
        whisper_chunks = _audio_processor.get_whisper_chunk(
            whisper_features,
            device,
            dtype,
            _whisper,
            librosa_len,
            fps=fps,
        )
        print(f"[MuseTalk] {len(whisper_chunks)} audio chunks for {len(frames)} video frames")

        # ── 5. UNet inference ──────────────────────────────────────────────
        print("[MuseTalk] Running UNet inference...")
        timesteps = torch.tensor([0], device=device)
        result_frames = []
        total_batches = math.ceil(len(whisper_chunks) / batch_size)

        with torch.no_grad():
            for w_batch, l_batch in tqdm(
                _datagen(whisper_chunks, vae_latents, batch_size),
                total=total_batches,
            ):
                audio_feat = _pe(w_batch.to(device=device, dtype=dtype))
                l_batch = l_batch.to(device=device, dtype=dtype)
                pred = _unet.model(l_batch, timesteps, encoder_hidden_states=audio_feat).sample
                recon = _vae.decode_latents(pred)
                result_frames.extend(recon)

        # ── 6. Paste generated faces back onto original frames ─────────────
        print("[MuseTalk] Compositing frames...")
        coord_cycle = coord_list + coord_list[::-1]
        frame_cycle = frames + frames[::-1]

        from musetalk.utils.blending import get_image as musetalk_blend

        for i, res in enumerate(tqdm(result_frames)):
            bbox = coord_cycle[i % len(coord_cycle)]
            ori = copy.deepcopy(frame_cycle[i % len(frame_cycle)])

            if bbox == _COORD_PLACEHOLDER:
                cv2.imwrite(os.path.join(result_dir, f"{i:08d}.png"), ori)
                continue

            x1, y1, x2, y2 = (int(v) for v in bbox)
            y2 = min(y2 + extra_margin, ori.shape[0])

            try:
                res_resized = cv2.resize(
                    res.astype(np.uint8), (x2 - x1, y2 - y1)
                )
                combined = musetalk_blend(
                    ori, res_resized, [x1, y1, x2, y2],
                    mode="jaw", fp=_fp,
                )
            except Exception:
                combined = ori

            cv2.imwrite(os.path.join(result_dir, f"{i:08d}.png"), combined)

        # ── 7. Assemble video + audio ──────────────────────────────────────
        print("[MuseTalk] Assembling final video...")
        tmp_vid = os.path.join(tmpdir, "temp_silent.mp4")
        os.system(
            f"ffmpeg -y -v warning -r {fps:.2f} -f image2 "
            f"-i {result_dir}/%08d.png "
            f"-vcodec libx264 -vf format=yuv420p -crf 18 {tmp_vid}"
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        os.system(
            f"ffmpeg -y -v warning "
            f"-i {audio_path} -i {tmp_vid} "
            f"-c:v copy -c:a aac -shortest {output_path}"
        )

    if not Path(output_path).exists():
        raise RuntimeError(f"MuseTalk output not created: {output_path}")

    print(f"[MuseTalk] Done → {output_path}")
    return output_path
