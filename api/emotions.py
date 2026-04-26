# coding: utf-8

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).parent.parent

AnimationRegion = Literal["exp", "pose", "lip", "eyes", "all"]
DrivingOption = Literal["expression-friendly", "pose-friendly"]


@dataclass
class EmotionPreset:
    name: str
    description: str
    driving_file: str          # relative to ROOT
    animation_region: AnimationRegion = "all"
    driving_option: DrivingOption = "expression-friendly"
    driving_multiplier: float = 1.0
    flag_stitching: bool = True  # False for large head movements


EMOTIONS: dict[str, EmotionPreset] = {
    "talking": EmotionPreset(
        name="talking",
        description="Talking or giving a speech with natural lip and expression movement",
        driving_file="assets/examples/driving/talking.pkl",
        animation_region="all",
        driving_option="expression-friendly",
        driving_multiplier=1.0,
    ),
    "speech": EmotionPreset(
        name="speech",
        description="Giving a formal speech with slightly amplified expression and head motion",
        driving_file="assets/examples/driving/talking.pkl",
        animation_region="all",
        driving_option="expression-friendly",
        driving_multiplier=1.2,
    ),
    "smiling": EmotionPreset(
        name="smiling",
        description="Gentle smile, suitable for an interview or friendly conversation",
        driving_file="assets/examples/driving/laugh.pkl",
        animation_region="exp",
        driving_option="expression-friendly",
        driving_multiplier=0.7,
    ),
    "interview_smile": EmotionPreset(
        name="interview_smile",
        description="Subtle, professional interview smile",
        driving_file="assets/examples/driving/laugh.pkl",
        animation_region="exp",
        driving_option="expression-friendly",
        driving_multiplier=0.5,
    ),
    "laughing": EmotionPreset(
        name="laughing",
        description="Full laugh with head and expression movement",
        driving_file="assets/examples/driving/laugh.pkl",
        animation_region="all",
        driving_option="expression-friendly",
        driving_multiplier=1.3,
        flag_stitching=False,
    ),
    "nodding": EmotionPreset(
        name="nodding",
        description="Nodding head in agreement",
        driving_file="assets/examples/driving/d0.mp4",
        animation_region="pose",
        driving_option="pose-friendly",
        driving_multiplier=1.0,
        flag_stitching=False,
    ),
    "disagreeing": EmotionPreset(
        name="disagreeing",
        description="Shaking head side-to-side in disagreement",
        driving_file="assets/examples/driving/shake_face.pkl",
        animation_region="pose",
        driving_option="pose-friendly",
        driving_multiplier=1.0,
        flag_stitching=False,
    ),
    "listening": EmotionPreset(
        name="listening",
        description="Subtle attentive head tilt and expression while listening",
        driving_file="assets/examples/driving/d3.mp4",
        animation_region="all",
        driving_option="expression-friendly",
        driving_multiplier=0.6,
    ),
    "winking": EmotionPreset(
        name="winking",
        description="Winking one eye",
        driving_file="assets/examples/driving/wink.pkl",
        animation_region="eyes",
        driving_option="expression-friendly",
        driving_multiplier=1.0,
    ),
    "shy": EmotionPreset(
        name="shy",
        description="Shy or bashful facial expression",
        driving_file="assets/examples/driving/shy.pkl",
        animation_region="exp",
        driving_option="expression-friendly",
        driving_multiplier=1.0,
    ),
    "aggrieved": EmotionPreset(
        name="aggrieved",
        description="Aggrieved or concerned expression",
        driving_file="assets/examples/driving/aggrieved.pkl",
        animation_region="exp",
        driving_option="expression-friendly",
        driving_multiplier=1.0,
    ),
    "open_mouth": EmotionPreset(
        name="open_mouth",
        description="Open mouth, surprised or starting to speak",
        driving_file="assets/examples/driving/open_lip.pkl",
        animation_region="lip",
        driving_option="expression-friendly",
        driving_multiplier=1.0,
    ),
}
