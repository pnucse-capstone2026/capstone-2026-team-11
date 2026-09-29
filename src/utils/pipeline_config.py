import json
import os
from pathlib import Path


# ============================================================
# Project Root
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CURRENT_CONFIG_PATH = (
    ROOT_DIR
    / "configs"
    / "current_pitch.json"
)


# ============================================================
# Load
# ============================================================

def load_current_pitch():

    if not CURRENT_CONFIG_PATH.exists():

        raise FileNotFoundError(
            "현재 Pitch 설정 파일이 없습니다.\n"
            f"{CURRENT_CONFIG_PATH}\n"
            "run_pipeline.py를 통해 실행했는지 "
            "확인하세요."
        )

    with open(
        CURRENT_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    return config


# ============================================================
# Basic Metadata
# ============================================================

def get_pitch_id():

    return str(
        load_current_pitch()[
            "pitch_id"
        ]
    )


def get_pitcher_name():

    return str(
        load_current_pitch()[
            "pitcher_name"
        ]
    )


def get_throwing_side():

    value = str(
        load_current_pitch()[
            "throwing_side"
        ]
    )

    value = (
        value
        .strip()
        .lower()
    )

    if value not in (
        "left",
        "right"
    ):

        raise ValueError(
            f"잘못된 throwing_side: {value}"
        )

    return value


def get_arm_slot():

    return str(
        load_current_pitch()
        .get(
            "arm_slot",
            "unknown"
        )
    )


# ============================================================
# Video
# ============================================================

def get_video_path():

    config = (
        load_current_pitch()
    )

    video_path = (
        ROOT_DIR
        / "data"
        / "raw_videos"
        / config[
            "video_file"
        ]
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"영상 파일이 없습니다: "
            f"{video_path}"
        )

    return str(
        video_path
    )


# ============================================================
# Manual Override
# ============================================================

def _optional_int(
    value
):

    if value is None:
        return None

    if isinstance(
        value,
        str
    ):

        value = (
            value.strip()
        )

        if value == "":
            return None

    try:

        return int(
            float(value)
        )

    except (
        ValueError,
        TypeError
    ):

        return None


def get_manual_knee_lift():

    config = (
        load_current_pitch()
    )

    return _optional_int(
        config.get(
            "manual_knee_lift_frame"
        )
    )


def get_manual_fc():

    config = (
        load_current_pitch()
    )

    return _optional_int(
        config.get(
            "manual_fc_frame"
        )
    )


def get_manual_release():

    config = (
        load_current_pitch()
    )

    return _optional_int(
        config.get(
            "manual_release_frame"
        )
    )
