"""Prepare comparable release-arm poses and lead-knee lift measurements.

Input is the existing per-pitch motion-normalized CSV plus its feature CSV.
No plots are produced here. Coordinates remain *projected 2D* measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import pandas as pd


@dataclass(frozen=True)
class PitchInput:
    pitch_id: str
    motion_csv: Path
    features_csv: Path
    throwing_side: str | None = None  # "L" / "R", if absent from feature CSV


def _number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _value(row: pd.Series, *names: str) -> object | None:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return row[name]
    return None


def _side(value: object) -> str:
    side = str(value or "").strip().lower()
    if side in {"l", "left", "좌", "좌투"}:
        return "L"
    if side in {"r", "right", "우", "우투"}:
        return "R"
    raise ValueError("throwing_side가 필요합니다 (L/R 또는 left/right).")


def _coordinate(row: pd.Series, joint: str) -> tuple[float, float]:
    """Read Sports2D left_shoulder_x_clean or LShoulder_x style columns."""
    canonical = re.sub(r"[^a-z0-9]", "", joint.lower())
    expanded = ("left" if canonical.startswith("l") else "right") + canonical[1:]
    columns = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in row.index}
    for base in (canonical, expanded):
        invalid = columns.get(base + "invalid")
        if invalid is not None and str(row[invalid]).strip().lower() in {"true", "1"}:
            raise ValueError(f"{joint} 관절이 이 프레임에서 invalid로 표시되었습니다.")
    for suffix in ("clean", ""):
        for base in (expanded, canonical):
            keys = [columns.get(base + axis + suffix) for axis in "xy"]
            if all(key is not None for key in keys):
                xy = tuple(_number(row[key]) for key in keys)
                if all(value is not None for value in xy):
                    return xy  # type: ignore[return-value]
    else:
        raise ValueError(f"{joint}의 x/y 좌표 열이 없습니다. CSV 열: {list(row.index)}")


def _event_row(motion: pd.DataFrame, frame: int, event: str) -> pd.Series:
    frame_column = next((c for c in ("frame", "frame_idx", "frame_index", "Frame#") if c in motion), None)
    if frame_column is None:
        raise ValueError("정규화 CSV에 원본 frame 열이 필요합니다.")
    candidates = motion.loc[pd.to_numeric(motion[frame_column], errors="coerce") == frame]
    if candidates.empty:
        raise ValueError(f"{event} frame={frame}이 정규화 CSV에 없습니다. 원본 프레임 번호를 확인하세요.")
    return candidates.iloc[0]


def _length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _body_scale(row: pd.Series) -> float:
    # Hip center to shoulder center is insensitive to image resolution.
    shoulders = [_coordinate(row, f"{s}Shoulder") for s in "LR"]
    hips = [_coordinate(row, f"{s}Hip") for s in "LR"]
    shoulder_mid = tuple(sum(p[i] for p in shoulders) / 2 for i in (0, 1))
    hip_mid = tuple(sum(p[i] for p in hips) / 2 for i in (0, 1))
    scale = _length(shoulder_mid, hip_mid)
    if scale <= 1e-9:
        raise ValueError("몸통 길이가 0이라 좌표를 정규화할 수 없습니다.")
    return scale


def prepare_pitch(pitch: PitchInput) -> dict:
    """Return one pitch's arm overlay coordinates and lead-knee lift metric.

    Arm points are shoulder-origin, torso-length-normalized, and mirrored so
    left/right throwing arms point to a common horizontal side. The vertical
    axis points upward (image CSV y normally points downward).
    """
    motion = pd.read_csv(pitch.motion_csv)
    features = pd.read_csv(pitch.features_csv)
    if motion.empty or features.empty:
        raise ValueError(f"{pitch.pitch_id}: CSV가 비어 있습니다.")
    info = features.iloc[0]
    side = _side(pitch.throwing_side or _value(info, "throwing_side", "pitcher_hand"))
    release = _number(_value(info, "release_candidate_frame", "release_frame", "pose_release_candidate_frame"))
    knee_frame = _number(_value(info, "knee_lift_frame", "maximum_knee_lift_frame", "lead_leg_lift_frame"))
    if release is None or knee_frame is None:
        raise ValueError(f"{pitch.pitch_id}: feature CSV에 release 및 knee_lift frame이 필요합니다.")
    release_row = _event_row(motion, round(release), "Release")
    knee_row = _event_row(motion, round(knee_frame), "Knee Lift")
    scale = _body_scale(release_row)
    shoulder = _coordinate(release_row, side + "Shoulder")
    elbow = _coordinate(release_row, side + "Elbow")
    wrist = _coordinate(release_row, side + "Wrist")
    # Side labels are anatomical only when the pose CSV's L/R mapping is correct.
    horizontal_sign = -1 if side == "L" else 1
    def align(point: tuple[float, float]) -> tuple[float, float]:
        return (horizontal_sign * (point[0] - shoulder[0]) / scale,
                (shoulder[1] - point[1]) / scale)
    arm = {"shoulder": align(shoulder), "elbow": align(elbow), "wrist": align(wrist)}

    # Use the stride leg (opposite throwing arm). Upward displacement from
    # stride hip is a projection, and may be negative when knee stays below hip.
    stride = "R" if side == "L" else "L"
    hip = _coordinate(knee_row, stride + "Hip")
    knee = _coordinate(knee_row, stride + "Knee")
    knee_scale = _body_scale(knee_row)
    lift_height = (hip[1] - knee[1]) / knee_scale
    lift_angle = _number(_value(info, "lead_leg_lift_angle_2d_deg", "lead_leg_lift_angle_deg"))
    elbow_valid = _value(release_row, "throwing_elbow_angle_valid", "elbow_angle_conf_valid")
    return {
        "pitch_id": pitch.pitch_id,
        "throwing_side": side,
        "release_frame": round(release),
        "knee_lift_frame": round(knee_frame),
        "release_arm": arm,
        "lead_knee_height_torso": lift_height,
        "lead_leg_lift_angle_2d_deg": lift_angle,
        "release_elbow_angle_valid": None if elbow_valid is None else str(elbow_valid).lower() in {"true", "1"},
    }


def compare_pitches(pitches: list[PitchInput]) -> dict:
    """Build data for an arm overlay and a ranked lead-knee comparison."""
    if len(pitches) < 2:
        raise ValueError("비교하려면 투구가 최소 2개 필요합니다.")
    ids = [p.pitch_id for p in pitches]
    if len(set(ids)) != len(ids):
        raise ValueError("pitch_id는 중복될 수 없습니다.")
    prepared = [prepare_pitch(p) for p in pitches]
    return {
        "pitches": prepared,
        "knee_height_ranking": sorted(
            ({"pitch_id": p["pitch_id"], "height_torso": p["lead_knee_height_torso"]}
             for p in prepared), key=lambda p: p["height_torso"], reverse=True),
        "notes": [
            "투구 팔은 어깨 원점과 몸통 길이로 정렬한 2D 투영 좌표입니다.",
            "무릎 높이는 Knee Lift 프레임에서 앞다리 무릎이 같은 쪽 엉덩이보다 높은 정도를 몸통 길이로 나눈 값입니다.",
            "서로 다른 촬영 각도, 카메라 회전 및 좌우 관절 오인식은 보정하지 않습니다.",
        ],
    }
