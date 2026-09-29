#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canonical Feature Extraction

Inputs
------
1) Canonical motion-normalized CSV
2) pipeline_outputs/<pitch_id>/analysis_events.json

Output
------
One-row feature CSV for pitcher comparison / web app use.

This version does NOT depend on legacy columns such as:
- final_*_v10
- pitch_event_v10
- release_candidate_simple
- release_*_simple
- normalization_* legacy metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Basic helpers
# ============================================================

def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"analysis_events.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(row, column):
    if row is None or column not in row.index:
        return np.nan

    value = row[column]

    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def get_frame_row(df: pd.DataFrame, frame):
    if frame is None:
        return None

    rows = df.loc[
        pd.to_numeric(
            df["frame"],
            errors="coerce",
        ) == int(frame)
    ]

    if len(rows) == 0:
        return None

    return rows.iloc[0]


def find_nearest_valid_release_value(
    df: pd.DataFrame,
    target_frame,
    value_column,
    valid_column=None,
    max_radius=3,
):
    """
    Release frame value fallback.

    Priority:
    exact Release -> +/-1F -> +/-2F -> +/-3F

    At the same distance, the earlier frame is preferred.
    Returns:
        value, frame_used, source
    """

    if (
        target_frame is None
        or value_column is None
        or value_column not in df.columns
    ):
        return np.nan, np.nan, "unavailable"

    frame_values = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    def read_candidate(frame_number):
        rows = df.loc[
            frame_values == int(frame_number)
        ]

        if len(rows) == 0:
            return None

        row = rows.iloc[0]

        value = safe_float(
            row,
            value_column,
        )

        if not np.isfinite(value):
            return None

        if (
            valid_column is not None
            and valid_column in df.columns
        ):
            raw_valid = row[valid_column]

            if pd.isna(raw_valid):
                return None

            if isinstance(raw_valid, (bool, np.bool_)):
                is_valid = bool(raw_valid)
            else:
                is_valid = (
                    str(raw_valid)
                    .strip()
                    .lower()
                    in ["true", "1", "yes"]
                )

            if not is_valid:
                return None

        return float(value)

    exact = read_candidate(
        int(target_frame)
    )

    if exact is not None:
        return (
            exact,
            int(target_frame),
            "exact_release",
        )

    for distance in range(
        1,
        int(max_radius) + 1,
    ):
        for frame_number in (
            int(target_frame) - distance,
            int(target_frame) + distance,
        ):
            value = read_candidate(
                frame_number
            )

            if value is not None:
                return (
                    value,
                    int(frame_number),
                    "nearest_valid_release_frame",
                )

    return np.nan, np.nan, "unavailable"


def find_column(df: pd.DataFrame, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def calculate_valid_ratio(
    df: pd.DataFrame,
    column,
    start_frame=None,
    end_frame=None,
):
    if column not in df.columns:
        return np.nan

    target = df

    if start_frame is not None:
        target = target.loc[
            target["frame"] >= start_frame
        ]

    if end_frame is not None:
        target = target.loc[
            target["frame"] <= end_frame
        ]

    if len(target) == 0:
        return np.nan

    raw = target[column].fillna(False)

    if pd.api.types.is_bool_dtype(raw):
        boolean = raw.astype(bool)
    else:
        boolean = (
            raw.astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    return float(boolean.mean())


def get_peak_row(
    df: pd.DataFrame,
    value_column,
    start_frame=None,
    end_frame=None,
):
    if value_column is None or value_column not in df.columns:
        return None

    target = df.copy()

    if start_frame is not None:
        target = target.loc[
            target["frame"] >= start_frame
        ]

    if end_frame is not None:
        target = target.loc[
            target["frame"] <= end_frame
        ]

    if len(target) == 0:
        return None

    values = pd.to_numeric(
        target[value_column],
        errors="coerce",
    )

    valid = values.notna()

    if not valid.any():
        return None

    idx = values.loc[valid].idxmax()

    return df.loc[idx]


def get_angle_extrema(
    df: pd.DataFrame,
    angle_column,
    start_frame,
    end_frame,
):
    if angle_column is None or angle_column not in df.columns:
        return np.nan, np.nan

    target = df.loc[
        (df["frame"] >= start_frame)
        & (df["frame"] <= end_frame)
    ]

    values = pd.to_numeric(
        target[angle_column],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan, np.nan

    return float(values.min()), float(values.max())


# ============================================================
# 2D geometry helpers
# ============================================================

def _find_joint_coord_column(row, side, joint, axis):
    if row is None:
        return None

    side = str(side).lower().strip()
    joint = str(joint).lower().strip()
    axis = str(axis).lower().strip()

    candidates = [
        f"{side}_{joint}_{axis}",
        f"{side}_{joint}_{axis}_clean",
        f"{side}_{joint}_{axis}_filtered",
        f"{side}_{joint}_{axis}_smooth",
        f"{side}_{joint}_{axis}_smoothed",
        f"{side}_{joint}_{axis}_interp",
        f"{side}_{joint}_{axis}_interpolated",
    ]

    for column in candidates:
        if column in row.index:
            return column

    prefix = f"{side}_{joint}_{axis}"

    blocked = (
        "conf",
        "score",
        "vel",
        "speed",
        "acc",
        "valid",
        "angle",
    )

    for column in row.index:
        name = str(column).lower().strip()

        if not name.startswith(prefix):
            continue

        if any(token in name for token in blocked):
            continue

        return column

    return None


def get_joint_xy(row, side, joint):
    x_col = _find_joint_coord_column(
        row,
        side,
        joint,
        "x",
    )

    y_col = _find_joint_coord_column(
        row,
        side,
        joint,
        "y",
    )

    if x_col is None or y_col is None:
        return np.nan, np.nan

    return (
        safe_float(row, x_col),
        safe_float(row, y_col),
    )


def calculate_vector_angle_deg(vector_a, vector_b):
    ax, ay = vector_a
    bx, by = vector_b

    values = [ax, ay, bx, by]

    if not all(np.isfinite(v) for v in values):
        return np.nan

    norm_a = float(np.hypot(ax, ay))
    norm_b = float(np.hypot(bx, by))

    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return np.nan

    cosine = (
        ax * bx + ay * by
    ) / (
        norm_a * norm_b
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def calculate_lead_leg_lift_angle_2d(
    knee_lift_row,
    stride_side,
    pivot_side,
):
    """
    2D projected lead-leg lift angle.

    Vertex:
        pivot hip

    Vectors:
        pivot hip -> stride knee
        pivot hip -> pivot ankle
    """
    pivot_hip_x, pivot_hip_y = get_joint_xy(
        knee_lift_row,
        pivot_side,
        "hip",
    )

    stride_knee_x, stride_knee_y = get_joint_xy(
        knee_lift_row,
        stride_side,
        "knee",
    )

    pivot_ankle_x, pivot_ankle_y = get_joint_xy(
        knee_lift_row,
        pivot_side,
        "ankle",
    )

    required = [
        pivot_hip_x,
        pivot_hip_y,
        stride_knee_x,
        stride_knee_y,
    ]

    if not all(np.isfinite(v) for v in required):
        return np.nan, "unavailable"

    lead_vector = (
        stride_knee_x - pivot_hip_x,
        stride_knee_y - pivot_hip_y,
    )

    if (
        np.isfinite(pivot_ankle_x)
        and np.isfinite(pivot_ankle_y)
    ):
        reference_vector = (
            pivot_ankle_x - pivot_hip_x,
            pivot_ankle_y - pivot_hip_y,
        )
        method = "pivot_hip_to_stride_knee_vs_pivot_ankle"
    else:
        reference_vector = (0.0, 1.0)
        method = "pivot_hip_to_stride_knee_vs_vertical_down"

    angle = calculate_vector_angle_deg(
        lead_vector,
        reference_vector,
    )

    return angle, method


def calculate_arm_angle_2d(
    release_row,
    throwing_side,
):
    """
    Shoulder-to-wrist elevation relative to image horizontal.

    This is a 2D projected metric and is NOT Statcast Arm Angle.
    """
    shoulder_x, shoulder_y = get_joint_xy(
        release_row,
        throwing_side,
        "shoulder",
    )

    wrist_x, wrist_y = get_joint_xy(
        release_row,
        throwing_side,
        "wrist",
    )

    values = [
        shoulder_x,
        shoulder_y,
        wrist_x,
        wrist_y,
    ]

    if not all(np.isfinite(v) for v in values):
        return np.nan

    dx = wrist_x - shoulder_x
    dy_up = shoulder_y - wrist_y

    if abs(dx) <= 1e-9 and abs(dy_up) <= 1e-9:
        return np.nan

    return float(
        np.degrees(
            np.arctan2(
                dy_up,
                abs(dx),
            )
        )
    )


# ============================================================
# Canonical event helpers
# ============================================================

def get_event(events_json: dict, key: str):
    block = (
        events_json
        .get("events", {})
        .get(key, {})
    )

    frame = block.get("frame")

    if frame is not None:
        frame = int(round(float(frame)))

    return {
        "frame": frame,
        "source": str(
            block.get(
                "source",
                "unknown",
            )
        ),
        "confidence": str(
            block.get(
                "confidence",
                "unknown",
            )
        ),
    }


# ============================================================
# Main feature extraction
# ============================================================

def extract_pitch_features(
    csv_path,
    events_json_path,
    output_csv_path,
):
    csv_path = Path(csv_path)
    events_json_path = Path(events_json_path)
    output_csv_path = Path(output_csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    events_json = read_json(events_json_path)

    required = [
        "frame",
        "timestamp",
        "pitch_progress_time_percent",
        "fc_release_progress_time_percent",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing)
        )

    df["frame"] = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    if df["frame"].isna().any():
        raise ValueError(
            "frame column contains NaN."
        )

    df["frame"] = df["frame"].astype(int)

    pitch_id = str(
        events_json.get(
            "pitch_id",
            "",
        )
    )

    pitcher_name = str(
        events_json.get(
            "pitcher_name",
            "",
        )
    )

    throwing_side = str(
        events_json.get(
            "throwing_side",
            "",
        )
    ).lower().strip()

    if throwing_side not in {
        "left",
        "right",
    }:
        raise ValueError(
            "throwing_side must be 'left' or 'right'."
        )

    knee = get_event(
        events_json,
        "knee_lift",
    )

    fc = get_event(
        events_json,
        "front_foot_contact",
    )

    release = get_event(
        events_json,
        "release",
    )

    knee_lift_frame = knee["frame"]
    foot_contact_frame = fc["frame"]
    release_frame = release["frame"]

    if (
        knee_lift_frame is None
        or foot_contact_frame is None
        or release_frame is None
    ):
        raise ValueError(
            "analysis_events.json contains a missing event frame."
        )

    if not (
        knee_lift_frame
        < foot_contact_frame
        < release_frame
    ):
        raise ValueError(
            "Invalid event order: "
            f"Knee={knee_lift_frame}, "
            f"FC={foot_contact_frame}, "
            f"Release={release_frame}"
        )

    knee_lift_row = get_frame_row(
        df,
        knee_lift_frame,
    )

    foot_contact_row = get_frame_row(
        df,
        foot_contact_frame,
    )

    release_row = get_frame_row(
        df,
        release_frame,
    )

    if (
        knee_lift_row is None
        or foot_contact_row is None
        or release_row is None
    ):
        raise ValueError(
            "One or more event frames do not exist in the normalized CSV."
        )

    knee_time = safe_float(
        knee_lift_row,
        "timestamp",
    )

    fc_time = safe_float(
        foot_contact_row,
        "timestamp",
    )

    release_time = safe_float(
        release_row,
        "timestamp",
    )

    knee_to_fc_time = (
        fc_time - knee_time
    )

    fc_to_release_time = (
        release_time - fc_time
    )

    knee_to_release_time = (
        release_time - knee_time
    )

    foot_contact_percent = safe_float(
        foot_contact_row,
        "pitch_progress_time_percent",
    )

    release_percent = safe_float(
        release_row,
        "pitch_progress_time_percent",
    )

    # --------------------------------------------------------
    # Joint-angle columns
    # --------------------------------------------------------

    elbow_angle_col = find_column(
        df,
        [
            "throwing_elbow_angle_filtered",
            "throwing_elbow_angle",
            "throwing_elbow_angle_deg",
        ],
    )

    left_knee_col = find_column(
        df,
        [
            "left_knee_angle",
            "left_knee_angle_filtered",
        ],
    )

    right_knee_col = find_column(
        df,
        [
            "right_knee_angle",
            "right_knee_angle_filtered",
        ],
    )

    elbow_angle_fc = (
        safe_float(
            foot_contact_row,
            elbow_angle_col,
        )
        if elbow_angle_col
        else np.nan
    )

    if elbow_angle_col:
        (
            elbow_angle_release,
            elbow_angle_release_frame_used,
            elbow_angle_release_source,
        ) = find_nearest_valid_release_value(
            df=df,
            target_frame=release_frame,
            value_column=elbow_angle_col,
            valid_column=(
                "throwing_elbow_angle_valid"
                if "throwing_elbow_angle_valid" in df.columns
                else None
            ),
            max_radius=3,
        )
    else:
        elbow_angle_release = np.nan
        elbow_angle_release_frame_used = np.nan
        elbow_angle_release_source = "unavailable"

    left_knee_fc = (
        safe_float(
            foot_contact_row,
            left_knee_col,
        )
        if left_knee_col
        else np.nan
    )

    right_knee_fc = (
        safe_float(
            foot_contact_row,
            right_knee_col,
        )
        if right_knee_col
        else np.nan
    )

    left_knee_release = (
        safe_float(
            release_row,
            left_knee_col,
        )
        if left_knee_col
        else np.nan
    )

    right_knee_release = (
        safe_float(
            release_row,
            right_knee_col,
        )
        if right_knee_col
        else np.nan
    )

    if throwing_side == "left":
        stride_side = "right"
        pivot_side = "left"

        stride_knee_fc = right_knee_fc
        pivot_knee_fc = left_knee_fc

        stride_knee_release = right_knee_release
        pivot_knee_release = left_knee_release
    else:
        stride_side = "left"
        pivot_side = "right"

        stride_knee_fc = left_knee_fc
        pivot_knee_fc = right_knee_fc

        stride_knee_release = left_knee_release
        pivot_knee_release = right_knee_release

    # --------------------------------------------------------
    # 2D geometry
    # --------------------------------------------------------

    (
        lead_leg_lift_angle_deg,
        lead_leg_lift_angle_method,
    ) = calculate_lead_leg_lift_angle_2d(
        knee_lift_row=knee_lift_row,
        stride_side=stride_side,
        pivot_side=pivot_side,
    )

    arm_angle_2d_estimated_deg = (
        calculate_arm_angle_2d(
            release_row=release_row,
            throwing_side=throwing_side,
        )
    )

    arm_angle_2d_method = (
        "throwing_shoulder_to_wrist_vs_horizontal"
        if np.isfinite(
            arm_angle_2d_estimated_deg
        )
        else "unavailable"
    )

    # --------------------------------------------------------
    # Wrist motion
    # --------------------------------------------------------

    wrist_speed_col = find_column(
        df,
        [
            "wrist_relative_speed_body_s_validated",
            "wrist_relative_speed_body_s_smooth",
            "wrist_speed_body_s_validated",
            "wrist_speed_body_s",
            "wrist_relative_speed_body_s",
        ],
    )

    # Release wrist speed:
    # 1) exact/nearby validated value within ±3F
    # 2) if unavailable, use the raw/smoothed value at the exact Release frame
    #    (important for fallback-style release detections such as Rogers)
    raw_wrist_speed_col = find_column(
        df,
        [
            "wrist_relative_speed_body_s_smooth",
            "wrist_speed_body_s_smooth",
            "wrist_relative_speed_body_s",
            "wrist_speed_body_s",
        ],
    )

    if wrist_speed_col:
        (
            release_wrist_speed,
            release_wrist_speed_frame_used,
            release_wrist_speed_source,
        ) = find_nearest_valid_release_value(
            df=df,
            target_frame=release_frame,
            value_column=wrist_speed_col,
            valid_column=(
                "wrist_speed_valid"
                if "wrist_speed_valid" in df.columns
                else None
            ),
            max_radius=3,
        )
    else:
        release_wrist_speed = np.nan
        release_wrist_speed_frame_used = np.nan
        release_wrist_speed_source = "unavailable"

    if (
        not np.isfinite(release_wrist_speed)
        and raw_wrist_speed_col is not None
    ):
        raw_release_speed = safe_float(
            release_row,
            raw_wrist_speed_col,
        )

        if np.isfinite(raw_release_speed):
            release_wrist_speed = raw_release_speed
            release_wrist_speed_frame_used = int(release_frame)
            release_wrist_speed_source = "exact_release_raw_fallback"

    peak_knee_release_row = get_peak_row(
        df=df,
        value_column=wrist_speed_col,
        start_frame=knee_lift_frame,
        end_frame=release_frame,
    )

    if peak_knee_release_row is not None:
        peak_knee_release_speed = safe_float(
            peak_knee_release_row,
            wrist_speed_col,
        )

        peak_knee_release_frame = int(
            peak_knee_release_row["frame"]
        )

        peak_knee_release_percent = safe_float(
            peak_knee_release_row,
            "pitch_progress_time_percent",
        )
    else:
        peak_knee_release_speed = np.nan
        peak_knee_release_frame = np.nan
        peak_knee_release_percent = np.nan

    peak_fc_release_row = get_peak_row(
        df=df,
        value_column=wrist_speed_col,
        start_frame=foot_contact_frame,
        end_frame=release_frame,
    )

    if peak_fc_release_row is not None:
        peak_fc_release_speed = safe_float(
            peak_fc_release_row,
            wrist_speed_col,
        )

        peak_fc_release_frame = int(
            peak_fc_release_row["frame"]
        )

        peak_fc_release_percent = safe_float(
            peak_fc_release_row,
            "fc_release_progress_time_percent",
        )
    else:
        peak_fc_release_speed = np.nan
        peak_fc_release_frame = np.nan
        peak_fc_release_percent = np.nan

    (
        min_elbow_angle,
        max_elbow_angle,
    ) = get_angle_extrema(
        df=df,
        angle_column=elbow_angle_col,
        start_frame=knee_lift_frame,
        end_frame=release_frame,
    )

    # --------------------------------------------------------
    # Optional lower-body distance
    # --------------------------------------------------------

    ankle_separation_col = find_column(
        df,
        [
            "ankle_separation_body",
            "ankle_distance_body",
        ],
    )

    ankle_separation_fc = (
        safe_float(
            foot_contact_row,
            ankle_separation_col,
        )
        if ankle_separation_col
        else np.nan
    )

    ankle_separation_release = (
        safe_float(
            release_row,
            ankle_separation_col,
        )
        if ankle_separation_col
        else np.nan
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    elbow_valid_ratio_total = (
        calculate_valid_ratio(
            df,
            "throwing_elbow_angle_valid",
        )
    )

    wrist_valid_ratio_total = (
        calculate_valid_ratio(
            df,
            "wrist_speed_valid",
        )
    )

    elbow_valid_ratio_motion = (
        calculate_valid_ratio(
            df,
            "throwing_elbow_angle_valid",
            start_frame=knee_lift_frame,
            end_frame=release_frame,
        )
    )

    wrist_valid_ratio_motion = (
        calculate_valid_ratio(
            df,
            "wrist_speed_valid",
            start_frame=knee_lift_frame,
            end_frame=release_frame,
        )
    )

    wrist_valid_ratio_fc_release = (
        calculate_valid_ratio(
            df,
            "wrist_speed_valid",
            start_frame=foot_contact_frame,
            end_frame=release_frame,
        )
    )

    # --------------------------------------------------------
    # Sports2D arm-slot result from canonical events JSON
    # --------------------------------------------------------

    arm_slot = events_json.get(
        "arm_slot",
        {},
    )

    arm_slot_category = arm_slot.get(
        "category"
    )

    arm_slot_angle_deg = arm_slot.get(
        "angle_deg"
    )

    arm_slot_confidence = arm_slot.get(
        "confidence",
        "unknown",
    )

    arm_slot_source = arm_slot.get(
        "source",
        "unknown",
    )

    # --------------------------------------------------------
    # Final feature row
    # --------------------------------------------------------

    feature_row = {
        # Identification
        "pitch_id":
            pitch_id,

        "pitcher_name":
            pitcher_name,

        "throwing_side":
            throwing_side,

        "stride_side":
            stride_side,

        "pivot_side":
            pivot_side,

        # Events
        "knee_lift_frame":
            knee_lift_frame,

        "front_foot_contact_frame":
            foot_contact_frame,

        "release_frame":
            release_frame,

        # Event source / confidence
        "knee_lift_source":
            knee["source"],

        "knee_lift_confidence":
            knee["confidence"],

        "front_foot_contact_source":
            fc["source"],

        "front_foot_contact_confidence":
            fc["confidence"],

        "release_source":
            release["source"],

        "release_confidence":
            release["confidence"],

        # Timing
        "knee_lift_to_fc_time_s":
            knee_to_fc_time,

        "fc_to_release_time_s":
            fc_to_release_time,

        "knee_lift_to_release_time_s":
            knee_to_release_time,

        # Normalized timing
        "front_foot_contact_percent":
            foot_contact_percent,

        "release_percent":
            release_percent,

        # Elbow
        "elbow_angle_front_foot_contact_deg":
            elbow_angle_fc,

        "elbow_angle_release_deg":
            elbow_angle_release,

        "elbow_angle_release_frame_used":
            elbow_angle_release_frame_used,

        "elbow_angle_release_source":
            elbow_angle_release_source,

        "min_elbow_angle_knee_to_release_deg":
            min_elbow_angle,

        "max_elbow_angle_knee_to_release_deg":
            max_elbow_angle,

        # Knee @ FC
        "left_knee_angle_front_foot_contact_deg":
            left_knee_fc,

        "right_knee_angle_front_foot_contact_deg":
            right_knee_fc,

        "stride_knee_angle_front_foot_contact_deg":
            stride_knee_fc,

        "pivot_knee_angle_front_foot_contact_deg":
            pivot_knee_fc,

        # Knee @ Release
        "left_knee_angle_release_deg":
            left_knee_release,

        "right_knee_angle_release_deg":
            right_knee_release,

        "stride_knee_angle_release_deg":
            stride_knee_release,

        "pivot_knee_angle_release_deg":
            pivot_knee_release,

        # 2D geometry
        "lead_leg_lift_angle_2d_deg":
            lead_leg_lift_angle_deg,

        "lead_leg_lift_angle_method":
            lead_leg_lift_angle_method,

        "arm_angle_2d_estimated_deg":
            arm_angle_2d_estimated_deg,

        "arm_angle_2d_method":
            arm_angle_2d_method,

        # Sports2D arm slot classification
        "arm_slot":
            arm_slot_category,

        "arm_slot_angle_deg":
            arm_slot_angle_deg,

        "arm_slot_confidence":
            arm_slot_confidence,

        "arm_slot_source":
            arm_slot_source,

        # Wrist motion
        "release_wrist_speed_body_s":
            release_wrist_speed,

        "release_wrist_speed_frame_used":
            release_wrist_speed_frame_used,

        "release_wrist_speed_source":
            release_wrist_speed_source,

        "peak_wrist_speed_knee_to_release_body_s":
            peak_knee_release_speed,

        "peak_wrist_speed_knee_to_release_frame":
            peak_knee_release_frame,

        "peak_wrist_speed_knee_to_release_percent":
            peak_knee_release_percent,

        "peak_wrist_speed_fc_to_release_body_s":
            peak_fc_release_speed,

        "peak_wrist_speed_fc_to_release_frame":
            peak_fc_release_frame,

        "peak_wrist_speed_fc_to_release_percent":
            peak_fc_release_percent,

        # Lower body
        "ankle_separation_front_foot_contact_body":
            ankle_separation_fc,

        "ankle_separation_release_body":
            ankle_separation_release,

        # Quality
        "valid_elbow_frame_ratio_total":
            elbow_valid_ratio_total,

        "valid_wrist_speed_frame_ratio_total":
            wrist_valid_ratio_total,

        "valid_elbow_frame_ratio_knee_to_release":
            elbow_valid_ratio_motion,

        "valid_wrist_speed_frame_ratio_knee_to_release":
            wrist_valid_ratio_motion,

        "valid_wrist_speed_frame_ratio_fc_to_release":
            wrist_valid_ratio_fc_release,

        "total_analysis_frames":
            len(df),
    }

    feature_df = pd.DataFrame(
        [feature_row]
    )

    output_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("===== Canonical Feature Extraction =====")
    print(f"Pitch ID       : {pitch_id}")
    print(f"Pitcher        : {pitcher_name}")
    print(f"Throwing Side  : {throwing_side}")
    print()
    print(
        f"Knee Lift      : {knee_lift_frame} "
        f"[{knee['source']}]"
    )
    print(
        f"FC             : {foot_contact_frame} "
        f"[{fc['source']}]"
    )
    print(
        f"Release        : {release_frame} "
        f"[{release['source']}, {release['confidence']}]"
    )
    print()
    print(
        f"Knee -> FC     : {knee_to_fc_time:.4f} sec"
    )
    print(
        f"FC -> Release  : {fc_to_release_time:.4f} sec"
    )
    print()
    print(
        f"Arm Slot       : {arm_slot_category} "
        f"[{arm_slot_confidence}]"
    )

    if arm_slot_angle_deg is not None:
        try:
            print(
                f"Arm Slot Angle : "
                f"{float(arm_slot_angle_deg):.2f} deg"
            )
        except (TypeError, ValueError):
            pass

    if np.isfinite(
        arm_angle_2d_estimated_deg
    ):
        print(
            f"Arm Angle 2D   : "
            f"{arm_angle_2d_estimated_deg:.2f} deg"
        )

    if np.isfinite(
        lead_leg_lift_angle_deg
    ):
        print(
            f"Lead-Leg Lift  : "
            f"{lead_leg_lift_angle_deg:.2f} deg"
        )

    if np.isfinite(
        elbow_angle_release
    ):
        print(
            f"Elbow @ Release: "
            f"{elbow_angle_release:.2f} deg "
            f"[frame {int(elbow_angle_release_frame_used)}, "
            f"{elbow_angle_release_source}]"
        )

    if np.isfinite(
        release_wrist_speed
    ):
        print(
            f"Wrist @ Release: "
            f"{release_wrist_speed:.3f} body/s "
            f"[frame {int(release_wrist_speed_frame_used)}, "
            f"{release_wrist_speed_source}]"
        )

    print()
    print(f"Output CSV     : {output_csv_path}")
    print("[DONE] Canonical Feature Extraction")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Canonical motion_normalized.csv",
    )

    parser.add_argument(
        "--events",
        required=True,
        help="analysis_events.json",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output feature CSV",
    )

    args = parser.parse_args()

    extract_pitch_features(
        csv_path=args.input,
        events_json_path=args.events,
        output_csv_path=args.output,
    )


if __name__ == "__main__":
    main()
