#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sports2D Arm Slot Classifier v8.5
---------------------------------

Experimental height-aware 2D arm-slot classifier.

Why v8.3
--------
v7.3 depended too heavily on the screen-horizontal shoulder->wrist angle.
v8/v8.1 tried shoulder-line / torso-axis relative angles, but pitchers with
large lateral trunk tilt could still be misclassified.

v8.3 does NOT compress the whole posture into one geometric angle.
Instead it combines:
  1) wrist height above throwing shoulder (normalized by shoulder width)
  2) elbow height above throwing shoulder (normalized by shoulder width)
  3) shoulder->wrist screen angle
  4) shoulder->elbow screen angle
  5) persistent wrist-below state for Underhand

The final category is therefore a posture-pattern classification.

Important
---------
- Single-camera 2D projected categorical estimate only.
- Not Statcast Arm Angle.
- Not true 3D biomechanical arm slot.
- Thresholds are experimental/sample-calibrated and should be validated
  across the full pitcher set before treating them as general thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_xy_cols(df: pd.DataFrame, joint: str):
    for sx, sy in [
        ("_x_clean", "_y_clean"),
        ("_x", "_y"),
    ]:
        xcol = joint + sx
        ycol = joint + sy
        if xcol in df.columns and ycol in df.columns:
            return xcol, ycol
    return None, None


def _numeric_series(df: pd.DataFrame, col: str):
    return pd.to_numeric(df[col], errors="coerce")


def _safe_point(x: pd.Series, y: pd.Series, idx):
    xv = x.loc[idx]
    yv = y.loc[idx]
    if not (np.isfinite(xv) and np.isfinite(yv)):
        return None
    return np.array([float(xv), float(yv)], dtype=float)


def _screen_angle_deg(vec: np.ndarray) -> float:
    """
    Absolute 2D angle against screen horizontal, constrained to 0..90 deg.
    """
    dx = abs(float(vec[0]))
    dy = abs(float(vec[1]))
    if dx < 1e-8 and dy < 1e-8:
        return float("nan")
    return float(np.degrees(np.arctan2(dy, dx)))


def frame_geometry(
    df: pd.DataFrame,
    target_frame: int,
    lsx: pd.Series,
    lsy: pd.Series,
    rsx: pd.Series,
    rsy: pd.Series,
    ex: pd.Series,
    ey: pd.Series,
    wx: pd.Series,
    wy: pd.Series,
    throwing_side: str,
):
    frame = pd.to_numeric(df["frame"], errors="coerce")
    matches = df.index[frame == target_frame].tolist()

    if not matches:
        return None

    for idx in matches:
        left_shoulder = _safe_point(lsx, lsy, idx)
        right_shoulder = _safe_point(rsx, rsy, idx)
        elbow = _safe_point(ex, ey, idx)
        wrist = _safe_point(wx, wy, idx)

        if any(
            p is None
            for p in [left_shoulder, right_shoulder, elbow, wrist]
        ):
            continue

        throwing_shoulder = (
            left_shoulder if throwing_side == "left" else right_shoulder
        )

        shoulder_width = float(
            np.linalg.norm(right_shoulder - left_shoulder)
        )
        if shoulder_width < 1e-8:
            continue

        upper_arm = elbow - throwing_shoulder
        shoulder_to_wrist = wrist - throwing_shoulder

        if (
            np.linalg.norm(upper_arm) < 1e-8
            or np.linalg.norm(shoulder_to_wrist) < 1e-8
        ):
            continue

        wrist_height = float(
            (throwing_shoulder[1] - wrist[1]) / shoulder_width
        )
        elbow_height = float(
            (throwing_shoulder[1] - elbow[1]) / shoulder_width
        )

        wrist_angle = _screen_angle_deg(shoulder_to_wrist)
        upper_arm_angle = _screen_angle_deg(upper_arm)

        wrist_below = bool(wrist[1] > throwing_shoulder[1])
        elbow_below = bool(elbow[1] > throwing_shoulder[1])

        return {
            "frame": int(target_frame),

            "throwing_shoulder_x": float(throwing_shoulder[0]),
            "throwing_shoulder_y": float(throwing_shoulder[1]),
            "opposite_shoulder_x": float(
                right_shoulder[0]
                if throwing_side == "left"
                else left_shoulder[0]
            ),
            "opposite_shoulder_y": float(
                right_shoulder[1]
                if throwing_side == "left"
                else left_shoulder[1]
            ),
            "elbow_x": float(elbow[0]),
            "elbow_y": float(elbow[1]),
            "wrist_x": float(wrist[0]),
            "wrist_y": float(wrist[1]),

            "shoulder_width_px": shoulder_width,
            "wrist_height_above_shoulder_body": wrist_height,
            "elbow_height_above_shoulder_body": elbow_height,
            "shoulder_wrist_screen_angle_deg": float(wrist_angle),
            "upper_arm_screen_angle_deg": float(upper_arm_angle),

            "wrist_below_shoulder": wrist_below,
            "elbow_below_shoulder": elbow_below,
        }

    return None


def posture_score(row: dict) -> tuple[int, dict]:
    """
    Height-aware categorical score.

    Score design:
      Wrist height:
        >= 0.75 shoulder-widths : +2
        >= 0.30                 : +1

      Elbow height:
        >= 0.25                 : +2
        >= 0.05                 : +1

      Screen angles:
        shoulder->wrist >= 50°  : +1
        shoulder->elbow >= 45°  : +1

    Max score = 6.

    The score is not a biomechanical angle. It is only a compact diagnostic
    for the posture-pattern classifier.
    """
    wrist_h = float(row["wrist_height_above_shoulder_body"])
    elbow_h = float(row["elbow_height_above_shoulder_body"])
    wrist_ang = float(row["shoulder_wrist_screen_angle_deg"])
    upper_ang = float(row["upper_arm_screen_angle_deg"])

    details = {
        "wrist_height_points": 0,
        "elbow_height_points": 0,
        "wrist_angle_points": 0,
        "upper_arm_angle_points": 0,
    }

    if wrist_h >= 0.75:
        details["wrist_height_points"] = 2
    elif wrist_h >= 0.30:
        details["wrist_height_points"] = 1

    if elbow_h >= 0.25:
        details["elbow_height_points"] = 2
    elif elbow_h >= 0.05:
        details["elbow_height_points"] = 1

    if wrist_ang >= 50.0:
        details["wrist_angle_points"] = 1

    if upper_ang >= 45.0:
        details["upper_arm_angle_points"] = 1

    score = int(sum(details.values()))
    return score, details


def classify_non_underhand(row: dict):
    """
    v8.5 posture-pattern rules.

    Overhand is intentionally stricter than v8.4.

    A) Strong overall high-slot posture:
       - score >= 5
       - wrist height >= 0.60 shoulder widths
       - elbow height >= 0.30 shoulder widths

    B) Very-high arm posture:
       - wrist height >= 0.80 shoulder widths
       - elbow height >= 0.30 shoulder widths
       - shoulder->elbow screen angle >= 35 deg

    Three-Quarter:
       - score >= 2
         OR wrist height >= 0.18
         OR (wrist height >= 0.10 and elbow height >= 0.05)

    Sidearm:
       - otherwise

    This preserves the v8.4 three-frame representative posture logic while
    reducing Overhand over-classification for high three-quarter pitchers.
    """
    score, details = posture_score(row)

    wrist_h = float(row["wrist_height_above_shoulder_body"])
    elbow_h = float(row["elbow_height_above_shoulder_body"])
    upper_ang = float(row["upper_arm_screen_angle_deg"])

    overhand_by_strong_posture = (
        score >= 5
        and wrist_h >= 0.60
        and elbow_h >= 0.30
    )

    overhand_by_very_high_arm = (
        wrist_h >= 0.80
        and elbow_h >= 0.30
        and upper_ang >= 35.0
    )

    if overhand_by_strong_posture or overhand_by_very_high_arm:
        return "Overhand", score, details

    three_quarter = (
        score >= 2
        or wrist_h >= 0.18
        or (wrist_h >= 0.10 and elbow_h >= 0.05)
    )

    if three_quarter:
        return "Three-Quarter", score, details

    return "Sidearm", score, details

def confidence_from_window(
    arm_slot: str,
    final_score: int,
    window_scores: list[int],
    valid_frames: int,
):
    """
    Internal consistency confidence, not biomechanical confidence.
    """
    if valid_frames < 2:
        return "low"

    if not window_scores:
        return "low"

    spread = max(window_scores) - min(window_scores)

    if arm_slot == "Underhand":
        return "medium"

    if spread >= 3:
        return "low"

    if arm_slot == "Overhand":
        if final_score >= 5 and spread <= 1:
            return "high"
        if final_score >= 3 and spread <= 2:
            return "medium"
        return "low"

    if arm_slot == "Three-Quarter":
        if 2 <= final_score <= 4 and spread <= 1:
            return "high"
        if spread <= 2:
            return "medium"
        return "low"

    # Sidearm
    if final_score <= 1 and spread <= 1:
        return "high"
    if spread <= 2:
        return "medium"
    return "low"


def run_arm_slot_v8_4(
    df: pd.DataFrame,
    throwing_side: str,
    release_frame: int,
    pre_frames: int = 2,
    overhand_threshold: float = 78.0,         # kept for CLI compatibility
    three_quarter_threshold: float = 60.0,    # kept for CLI compatibility
    underhand_angle_threshold: float = 45.0,
    underhand_below_ratio_threshold: float = 0.67,
):
    if "frame" not in df.columns:
        raise ValueError("입력 CSV에 frame 컬럼이 없습니다.")

    side = throwing_side.lower().strip()
    if side not in {"left", "right"}:
        raise ValueError("throwing_side는 left 또는 right여야 합니다.")

    required_joints = [
        "left_shoulder",
        "right_shoulder",
        f"{side}_elbow",
        f"{side}_wrist",
    ]

    cols = {}
    for joint in required_joints:
        xcol, ycol = find_xy_cols(df, joint)
        if xcol is None or ycol is None:
            raise RuntimeError(
                f"{joint} 좌표 컬럼을 찾지 못했습니다."
            )
        cols[joint] = (
            _numeric_series(df, xcol),
            _numeric_series(df, ycol),
        )

    lsx, lsy = cols["left_shoulder"]
    rsx, rsy = cols["right_shoulder"]
    ex, ey = cols[f"{side}_elbow"]
    wx, wy = cols[f"{side}_wrist"]

    def geom(f):
        return frame_geometry(
            df=df,
            target_frame=f,
            lsx=lsx,
            lsy=lsy,
            rsx=rsx,
            rsy=rsy,
            ex=ex,
            ey=ey,
            wx=wx,
            wy=wy,
            throwing_side=side,
        )

    window_rows = []
    for f in range(release_frame - pre_frames, release_frame + 1):
        g = geom(f)
        if g is not None:
            score, details = posture_score(g)
            g["posture_score"] = score
            for k, v in details.items():
                g[k] = v
            g["frame_offset_from_release"] = int(f - release_frame)
            window_rows.append(g)

    if not window_rows:
        raise RuntimeError(
            "Release 직전/순간에서 유효한 shoulder/elbow/wrist 좌표를 찾지 못했습니다."
        )

    window_df = pd.DataFrame(window_rows)

    below_ratio = float(
        window_df["wrist_below_shoulder"].mean()
    )

    median_screen_angle = float(
        pd.to_numeric(
            window_df["shoulder_wrist_screen_angle_deg"],
            errors="coerce",
        ).median()
    )

    # Preserve Rogers-style persistent Underhand state.
    is_underhand_state = (
        below_ratio >= underhand_below_ratio_threshold
        and median_screen_angle < underhand_angle_threshold
    )

    final_frame_used = None
    fallback_used = False
    fallback_reason = None

    # ------------------------------------------------------------------
    # v8.4 representative posture
    # ------------------------------------------------------------------
    # Do not classify from only the exact Release frame.
    # Use the median posture over Release-pre_frames ... Release.
    # This suppresses one-frame collapse while still rejecting a one-frame
    # artificial high pose.
    representative_row = {
        "wrist_height_above_shoulder_body": float(
            pd.to_numeric(
                window_df["wrist_height_above_shoulder_body"],
                errors="coerce",
            ).median()
        ),
        "elbow_height_above_shoulder_body": float(
            pd.to_numeric(
                window_df["elbow_height_above_shoulder_body"],
                errors="coerce",
            ).median()
        ),
        "shoulder_wrist_screen_angle_deg": float(
            pd.to_numeric(
                window_df["shoulder_wrist_screen_angle_deg"],
                errors="coerce",
            ).median()
        ),
        "upper_arm_screen_angle_deg": float(
            pd.to_numeric(
                window_df["upper_arm_screen_angle_deg"],
                errors="coerce",
            ).median()
        ),
    }

    representative_score, representative_details = posture_score(
        representative_row
    )

    # Choose an actual frame only for diagnostics / downstream metadata.
    # The classification itself uses the 3-frame median posture above.
    candidate_rows = []
    for _, row in window_df.iterrows():
        score_distance = abs(
            int(row["posture_score"]) - int(representative_score)
        )
        release_distance = abs(
            int(release_frame) - int(row["frame"])
        )
        candidate_rows.append(
            (score_distance, release_distance, int(row["frame"]), row)
        )

    candidate_rows.sort(key=lambda item: (item[0], item[1]))
    representative_frame_row = candidate_rows[0][3].to_dict()
    final_frame_used = int(representative_frame_row["frame"])

    if is_underhand_state:
        arm_slot = "Underhand"
        final_score = int(representative_score)
        score_details = representative_details
    else:
        arm_slot, final_score, score_details = classify_non_underhand(
            representative_row
        )

    # This is not an error fallback anymore; it records whether the
    # representative diagnostic frame differs from the exact Release frame.
    if final_frame_used != int(release_frame):
        fallback_used = True
        fallback_reason = "three_frame_representative_posture"
    else:
        fallback_used = False
        fallback_reason = None

    # Per-frame labels are useful for confidence / debugging.
    frame_labels = []
    for _, row in window_df.iterrows():
        if is_underhand_state:
            frame_labels.append("Underhand")
        else:
            label_i, _, _ = classify_non_underhand(row.to_dict())
            frame_labels.append(label_i)

    window_scores = [
        int(v)
        for v in pd.to_numeric(
            window_df["posture_score"],
            errors="coerce",
        ).dropna().tolist()
    ]

    confidence = confidence_from_window(
        arm_slot=arm_slot,
        final_score=int(final_score),
        window_scores=window_scores,
        valid_frames=len(window_df),
    )

    # Legacy-compatible numeric field:
    # use the 3-frame median shoulder->wrist screen angle.
    legacy_angle = float(
        representative_row["shoulder_wrist_screen_angle_deg"]
    )

    label_agreement = float(
        sum(label == arm_slot for label in frame_labels)
        / max(1, len(frame_labels))
    )

    summary = {
        "throwing_side": side,
        "release_frame_requested": int(release_frame),
        "analysis_start_frame": int(release_frame - pre_frames),
        "analysis_end_frame": int(release_frame),
        "valid_window_frames": int(len(window_df)),

        "wrist_below_ratio": float(below_ratio),
        "window_median_screen_shoulder_wrist_angle_deg": float(
            median_screen_angle
        ),
        "underhand_state_detected": bool(is_underhand_state),

        "classification_window": "release_minus_2_to_release_median",
        "representative_frame_used": int(final_frame_used),
        "frame_label_agreement": float(label_agreement),
        "frame_labels": frame_labels,

        "final_frame_used": int(final_frame_used),

        # Keep old downstream contract.
        "final_arm_angle_deg": legacy_angle,

        # v8.2 diagnostics.
        "final_posture_score": int(final_score),
        "final_wrist_height_above_shoulder_body": float(
            representative_row["wrist_height_above_shoulder_body"]
        ),
        "final_elbow_height_above_shoulder_body": float(
            representative_row["elbow_height_above_shoulder_body"]
        ),
        "final_shoulder_wrist_screen_angle_deg": legacy_angle,
        "final_upper_arm_screen_angle_deg": float(
            representative_row["upper_arm_screen_angle_deg"]
        ),
        "wrist_height_points": int(
            score_details["wrist_height_points"]
        ),
        "elbow_height_points": int(
            score_details["elbow_height_points"]
        ),
        "wrist_angle_points": int(
            score_details["wrist_angle_points"]
        ),
        "upper_arm_angle_points": int(
            score_details["upper_arm_angle_points"]
        ),

        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,

        "estimated_arm_slot": arm_slot,
        "arm_slot_confidence": confidence,

        # Kept only for compatibility/debug; not used for v8.2 classification.
        "overhand_threshold_deg": float(overhand_threshold),
        "three_quarter_threshold_deg": float(
            three_quarter_threshold
        ),
        "underhand_angle_threshold_deg": float(
            underhand_angle_threshold
        ),
        "underhand_below_ratio_threshold": float(
            underhand_below_ratio_threshold
        ),

        "classification_version": "v8.5",
        "classification_method": (
            "3-frame median posture with stricter Overhand gate: "
            "normalized wrist/elbow height + screen arm angles + persistent underhand state"
        ),
        "note": (
            "Single-camera 2D projected categorical estimate. "
            "Not Statcast Arm Angle or true 3D biomechanical arm slot. "
            "v8.5 thresholds are experimental/sample-calibrated."
        ),
    }

    return window_df, summary


def save_graph(
    window_df: pd.DataFrame,
    summary: dict,
    output_path: Path,
):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        window_df["frame"],
        window_df["wrist_height_above_shoulder_body"],
        marker="o",
        label="Wrist height / shoulder width",
    )

    ax.plot(
        window_df["frame"],
        window_df["elbow_height_above_shoulder_body"],
        marker="o",
        label="Elbow height / shoulder width",
    )

    ax.axhline(
        0.0,
        linestyle=":",
        alpha=0.6,
        label="Throwing shoulder height",
    )

    ax.axvline(
        summary["release_frame_requested"],
        linestyle="--",
        label=f"Release {summary['release_frame_requested']}",
    )

    ax.axvline(
        summary["final_frame_used"],
        linestyle="-.",
        label=f"Frame used {summary['final_frame_used']}",
    )

    ax.set_title(
        f"Sports2D Arm Slot v8.5 - "
        f"{summary['estimated_arm_slot']} "
        f"(score={summary['final_posture_score']})"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Height above throwing shoulder / shoulder width")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--throwing_side",
        required=True,
        choices=["left", "right"],
    )
    parser.add_argument(
        "--release_frame",
        type=int,
        required=True,
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--pre_frames",
        type=int,
        default=2,
    )

    # Kept because run_sports2d_stage3.py already passes these.
    parser.add_argument(
        "--overhand_threshold",
        type=float,
        default=78.0,
    )
    parser.add_argument(
        "--three_quarter_threshold",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--underhand_angle_threshold",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--underhand_below_ratio",
        type=float,
        default=0.67,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 CSV 없음: {input_path}"
        )

    label = (
        args.label.strip()
        if args.label
        else input_path.stem
    )

    df = pd.read_csv(input_path)

    window_df, summary = run_arm_slot_v8_4(
        df=df,
        throwing_side=args.throwing_side,
        release_frame=args.release_frame,
        pre_frames=args.pre_frames,
        overhand_threshold=args.overhand_threshold,
        three_quarter_threshold=args.three_quarter_threshold,
        underhand_angle_threshold=args.underhand_angle_threshold,
        underhand_below_ratio_threshold=args.underhand_below_ratio,
    )

    summary["label"] = label

    # Keep legacy filenames for current Stage3 compatibility.
    window_csv_path = (
        output_dir / f"{label}_arm_slot_v7_3_window.csv"
    )
    summary_csv_path = (
        output_dir / "arm_slot_classification_summary_v7_3.csv"
    )
    summary_json_path = (
        output_dir / "arm_slot_classification_summary_v7_3.json"
    )
    graph_path = (
        output_dir / f"{label}_arm_slot_v7_3.png"
    )

    window_df.to_csv(
        window_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([summary]).to_csv(
        summary_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        summary_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_graph(
        window_df,
        summary,
        graph_path,
    )

    print()
    print("===== Sports2D Arm Slot Classifier v8.5 =====")
    print(f"Label                   : {label}")
    print(f"Throwing side           : {summary['throwing_side']}")
    print(
        f"Release requested       : "
        f"{summary['release_frame_requested']}"
    )
    print(
        f"Analysis window         : "
        f"{summary['analysis_start_frame']} ~ "
        f"{summary['analysis_end_frame']}"
    )
    print(
        f"Valid window frames     : "
        f"{summary['valid_window_frames']}"
    )
    print(
        f"Representative method   : "
        f"{summary['classification_window']}"
    )
    print(
        f"Frame labels            : "
        f"{summary['frame_labels']}"
    )
    print(
        f"Frame label agreement   : "
        f"{summary['frame_label_agreement']:.3f}"
    )
    print()

    print(
        f"Posture score           : "
        f"{summary['final_posture_score']} / 6"
    )
    print(
        f"Wrist height / shoulder : "
        f"{summary['final_wrist_height_above_shoulder_body']:.3f}"
    )
    print(
        f"Elbow height / shoulder : "
        f"{summary['final_elbow_height_above_shoulder_body']:.3f}"
    )
    print(
        f"Shoulder→Wrist angle    : "
        f"{summary['final_shoulder_wrist_screen_angle_deg']:.2f} deg"
    )
    print(
        f"Shoulder→Elbow angle    : "
        f"{summary['final_upper_arm_screen_angle_deg']:.2f} deg"
    )
    print(
        f"Score components        : "
        f"wrist_h={summary['wrist_height_points']}, "
        f"elbow_h={summary['elbow_height_points']}, "
        f"wrist_ang={summary['wrist_angle_points']}, "
        f"upper_ang={summary['upper_arm_angle_points']}"
    )
    print()

    print(
        f"Wrist below ratio       : "
        f"{summary['wrist_below_ratio']:.3f}"
    )
    print(
        f"Underhand state?        : "
        f"{summary['underhand_state_detected']}"
    )
    print()

    print(
        f"Final frame used        : "
        f"{summary['final_frame_used']}"
    )
    print(
        f"Fallback used           : "
        f"{summary['fallback_used']}"
    )
    print(
        f"Fallback reason         : "
        f"{summary['fallback_reason']}"
    )
    print()

    print(
        f"Estimated Arm Slot      : "
        f"{summary['estimated_arm_slot']}"
    )
    print(
        f"Confidence              : "
        f"{summary['arm_slot_confidence']}"
    )
    print()

    print(
        "※ v8.5는 Release-2F~Release 대표 자세를 사용하며 Overhand 조건을 더 엄격하게 적용합니다. "
        "조합한 2D posture-pattern classifier입니다."
    )
    print(
        "※ Statcast Arm Angle 또는 실제 3D biomechanical arm slot이 아닙니다."
    )
    print()

    print(f"Window CSV              : {window_csv_path}")
    print(f"Summary CSV             : {summary_csv_path}")
    print(f"Summary JSON            : {summary_json_path}")
    print(f"Graph                   : {graph_path}")


if __name__ == "__main__":
    main()
