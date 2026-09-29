#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canonical Motion Normalization

입력
----
1) Sports2D Stage 2 wrist-motion validated CSV
2) pipeline_outputs/<pitch_id>/analysis_events.json

출력
----
- motion_normalized.csv
- normalized_events.csv
- motion_normalization_summary.csv
- pitch_progress_normalization.png
- normalized_wrist_speed.png (가능할 때)
- normalized_elbow_angle.png (가능할 때)
- fc_to_release_wrist_speed.png (가능할 때)

이 버전은 final_*_v10, pitch_event_v10, release_candidate_simple 등
구버전 이벤트 컬럼에 의존하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_events_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"analysis_events.json 없음: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_event(events: dict, key: str) -> dict:
    block = events.get("events", {}).get(key, {})
    frame = block.get("frame")
    if frame is not None:
        frame = int(round(float(frame)))

    return {
        "frame": frame,
        "source": str(block.get("source", "unknown")),
        "confidence": str(block.get("confidence", "unknown")),
    }


def find_frame_index(frames: np.ndarray, target_frame: int):
    matches = np.where(frames == int(target_frame))[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def frame_to_percent(event_frame, start_frame, end_frame):
    if event_frame is None:
        return np.nan
    denominator = end_frame - start_frame
    if denominator <= 0:
        return np.nan
    return (event_frame - start_frame) / denominator * 100.0


def first_existing_column(df: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_pitch_time(csv_path, events_json_path, output_csv_path, graph_dir):
    csv_path = Path(csv_path)
    events_json_path = Path(events_json_path)
    output_csv_path = Path(output_csv_path)
    graph_dir = Path(graph_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"입력 CSV 없음: {csv_path}")

    df = pd.read_csv(csv_path)
    events_json = read_events_json(events_json_path)

    required = ["frame", "timestamp"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("필수 컬럼 없음: " + ", ".join(missing))

    frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)

    if np.isnan(frames).any():
        raise ValueError("frame 컬럼에 NaN이 있습니다.")

    frames = frames.astype(int)

    knee = get_event(events_json, "knee_lift")
    fc = get_event(events_json, "front_foot_contact")
    release = get_event(events_json, "release")

    knee_frame = knee["frame"]
    fc_frame = fc["frame"]
    release_frame = release["frame"]

    if knee_frame is None:
        raise ValueError("analysis_events.json: knee_lift.frame 없음")
    if fc_frame is None:
        raise ValueError("analysis_events.json: front_foot_contact.frame 없음")
    if release_frame is None:
        raise ValueError("analysis_events.json: release.frame 없음")

    if not (knee_frame < fc_frame < release_frame):
        raise ValueError(
            "이벤트 순서 오류: "
            f"Knee={knee_frame}, FC={fc_frame}, Release={release_frame}"
        )

    knee_idx = find_frame_index(frames, knee_frame)
    fc_idx = find_frame_index(frames, fc_frame)
    release_idx = find_frame_index(frames, release_frame)

    if knee_idx is None:
        raise ValueError(f"Knee Lift frame {knee_frame}이 입력 CSV에 없습니다.")
    if fc_idx is None:
        raise ValueError(f"FC frame {fc_frame}이 입력 CSV에 없습니다.")
    if release_idx is None:
        raise ValueError(f"Release frame {release_frame}이 입력 CSV에 없습니다.")

    knee_time = float(timestamps[knee_idx])
    fc_time = float(timestamps[fc_idx])
    release_time = float(timestamps[release_idx])

    frame_values = frames.astype(float)

    pitch_progress_percent = (
        (frame_values - knee_frame)
        / (release_frame - knee_frame)
        * 100.0
    )

    time_denominator = release_time - knee_time
    if np.isfinite(time_denominator) and time_denominator > 0:
        pitch_progress_time_percent = (
            (timestamps - knee_time)
            / time_denominator
            * 100.0
        )
    else:
        pitch_progress_time_percent = pitch_progress_percent.copy()

    fc_release_progress_percent = (
        (frame_values - fc_frame)
        / (release_frame - fc_frame)
        * 100.0
    )

    fc_release_time_denominator = release_time - fc_time
    if np.isfinite(fc_release_time_denominator) and fc_release_time_denominator > 0:
        fc_release_progress_time_percent = (
            (timestamps - fc_time)
            / fc_release_time_denominator
            * 100.0
        )
    else:
        fc_release_progress_time_percent = fc_release_progress_percent.copy()

    between_knee_and_release = (
        (frames >= knee_frame)
        & (frames <= release_frame)
    )

    between_fc_and_release = (
        (frames >= fc_frame)
        & (frames <= release_frame)
    )

    fc_percent = frame_to_percent(fc_frame, knee_frame, release_frame)

    normalization_df = pd.DataFrame(
        {
            "pitch_progress_percent": pitch_progress_percent,
            "pitch_progress_time_percent": pitch_progress_time_percent,
            "fc_release_progress_percent": fc_release_progress_percent,
            "fc_release_progress_time_percent": fc_release_progress_time_percent,
            "between_knee_and_release": between_knee_and_release,
            "between_fc_and_release": between_fc_and_release,
            "knee_lift_frame": knee_frame,
            "front_foot_contact_frame": fc_frame,
            "release_frame": release_frame,
            "knee_lift_source": knee["source"],
            "front_foot_contact_source": fc["source"],
            "release_source": release["source"],
            "knee_lift_confidence": knee["confidence"],
            "front_foot_contact_confidence": fc["confidence"],
            "release_confidence": release["confidence"],
        },
        index=df.index,
    )

    result_df = pd.concat([df.copy(), normalization_df], axis=1)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    event_summary = pd.DataFrame(
        [
            {
                "event": "Maximum Knee Lift",
                "frame": knee_frame,
                "timestamp": knee_time,
                "normalized_percent": 0.0,
                "source": knee["source"],
                "confidence": knee["confidence"],
            },
            {
                "event": "Front Foot Contact",
                "frame": fc_frame,
                "timestamp": fc_time,
                "normalized_percent": fc_percent,
                "source": fc["source"],
                "confidence": fc["confidence"],
            },
            {
                "event": "Release Candidate",
                "frame": release_frame,
                "timestamp": release_time,
                "normalized_percent": 100.0,
                "source": release["source"],
                "confidence": release["confidence"],
            },
        ]
    )

    event_summary_path = graph_dir / "normalized_events.csv"
    event_summary.to_csv(
        event_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    arm_slot = events_json.get("arm_slot", {})

    motion_summary = pd.DataFrame(
        [
            {
                "pitch_id": events_json.get("pitch_id"),
                "pitcher_name": events_json.get("pitcher_name"),
                "throwing_side": events_json.get("throwing_side"),
                "knee_lift_frame": knee_frame,
                "front_foot_contact_frame": fc_frame,
                "release_frame": release_frame,
                "knee_lift_to_fc_sec": fc_time - knee_time,
                "fc_to_release_sec": release_time - fc_time,
                "knee_lift_to_release_sec": release_time - knee_time,
                "fc_normalized_percent": fc_percent,
                "knee_lift_source": knee["source"],
                "front_foot_contact_source": fc["source"],
                "release_source": release["source"],
                "release_confidence": release["confidence"],
                "arm_slot": arm_slot.get("category"),
                "arm_slot_angle_deg": arm_slot.get("angle_deg"),
                "arm_slot_confidence": arm_slot.get("confidence"),
            }
        ]
    )

    motion_summary_path = graph_dir / "motion_normalization_summary.csv"
    motion_summary.to_csv(
        motion_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(15, 6))
    plt.plot(
        result_df["frame"],
        result_df["pitch_progress_time_percent"],
        label="Normalized Pitch Progress",
    )
    plt.axhline(y=0, linestyle="--", label="Knee Lift = 0%")
    plt.axhline(y=100, linestyle="--", label="Release = 100%")
    plt.axvline(
        x=fc_frame,
        linestyle="--",
        label=f"FC {fc_frame} ({fc_percent:.1f}%)",
    )
    plt.xlabel("Frame")
    plt.ylabel("Normalized Pitch Progress (%)")
    plt.title("Pitch Motion Time Normalization")
    plt.legend()
    plt.grid()

    progress_graph_path = graph_dir / "pitch_progress_normalization.png"
    plt.savefig(progress_graph_path, dpi=150, bbox_inches="tight")
    plt.close()

    wrist_speed_col = first_existing_column(
        result_df,
        [
            "wrist_relative_speed_body_s_validated",
            "wrist_relative_speed_body_s_smooth",
            "wrist_speed_body_s_validated",
            "wrist_speed_body_s",
        ],
    )

    wrist_graph_path = None
    fc_release_graph_path = None

    if wrist_speed_col is not None:
        wrist_values = pd.to_numeric(
            result_df[wrist_speed_col],
            errors="coerce",
        )

        valid = wrist_values.notna()

        if valid.any():
            plt.figure(figsize=(15, 7))
            plt.plot(
                result_df.loc[valid, "pitch_progress_time_percent"],
                wrist_values.loc[valid],
                label="Wrist Speed",
            )
            plt.axvline(x=0, linestyle="--", label="Knee Lift")
            plt.axvline(
                x=fc_percent,
                linestyle="--",
                label=f"FC ({fc_percent:.1f}%)",
            )
            plt.axvline(x=100, linestyle="--", label="Release Candidate")
            plt.xlabel("Normalized Pitch Progress (%)")
            plt.ylabel("Wrist Speed (body / second)")
            plt.title("Normalized Wrist Speed")
            plt.legend()
            plt.grid()

            wrist_graph_path = graph_dir / "normalized_wrist_speed.png"
            plt.savefig(wrist_graph_path, dpi=150, bbox_inches="tight")
            plt.close()

            fc_mask = result_df["between_fc_and_release"] & wrist_values.notna()

            if fc_mask.any():
                plt.figure(figsize=(15, 7))
                plt.plot(
                    result_df.loc[
                        fc_mask,
                        "fc_release_progress_time_percent",
                    ],
                    wrist_values.loc[fc_mask],
                    label="Wrist Speed",
                )
                plt.axvline(x=0, linestyle="--", label="Front Foot Contact")
                plt.axvline(x=100, linestyle="--", label="Release Candidate")
                plt.xlabel("FC -> Release Progress (%)")
                plt.ylabel("Wrist Speed (body / second)")
                plt.title("FC to Release Wrist-Speed Progress")
                plt.legend()
                plt.grid()

                fc_release_graph_path = (
                    graph_dir / "fc_to_release_wrist_speed.png"
                )
                plt.savefig(
                    fc_release_graph_path,
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

    angle_col = first_existing_column(
        result_df,
        [
            "throwing_elbow_angle_filtered",
            "throwing_elbow_angle",
            "throwing_elbow_angle_deg",
        ],
    )

    elbow_graph_path = None

    if angle_col is not None:
        angles = pd.to_numeric(result_df[angle_col], errors="coerce")
        valid_angle = angles.notna()

        if valid_angle.any():
            plt.figure(figsize=(15, 7))
            plt.plot(
                result_df.loc[valid_angle, "pitch_progress_time_percent"],
                angles.loc[valid_angle],
                label="Throwing Elbow Angle",
            )
            plt.axvline(x=0, linestyle="--", label="Knee Lift")
            plt.axvline(
                x=fc_percent,
                linestyle="--",
                label=f"FC ({fc_percent:.1f}%)",
            )
            plt.axvline(x=100, linestyle="--", label="Release Candidate")
            plt.xlabel("Normalized Pitch Progress (%)")
            plt.ylabel("2D Projected Elbow Angle (degree)")
            plt.title("Normalized Throwing Elbow Angle")
            plt.legend()
            plt.grid()

            elbow_graph_path = graph_dir / "normalized_elbow_angle.png"
            plt.savefig(elbow_graph_path, dpi=150, bbox_inches="tight")
            plt.close()

    print()
    print("===== Motion Normalization =====")
    print(f"Input CSV      : {csv_path}")
    print(f"Events JSON    : {events_json_path}")
    print(f"Knee Lift      : {knee_frame} [{knee['source']}]")
    print(f"FC             : {fc_frame} [{fc['source']}]")
    print(
        f"Release        : {release_frame} "
        f"[{release['source']}, {release['confidence']}]"
    )
    print(f"Knee -> FC     : {fc_time - knee_time:.4f} sec")
    print(f"FC -> Release  : {release_time - fc_time:.4f} sec")
    print(f"Output CSV     : {output_csv_path}")
    print(f"Event Summary  : {event_summary_path}")
    print(f"Motion Summary : {motion_summary_path}")
    print(f"Graph           : {progress_graph_path}")

    if wrist_graph_path:
        print(f"Wrist Graph     : {wrist_graph_path}")

    if elbow_graph_path:
        print(f"Elbow Graph     : {elbow_graph_path}")

    if fc_release_graph_path:
        print(f"FC-Release Graph: {fc_release_graph_path}")

    print("[DONE] Motion Normalization")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Sports2D Stage 2 wrist-motion validated CSV",
    )

    parser.add_argument(
        "--events",
        required=True,
        help="pipeline_outputs/<pitch_id>/analysis_events.json",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="motion_normalized.csv",
    )

    parser.add_argument(
        "--graph_dir",
        required=True,
        help="normalization graph/output folder",
    )

    args = parser.parse_args()

    normalize_pitch_time(
        csv_path=args.input,
        events_json_path=args.events,
        output_csv_path=args.output,
        graph_dir=args.graph_dir,
    )


if __name__ == "__main__":
    main()
