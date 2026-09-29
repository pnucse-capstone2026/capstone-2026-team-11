#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_sports2d_stage3.py

Final Integrated Sports2D Stage 3 Pipeline
------------------------------------------

Pipeline:
1) Preliminary Release Anchor v1
2) Front Foot Contact v1.5
3) Final Release Detection v2
4) Arm Slot Classifier v7.3

최종 구조:
Sports2D wrist-motion validated CSV
    ↓
Preliminary Release Anchor
    ↓
Front Foot Contact
    ↓
Final Release Candidate
    ↓
Arm Slot Classification
    ↓
Integrated Summary JSON / CSV

필수 스크립트(src/sports2d):
- preliminary_release_anchor.py
- front_foot_contact.py
- final_release_detection.py
- arm_slot_classifier.py

예시:
python src\\run_sports2d_stage3.py ^
  --input "sports2d_ab\\verlander_01\\pose_csv\\verlander_01_sports2d_wrist_motion_validated.csv" ^
  --throwing_side right ^
  --label verlander_01 ^
  --output_dir "results\\sports2d_stage3_final\\verlander_01"
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPORTS2D_SCRIPT_DIR = PROJECT_ROOT / "src" / "sports2d"



def resolve_script(script_dir: Path, filename: str) -> Path:
    path = script_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"필수 스크립트를 찾지 못했습니다: {path}"
        )

    return path


def run_command(cmd: list[str], title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(
        " ".join(
            f'"{x}"' if " " in str(x) else str(x)
            for x in cmd
        )
    )
    print()

    # stdout/stderr를 부모 콘솔로 직접 넘긴다.
    # Windows CMD에서 capture_output + UTF-8 decode 시
    # 한글이 깨지는 현상을 줄이기 위한 방식.
    proc = subprocess.run(cmd)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{title} 실패 (return code={proc.returncode})"
        )


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"결과 JSON 없음: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_summary_csv(
    path: Path,
    summary: dict,
):
    flat = {}

    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(
                value,
                ensure_ascii=False,
            )
        else:
            flat[key] = value

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(flat.keys()),
        )

        writer.writeheader()
        writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Sports2D wrist_motion_validated CSV",
    )

    parser.add_argument(
        "--throwing_side",
        required=True,
        choices=["left", "right"],
    )

    parser.add_argument(
        "--label",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--script_dir",
        default=None,
        help="Sports2D 개별 stage script 폴더. 기본값은 src/sports2d",
    )

    # Preliminary Release Anchor
    parser.add_argument(
        "--disagreement_sec",
        type=float,
        default=0.08,
    )

    # FC v1.5
    parser.add_argument(
        "--fc_min_offset_sec",
        type=float,
        default=-0.22,
    )

    parser.add_argument(
        "--fc_target_offset_sec",
        type=float,
        default=-0.10,
    )

    parser.add_argument(
        "--fc_max_offset_sec",
        type=float,
        default=0.04,
    )

    # Final Release v2
    parser.add_argument(
        "--release_search_min_sec",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--release_search_max_sec",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--release_normal_min_sec",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--release_normal_max_sec",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--release_early_anchor_sec",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--release_fallback_min_sec",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--release_fallback_target_sec",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--release_fallback_max_sec",
        type=float,
        default=0.18,
    )

    # Arm Slot v7.3
    parser.add_argument(
        "--arm_slot_pre_frames",
        type=int,
        default=2,
    )

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

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 CSV 없음: {input_path}"
        )

    label = (
        args.label.strip()
        if args.label
        else input_path.stem
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    script_dir = (
        Path(args.script_dir)
        if args.script_dir
        else SPORTS2D_SCRIPT_DIR
    )

    preliminary_script = resolve_script(
        script_dir,
        "preliminary_release_anchor.py",
    )

    fc_script = resolve_script(
        script_dir,
        "front_foot_contact.py",
    )

    final_release_script = resolve_script(
        script_dir,
        "final_release_detection.py",
    )

    arm_slot_script = resolve_script(
        script_dir,
        "arm_slot_classifier.py",
    )

    python_exe = sys.executable

    # ========================================================
    # STEP 1 - Preliminary Release Anchor
    # ========================================================

    preliminary_dir = (
        output_dir
        /
        "01_preliminary_release"
    )

    preliminary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        python_exe,
        str(preliminary_script),
        "--input",
        str(input_path),
        "--output_dir",
        str(preliminary_dir),
        "--disagreement_sec",
        str(args.disagreement_sec),
    ]

    run_command(
        cmd,
        "STEP 1/4 - Preliminary Release Anchor v1",
    )

    preliminary_summary_path = (
        preliminary_dir
        /
        "preliminary_release_anchor_v1.json"
    )

    preliminary = read_json(
        preliminary_summary_path
    )

    preliminary_anchor = int(
        preliminary[
            "preliminary_release_anchor_frame"
        ]
    )

    # ========================================================
    # STEP 2 - Front Foot Contact v1.5
    # ========================================================

    fc_dir = (
        output_dir
        /
        "02_front_foot_contact"
    )

    fc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        python_exe,
        str(fc_script),
        "--input",
        str(input_path),
        "--throwing_side",
        args.throwing_side,
        "--preliminary_anchor_frame",
        str(preliminary_anchor),
        "--output_dir",
        str(fc_dir),
        "--min_offset_sec",
        str(args.fc_min_offset_sec),
        "--target_offset_sec",
        str(args.fc_target_offset_sec),
        "--max_offset_sec",
        str(args.fc_max_offset_sec),
    ]

    run_command(
        cmd,
        "STEP 2/4 - Front Foot Contact v1.5",
    )

    fc_summary_path = (
        fc_dir
        /
        "fc_v1_5_summary.json"
    )

    fc_summary = read_json(
        fc_summary_path
    )

    fc_frame = int(
        fc_summary[
            "front_foot_contact_frame"
        ]
    )

    # ========================================================
    # STEP 3 - Final Release Detection v2
    # ========================================================

    release_dir = (
        output_dir
        /
        "03_final_release"
    )

    release_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        python_exe,
        str(final_release_script),
        "--input",
        str(input_path),
        "--fc_frame",
        str(fc_frame),
        "--preliminary_anchor_frame",
        str(preliminary_anchor),
        "--output_dir",
        str(release_dir),
        "--search_min_sec",
        str(args.release_search_min_sec),
        "--search_max_sec",
        str(args.release_search_max_sec),
        "--normal_min_sec",
        str(args.release_normal_min_sec),
        "--normal_max_sec",
        str(args.release_normal_max_sec),
        "--early_anchor_sec",
        str(args.release_early_anchor_sec),
        "--fallback_min_sec",
        str(args.release_fallback_min_sec),
        "--fallback_target_sec",
        str(args.release_fallback_target_sec),
        "--fallback_max_sec",
        str(args.release_fallback_max_sec),
    ]

    run_command(
        cmd,
        "STEP 3/4 - Final Release Detection v2",
    )

    release_summary_path = (
        release_dir
        /
        "final_release_v2_summary.json"
    )

    release_summary = read_json(
        release_summary_path
    )

    final_release_frame = int(
        release_summary[
            "final_release_frame"
        ]
    )

    # ========================================================
    # STEP 4 - Arm Slot v7.3
    # ========================================================

    arm_slot_dir = (
        output_dir
        /
        "04_arm_slot"
    )

    arm_slot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        python_exe,
        str(arm_slot_script),
        "--input",
        str(input_path),
        "--throwing_side",
        args.throwing_side,
        "--release_frame",
        str(final_release_frame),
        "--label",
        label,
        "--output_dir",
        str(arm_slot_dir),
        "--pre_frames",
        str(args.arm_slot_pre_frames),
        "--overhand_threshold",
        str(args.overhand_threshold),
        "--three_quarter_threshold",
        str(args.three_quarter_threshold),
        "--underhand_angle_threshold",
        str(args.underhand_angle_threshold),
        "--underhand_below_ratio",
        str(args.underhand_below_ratio),
    ]

    run_command(
        cmd,
        "STEP 4/4 - Arm Slot Classifier v7.3",
    )

    arm_slot_summary_path = (
        arm_slot_dir
        /
        "arm_slot_classification_summary_v7_3.json"
    )

    arm_slot_summary = read_json(
        arm_slot_summary_path
    )

    # ========================================================
    # Integrated Summary
    # ========================================================

    integrated = {
        "label":
            label,

        "input_csv":
            str(input_path),

        "throwing_side":
            args.throwing_side,

        "fps":
            release_summary.get(
                "fps",
                preliminary.get("fps"),
            ),

        # preliminary
        "preliminary_release_anchor_frame":
            preliminary_anchor,

        "preliminary_anchor_source":
            preliminary.get(
                "anchor_source"
            ),

        "preliminary_anchor_confidence":
            preliminary.get(
                "anchor_confidence"
            ),

        # FC
        "front_foot_contact_frame":
            fc_frame,

        "fc_offset_sec_from_preliminary_anchor":
            fc_summary.get(
                "fc_offset_sec_from_anchor"
            ),

        # final release
        "final_release_frame":
            final_release_frame,

        "fc_to_final_release_frames":
            release_summary.get(
                "fc_to_release_frames"
            ),

        "fc_to_final_release_sec":
            release_summary.get(
                "fc_to_release_sec"
            ),

        "final_release_source":
            release_summary.get(
                "release_source"
            ),

        "final_release_confidence":
            release_summary.get(
                "release_confidence"
            ),

        # arm slot
        "arm_slot":
            arm_slot_summary.get(
                "estimated_arm_slot"
            ),

        "arm_slot_confidence":
            arm_slot_summary.get(
                "arm_slot_confidence"
            ),

        "arm_slot_angle_deg":
            arm_slot_summary.get(
                "final_arm_angle_deg"
            ),

        "arm_slot_frame_used":
            arm_slot_summary.get(
                "final_frame_used"
            ),

        "arm_slot_fallback_used":
            arm_slot_summary.get(
                "fallback_used"
            ),

        "arm_slot_fallback_reason":
            arm_slot_summary.get(
                "fallback_reason"
            ),

        "arm_slot_underhand_state":
            arm_slot_summary.get(
                "underhand_state_detected"
            ),

        "notes": {
            "release": (
                "Wrist-motion based automatic release candidate; "
                "not direct ball-separation ground truth."
            ),

            "arm_slot": (
                "2D projected shoulder-to-wrist based categorical estimate; "
                "not Statcast Arm Angle or 3D biomechanical arm slot."
            ),
        },
    }

    summary_json_path = (
        output_dir
        /
        "stage3_events_final_summary.json"
    )

    summary_csv_path = (
        output_dir
        /
        "stage3_events_final_summary.csv"
    )

    with open(
        summary_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            integrated,
            f,
            ensure_ascii=False,
            indent=2,
        )

    write_summary_csv(
        summary_csv_path,
        integrated,
    )

    # ========================================================
    # Final console output
    # ========================================================

    print()
    print("=" * 80)
    print("Sports2D Stage 3 FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Label                   : {label}"
    )

    print(
        f"Throwing side           : {args.throwing_side}"
    )

    print()

    print(
        f"Preliminary Anchor      : "
        f"{preliminary_anchor} "
        f"({integrated['preliminary_anchor_source']}, "
        f"{integrated['preliminary_anchor_confidence']})"
    )

    print(
        f"Front Foot Contact      : "
        f"{fc_frame}"
    )

    print(
        f"Final Release Candidate : "
        f"{final_release_frame} "
        f"({integrated['final_release_source']}, "
        f"{integrated['final_release_confidence']})"
    )

    print(
        f"FC -> Release           : "
        f"{integrated['fc_to_final_release_frames']} frames "
        f"({integrated['fc_to_final_release_sec']:.4f} sec)"
    )

    print()

    print(
        f"Arm Slot                : "
        f"{integrated['arm_slot']} "
        f"({integrated['arm_slot_confidence']})"
    )

    print(
        f"Arm Slot Angle          : "
        f"{integrated['arm_slot_angle_deg']:.2f} deg"
    )

    print(
        f"Arm Slot Frame Used     : "
        f"{integrated['arm_slot_frame_used']}"
    )

    print(
        f"Arm Slot Fallback       : "
        f"{integrated['arm_slot_fallback_used']}"
    )

    if integrated[
        "arm_slot_fallback_reason"
    ] is not None:
        print(
            f"Fallback Reason         : "
            f"{integrated['arm_slot_fallback_reason']}"
        )

    print()

    print(
        f"Integrated JSON         : "
        f"{summary_json_path}"
    )

    print(
        f"Integrated CSV          : "
        f"{summary_csv_path}"
    )

    print()

    print(
        "※ Final Release Candidate는 wrist-motion 기반 자동 후보입니다."
    )

    print(
        "※ Arm Slot은 단일 카메라 2D 투영 기반 범주형 분류입니다."
    )

    print(
        "※ Statcast Arm Angle 또는 실제 3D arm slot과 동일하지 않습니다."
    )


if __name__ == "__main__":
    main()
