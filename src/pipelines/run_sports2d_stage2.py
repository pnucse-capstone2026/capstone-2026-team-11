#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sports2D CSV를 공통 2D 분석 모듈에 연결하는 Stage 2 runner.

Pipeline:
Sports2D pose CSV
-> Joint Angle
-> Angle Validation
-> Wrist Motion
-> Wrist Speed Validation

예:
python src\pipelines\run_sports2d_stage2.py ^
  --input "data\pose_csv\cole_01_sports2d_pose.csv" ^
  --pitch_id cole_01 ^
  --throwing_side right
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = PROJECT_ROOT / "src" / "core"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sports2d_ab"


JOINTS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def load_module(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"필요한 기존 스크립트를 찾을 수 없습니다: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 로드 실패: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def first_existing(df: pd.DataFrame, names: list[str]):
    for name in names:
        if name in df.columns:
            return name
    return None


def prepare_sports2d_csv(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """
    Sports2D adapter CSV를 기존 프로젝트의 'clean pose' 형식으로 맞춘다.

    핵심 원칙:
    - Sports2D는 이미 interpolation + Hampel + One-Euro 후처리를 거친 좌표이므로
      A/B 테스트에서는 별도 pose cleaning을 한 번 더 하지 않는다.
    - 원 좌표 x/y를 x_clean/y_clean으로 복사한다.
    - TRC에는 원래 likelihood가 없으므로 adapter의 conf는 좌표 존재 여부 flag다.
    - *_invalid는 좌표 NaN 여부로 생성한다.
    """
    df = pd.read_csv(input_csv)

    frame_col = first_existing(df, ["frame", "frame_idx", "frame_id", "frame_number"])
    if frame_col is None:
        df["frame"] = np.arange(len(df), dtype=int)
    elif frame_col != "frame":
        df["frame"] = pd.to_numeric(df[frame_col], errors="coerce")

    timestamp_col = first_existing(
        df,
        ["timestamp", "timestamp_sec", "time", "time_sec"]
    )
    if timestamp_col is None:
        raise ValueError(
            "Sports2D CSV에 timestamp/time 계열 컬럼이 없습니다. "
            "sports2d_trc_to_pose_csv.py로 만든 CSV인지 확인하세요."
        )
    if timestamp_col != "timestamp":
        df["timestamp"] = pd.to_numeric(df[timestamp_col], errors="coerce")

    missing_core = []

    for joint in JOINTS:
        x = f"{joint}_x"
        y = f"{joint}_y"

        if x not in df.columns or y not in df.columns:
            if joint != "nose":
                missing_core.append(joint)
            continue

        x_num = pd.to_numeric(df[x], errors="coerce")
        y_num = pd.to_numeric(df[y], errors="coerce")

        df[f"{joint}_x_clean"] = x_num
        df[f"{joint}_y_clean"] = y_num

        valid = np.isfinite(x_num) & np.isfinite(y_num)

        conf_col = f"{joint}_conf"
        if conf_col not in df.columns:
            df[conf_col] = valid.astype(float)
        else:
            df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0)

        df[f"{joint}_invalid"] = ~valid

    if missing_core:
        raise ValueError(
            "기존 파이프라인에 필요한 Sports2D 관절 컬럼이 없습니다: "
            + ", ".join(missing_core)
        )

    df["pose_source"] = "Sports2D_HALPE26_postprocessed"
    df["sports2d_stage2_prepared"] = True

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return df


def summarize_outputs(angle_csv: Path, angle_valid_csv: Path,
                      wrist_csv: Path, wrist_valid_csv: Path,
                      throwing_side: str):
    angle = pd.read_csv(angle_csv)
    angle_valid = pd.read_csv(angle_valid_csv)
    wrist = pd.read_csv(wrist_csv)
    wrist_valid = pd.read_csv(wrist_valid_csv)

    print("\n" + "=" * 72)
    print("Sports2D Stage 2 결과 요약")
    print("=" * 72)

    if "throwing_elbow_angle" in angle:
        s = pd.to_numeric(angle["throwing_elbow_angle"], errors="coerce")
        print(f"Elbow angle valid(raw) : {s.notna().sum()} / {len(s)}")
        print(f"Elbow angle min/max    : {s.min():.2f} / {s.max():.2f} deg")

    if "throwing_elbow_angle_valid" in angle_valid:
        ev = angle_valid["throwing_elbow_angle_valid"].astype(bool)
        print(f"Validated elbow frames : {int(ev.sum())} / {len(ev)}")

    if "wrist_relative_speed_body_s_smooth" in wrist:
        s = pd.to_numeric(
            wrist["wrist_relative_speed_body_s_smooth"], errors="coerce"
        )
        if s.notna().any():
            idx = s.idxmax()
            print(
                "Raw wrist peak          : "
                f"frame {int(wrist.loc[idx, 'frame'])}, "
                f"{float(s.loc[idx]):.3f} body/s"
            )

    if "wrist_speed_valid" in wrist_valid:
        wv = wrist_valid["wrist_speed_valid"].astype(bool)
        print(f"Validated wrist frames : {int(wv.sum())} / {len(wv)}")

    speed_col = "wrist_relative_speed_body_s_validated"
    if speed_col in wrist_valid:
        s = pd.to_numeric(wrist_valid[speed_col], errors="coerce")
        if s.notna().any():
            idx = s.idxmax()
            print(
                "Validated wrist peak    : "
                f"frame {int(wrist_valid.loc[idx, 'frame'])}, "
                f"{float(s.loc[idx]):.3f} body/s"
            )

    print(f"Throwing side           : {throwing_side}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Sports2D adapter CSV")
    ap.add_argument("--pitch_id", required=True)
    ap.add_argument("--throwing_side", choices=["left", "right"], required=True)
    ap.add_argument(
        "--core_dir",
        default=str(CORE_DIR),
        help="공통 분석 모듈이 있는 src/core 폴더"
    )
    ap.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Sports2D Stage 2 출력 루트"
    )
    args = ap.parse_args()

    input_csv = Path(args.input)
    core_dir = Path(args.core_dir)
    root = Path(args.output_root) / args.pitch_id
    pose_dir = root / "pose_csv"
    graph_root = root / "graphs"

    prepared_csv = pose_dir / f"{args.pitch_id}_sports2d_pose_prepared.csv"
    joint_csv = pose_dir / f"{args.pitch_id}_sports2d_joint_angles.csv"
    joint_valid_csv = pose_dir / f"{args.pitch_id}_sports2d_joint_angles_validated_v2.csv"
    wrist_csv = pose_dir / f"{args.pitch_id}_sports2d_wrist_motion.csv"
    wrist_valid_csv = pose_dir / f"{args.pitch_id}_sports2d_wrist_motion_validated.csv"

    print("\n[0/4] Sports2D CSV -> 기존 clean-pose 인터페이스 변환")
    prepared = prepare_sports2d_csv(input_csv, prepared_csv)
    print(f"Prepared CSV: {prepared_csv}")
    print(f"Frames      : {len(prepared)}")

    # 파일명은 사용자의 현재 src 기준.
    joint_mod = load_module(
        "sports2d_stage2_joint",
        core_dir / "joint_angle_analysis.py"
    )
    angle_val_mod = load_module(
        "sports2d_stage2_angle_validation",
        core_dir / "angle_validation.py"
    )
    wrist_mod = load_module(
        "sports2d_stage2_wrist",
        core_dir / "wrist_motion_analysis.py"
    )
    wrist_val_mod = load_module(
        "sports2d_stage2_wrist_validation",
        core_dir / "wrist_speed_validation.py"
    )

    print("\n[1/4] Joint Angle Analysis")
    joint_mod.calculate_joint_angles(
        csv_path=str(prepared_csv),
        output_csv_path=str(joint_csv),
        graph_dir=str(graph_root / "joint_angles"),
        throwing_side=args.throwing_side,
    )

    if not joint_csv.exists():
        raise RuntimeError("Joint Angle 결과 CSV가 생성되지 않았습니다.")

    print("\n[2/4] Angle Validation v2")
    angle_val_mod.validate_elbow_angles_v2(
        csv_path=str(joint_csv),
        output_csv_path=str(joint_valid_csv),
        graph_dir=str(graph_root / "angle_validation"),
        throwing_side=args.throwing_side,
    )

    if not joint_valid_csv.exists():
        raise RuntimeError("Angle Validation 결과 CSV가 생성되지 않았습니다.")

    print("\n[3/4] Wrist Motion Analysis")
    wrist_mod.analyze_wrist_motion(
        csv_path=str(joint_valid_csv),
        output_csv_path=str(wrist_csv),
        graph_dir=str(graph_root / "wrist_motion"),
        throwing_side=args.throwing_side,
    )

    if not wrist_csv.exists():
        raise RuntimeError("Wrist Motion 결과 CSV가 생성되지 않았습니다.")

    print("\n[4/4] Wrist Speed Validation")
    wrist_val_mod.validate_wrist_speed(
        csv_path=str(wrist_csv),
        output_csv_path=str(wrist_valid_csv),
        graph_dir=str(graph_root / "wrist_validation"),
        throwing_side=args.throwing_side,
    )

    if not wrist_valid_csv.exists():
        raise RuntimeError("Wrist Validation 결과 CSV가 생성되지 않았습니다.")

    summarize_outputs(
        joint_csv,
        joint_valid_csv,
        wrist_csv,
        wrist_valid_csv,
        args.throwing_side,
    )

    print("\n생성 위치:")
    print(f"  {root}")
    print("\n기존 pipeline_outputs/data 결과는 수정하지 않았습니다.")


if __name__ == "__main__":
    main()
