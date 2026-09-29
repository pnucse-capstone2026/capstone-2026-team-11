
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sports2D HALPE_26 TRC -> pitching_analysis CSV adapter

Usage example:
python sports2d_trc_to_pose_csv.py ^
  --trc "sports2d_results\cole_01\cole_01_Sports2D\cole_01_Sports2D_px_person00.trc" ^
  --output "data\pose_csv\cole_01_sports2d_pose.csv"

Notes
-----
- Sports2D의 *_px_person00.trc 파일을 사용하세요. meter TRC가 아니라 pixel TRC입니다.
- TRC에는 원래 keypoint likelihood가 저장되지 않으므로 *_conf 값은
  좌표가 유효하면 1.0, NaN이면 0.0으로 생성합니다.
- 기존 COCO 계열 13개 핵심 관절 + Sports2D foot keypoint를 함께 내보냅니다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


CORE_MARKERS = {
    "Nose": "nose",
    "LShoulder": "left_shoulder",
    "RShoulder": "right_shoulder",
    "LElbow": "left_elbow",
    "RElbow": "right_elbow",
    "LWrist": "left_wrist",
    "RWrist": "right_wrist",
    "LHip": "left_hip",
    "RHip": "right_hip",
    "LKnee": "left_knee",
    "RKnee": "right_knee",
    "LAnkle": "left_ankle",
    "RAnkle": "right_ankle",
}

FOOT_MARKERS = {
    "LBigToe": "left_big_toe",
    "LSmallToe": "left_small_toe",
    "LHeel": "left_heel",
    "RBigToe": "right_big_toe",
    "RSmallToe": "right_small_toe",
    "RHeel": "right_heel",
}

# Sports2D/pose-model 버전에 따라 이름이 조금 달라질 때를 대비한 별칭
MARKER_ALIASES = {
    "Nose": ["Nose", "nose"],
    "LShoulder": ["LShoulder", "LeftShoulder", "left_shoulder"],
    "RShoulder": ["RShoulder", "RightShoulder", "right_shoulder"],
    "LElbow": ["LElbow", "LeftElbow", "left_elbow"],
    "RElbow": ["RElbow", "RightElbow", "right_elbow"],
    "LWrist": ["LWrist", "LeftWrist", "left_wrist"],
    "RWrist": ["RWrist", "RightWrist", "right_wrist"],
    "LHip": ["LHip", "LeftHip", "left_hip"],
    "RHip": ["RHip", "RightHip", "right_hip"],
    "LKnee": ["LKnee", "LeftKnee", "left_knee"],
    "RKnee": ["RKnee", "RightKnee", "right_knee"],
    "LAnkle": ["LAnkle", "LeftAnkle", "left_ankle"],
    "RAnkle": ["RAnkle", "RightAnkle", "right_ankle"],
    "LBigToe": ["LBigToe", "LeftBigToe", "left_big_toe"],
    "LSmallToe": ["LSmallToe", "LeftSmallToe", "left_small_toe"],
    "LHeel": ["LHeel", "LeftHeel", "left_heel"],
    "RBigToe": ["RBigToe", "RightBigToe", "right_big_toe"],
    "RSmallToe": ["RSmallToe", "RightSmallToe", "right_small_toe"],
    "RHeel": ["RHeel", "RightHeel", "right_heel"],
}


def _read_trc_raw(path: Path):
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if len(lines) < 6:
        raise ValueError("TRC 파일이 너무 짧습니다.")

    # 보통 4번째 줄: Frame# Time Marker1 ... / 5번째 줄: X1 Y1 Z1 ...
    header_idx = None
    for i, line in enumerate(lines[:12]):
        if "Frame#" in line and "Time" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("TRC header에서 'Frame#' / 'Time'을 찾지 못했습니다.")

    marker_line = lines[header_idx].split("\t")
    axis_line = lines[header_idx + 1].split("\t")

    # Data starts one line after axis row. Blank row가 있으면 건너뜀.
    data_start = header_idx + 2
    while data_start < len(lines) and not lines[data_start].strip():
        data_start += 1

    rows = list(csv.reader(lines[data_start:], delimiter="\t"))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("TRC 데이터 행이 없습니다.")

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    # marker names: 첫 2열은 Frame#/Time, 그 뒤 각 marker가 3열씩.
    marker_names = []
    for col in range(2, min(len(marker_line), width), 3):
        name = marker_line[col].strip()
        if name:
            marker_names.append(name)
        else:
            # 드물게 marker name이 앞/뒤 칸에 위치한 파일 대응
            nearby = [marker_line[j].strip() for j in range(max(2, col-1), min(len(marker_line), col+2))]
            marker_names.append(next((x for x in nearby if x), f"marker_{len(marker_names)}"))

    numeric = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

    # 첫 두 컬럼은 Frame# / Time
    frame = numeric.iloc[:, 0]
    time = numeric.iloc[:, 1]

    coords = {}
    for i, name in enumerate(marker_names):
        base = 2 + i * 3
        if base + 1 >= numeric.shape[1]:
            break
        coords[name] = {
            "x": numeric.iloc[:, base],
            "y": numeric.iloc[:, base + 1],
            "z": numeric.iloc[:, base + 2] if base + 2 < numeric.shape[1] else pd.Series(np.nan, index=numeric.index),
        }

    return frame, time, coords


def _find_marker(coords: dict, canonical: str):
    for alias in MARKER_ALIASES.get(canonical, [canonical]):
        if alias in coords:
            return alias
    # case-insensitive fallback
    lower_map = {k.lower(): k for k in coords}
    for alias in MARKER_ALIASES.get(canonical, [canonical]):
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def convert(trc_path: Path, output_path: Path):
    frame, time, coords = _read_trc_raw(trc_path)

    out = pd.DataFrame()
    out["frame"] = frame
    out["frame_idx"] = frame
    out["time"] = time
    out["timestamp_sec"] = time
    out["pose_source"] = "Sports2D_HALPE26"

    missing = []

    all_markers = {}
    all_markers.update(CORE_MARKERS)
    all_markers.update(FOOT_MARKERS)

    for canonical, dest in all_markers.items():
        found = _find_marker(coords, canonical)
        if found is None:
            out[f"{dest}_x"] = np.nan
            out[f"{dest}_y"] = np.nan
            out[f"{dest}_conf"] = 0.0
            missing.append(canonical)
            continue

        x = pd.to_numeric(coords[found]["x"], errors="coerce")
        y = pd.to_numeric(coords[found]["y"], errors="coerce")
        valid = np.isfinite(x) & np.isfinite(y)

        out[f"{dest}_x"] = x
        out[f"{dest}_y"] = y
        # TRC는 likelihood를 보존하지 않으므로 presence flag 역할만 함
        out[f"{dest}_conf"] = valid.astype(float)

    # 기존 파이프라인에서 자주 쓰기 편한 validity
    core_cols = []
    for dest in CORE_MARKERS.values():
        core_cols.append(f"{dest}_conf")
    out["sports2d_core_valid_ratio"] = out[core_cols].mean(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("Sports2D TRC -> Pose CSV 변환 완료")
    print(f"TRC    : {trc_path}")
    print(f"Output : {output_path}")
    print(f"Frames : {len(out)}")
    print(f"TRC markers found ({len(coords)}):")
    print(", ".join(coords.keys()))
    if missing:
        print("\n[WARNING] 찾지 못한 marker:")
        print(", ".join(missing))
    else:
        print("\n모든 핵심/발 marker를 찾았습니다.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trc", required=True, help="Sports2D *_px_person00.trc")
    parser.add_argument("--output", required=True, help="output CSV path")
    args = parser.parse_args()

    trc = Path(args.trc)
    out = Path(args.output)

    if not trc.exists():
        raise FileNotFoundError(f"TRC 파일이 없습니다: {trc}")

    if "_m_" in trc.name.lower():
        print("[WARNING] meter TRC로 보입니다. *_px_person00.trc 사용을 권장합니다.")

    convert(trc, out)


if __name__ == "__main__":
    main()
