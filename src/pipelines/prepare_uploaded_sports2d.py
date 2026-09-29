#!/usr/bin/env python
"""Create the canonical Stage 2 input CSV from an uploaded pitch video.

Runs Sports2D on the selected ROI crop, then aligns its pixel TRC with the
original video's coordinates before passing it to the canonical pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sports2d_executable() -> str:
    configured = os.environ.get("SPORTS2D_EXE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SPORTS2D_EXE 경로를 찾을 수 없습니다: {path}")
        return str(path)
    name = "sports2d.exe" if os.name == "nt" else "sports2d"
    candidates = [Path(sysconfig.get_path("scripts")) / name]
    if os.name == "nt":
        candidates.append(Path.home() / ".venv" / "pose2sim" / "Scripts" / name)
    for path in candidates:
        if path.is_file():
            return str(path)
    found = shutil.which("sports2d")
    if found:
        return found
    raise FileNotFoundError(
        "Sports2D 실행 파일이 없습니다. Sports2D 환경을 활성화하거나 "
        "SPORTS2D_EXE를 sports2d.exe의 절대 경로로 지정하세요."
    )


def _pose_center(pose: pd.DataFrame) -> tuple[float, float]:
    """Median upper-body center near the ROI selection's first frame."""
    centers = []
    for _, row in pose.head(15).iterrows():
        xs, ys = [], []
        for joint in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            x, y = row.get(f"{joint}_x"), row.get(f"{joint}_y")
            if pd.notna(x) and pd.notna(y):
                xs.append(float(x))
                ys.append(float(y))
        if len(xs) >= 2:
            centers.append((float(np.mean(xs)), float(np.mean(ys))))
    if not centers:
        raise ValueError("ROI 좌표 확인에 필요한 몸통 관절이 없습니다.")
    return tuple(float(np.median([p[i] for p in centers])) for i in (0, 1))


def restore_roi_coordinates(pose: pd.DataFrame, roi: dict) -> tuple[pd.DataFrame, str]:
    """Choose crop-local or already-global coordinates against the selected pitcher ROI.

    Some Sports2D exports are already expressed in original video pixels.
    Blindly adding crop offsets would move the skeleton off the pitcher.
    """
    selected, expanded = roi["selected_roi"], roi["expanded_roi"]
    dx, dy = int(expanded["offset_x"]), int(expanded["offset_y"])
    center_x, center_y = _pose_center(pose)
    target_x = float(selected["x"]) + float(selected["width"]) / 2
    target_y = float(selected["y"]) + float(selected["height"]) / 2
    width, height = max(1, float(selected["width"])), max(1, float(selected["height"]))

    def score(x: float, y: float) -> float:
        return ((x - target_x) / width) ** 2 + ((y - target_y) / height) ** 2

    unshifted = score(center_x, center_y)
    shifted = score(center_x + dx, center_y + dy)
    if min(unshifted, shifted) > 2.25:
        raise ValueError(
            "Sports2D 몸통 위치가 선택한 투수 영역과 맞지 않습니다. "
            "ROI에 다른 인물이 포함됐는지 확인하고 다시 선택하세요."
        )
    if unshifted <= shifted:
        return pose, "TRC 좌표가 이미 원본 영상 위치와 일치해 ROI 오프셋을 더하지 않았습니다."
    for col in pose.columns:
        if col.endswith("_x"):
            pose[col] = pd.to_numeric(pose[col], errors="coerce") + dx
        elif col.endswith("_y"):
            pose[col] = pd.to_numeric(pose[col], errors="coerce") + dy
    return pose, f"크롭 좌표를 원본 영상으로 복원했습니다: x+{dx}, y+{dy}"


def prepare(video: Path, pitch_id: str, roi_json: Path, force: bool = False,
            reuse_trc: bool = False) -> Path:
    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"업로드 영상이 없습니다: {video}")
    roi_json = roi_json.resolve()
    if not roi_json.is_file():
        raise FileNotFoundError(f"투수 ROI가 지정되지 않았습니다: {roi_json}")
    roi = json.loads(roi_json.read_text(encoding="utf-8"))
    crop = Path(roi["output_video"]).resolve()
    if not crop.is_file() or crop.stat().st_size == 0:
        raise FileNotFoundError(f"ROI 크롭 영상을 찾을 수 없습니다: {crop}")
    if Path(roi["input_video"]).resolve() != video:
        raise ValueError("ROI가 다른 영상에 속합니다. 투수 영역을 다시 선택하세요.")
    if not pitch_id or not all(c.isascii() and (c.isalnum() or c in "_-") for c in pitch_id):
        raise ValueError("pitch_id에는 영문, 숫자, 밑줄, 하이픈만 사용할 수 있습니다.")

    output = ROOT / "sports2d" / "input" / f"{pitch_id}_sports2d_pose.csv"
    if output.is_file() and not force:
        print(f"[REUSE] Sports2D Stage 2 input: {output}", flush=True)
        return output

    converter = ROOT / "src" / "sports2d" / "trc_to_pose_csv.py"
    if not converter.is_file():
        raise FileNotFoundError(f"TRC 변환 스크립트가 없습니다: {converter}")
    results = ROOT / "sports2d" / "raw_results" / pitch_id
    results.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not reuse_trc:
        cmd = [
            sports2d_executable(),
            "--video_input", str(crop),
            "--result_dir", str(results),
            "--nb_persons_to_detect", "1",
            "--person_ordering_method", "largest_size",
            "--show_realtime_results", "false",
            "--show_graphs", "false",
            "--save_pose", "true",
            "--save_vid", "true",
        ]
        print("[SPORTS2D] 선택한 투수 영역에서 관절 추출 중:", crop, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    candidates = sorted(
        (p for p in results.rglob("*_px_person00.trc")
         if reuse_trc or p.stat().st_mtime >= started - 2),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Sports2D는 끝났지만 픽셀 TRC를 찾지 못했습니다: {results}"
        )
    trc = candidates[0]
    if reuse_trc and trc.stat().st_mtime_ns < crop.stat().st_mtime_ns:
        raise ValueError("기존 TRC가 현재 ROI 영상보다 오래됐습니다. Sports2D를 다시 실행하세요.")
    print("[SPORTS2D] 픽셀 TRC:", trc, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(converter), "--trc", str(trc), "--output", str(output)],
        cwd=ROOT, check=True,
    )
    pose = pd.read_csv(output)
    pose, coordinate_message = restore_roi_coordinates(pose, roi)
    pose.to_csv(output, index=False, encoding="utf-8-sig")
    print("[ROI] " + coordinate_message, flush=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Stage 2 입력 CSV가 생성되지 않았습니다.")
    print("[READY] Stage 2 입력 CSV:", output, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pitch-id", required=True)
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-trc", action="store_true",
                        help="기존 TRC에서 CSV만 다시 만들고 Sports2D 재추출은 생략")
    args = parser.parse_args()
    prepare(args.video, args.pitch_id, args.roi_json, args.force or args.reuse_trc,
            args.reuse_trc)


if __name__ == "__main__":
    main()
