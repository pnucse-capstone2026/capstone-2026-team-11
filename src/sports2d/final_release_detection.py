#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
21_sports2d_final_release_detection_v2.py

Sports2D Final Release Detection v2
-----------------------------------

목적
----
확정된 Front Foot Contact(FC) 이후 구간에서 최종 Release 후보를 찾는다.

구조
----
1) FC 이후만 검색한다.
2) preliminary release anchor를 참고한다.
3) validated wrist-speed peak가 정상적인 위치에 있으면 우선 사용한다.
4) preliminary anchor가 FC 직후 너무 이른 경우(예: Rogers),
   raw/validated peak를 그대로 믿지 않고 FC 이후 plausible release window에서
   raw wrist-speed local peak + timing prior를 사용한다.
5) 최종 Release는 "공의 실제 분리 순간"에 대한 직접 관측이 아니라
   wrist-motion 기반 자동 Release 후보이다.

기본 timing
-----------
- 전체 검색: FC + 0.02 ~ 0.30 sec
- 정상적인 release timing: FC + 0.06 ~ 0.20 sec
- extreme fallback timing: FC + 0.08 ~ 0.18 sec
- fallback target: FC + 0.12 sec

예시
----
python src\\21_sports2d_final_release_detection_v2.py ^
  --input "sports2d_ab\\verlander_01\\pose_csv\\verlander_01_sports2d_wrist_motion_validated.csv" ^
  --fc_frame 191 ^
  --preliminary_anchor_frame 197 ^
  --output_dir "results\\sports2d_final_release_v2\\verlander_01"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def estimate_fps(df: pd.DataFrame) -> float:
    for col in ["timestamp", "timestamp_sec", "time"]:
        if col in df.columns:
            t = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if len(dt):
                fps = 1.0 / np.nanmedian(dt)
                if np.isfinite(fps) and fps > 0:
                    return float(fps)
    return 30.0


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)

    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t", "valid"])
    )


def numeric_count(s: pd.Series) -> int:
    v = pd.to_numeric(s, errors="coerce")
    return int(np.isfinite(v.to_numpy(dtype=float)).sum())


def detect_speed_col(
    df: pd.DataFrame,
    explicit: str | None = None,
) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"speed column 없음: {explicit}")
        return explicit

    exact = [
        "wrist_relative_speed_body_s_smooth",
        "throwing_wrist_speed_normalized",
        "throwing_wrist_speed_norm",
        "wrist_speed_normalized",
        "wrist_speed_norm",
        "wrist_speed_body_per_sec",
        "normalized_wrist_speed",
        "wrist_relative_speed_normalized",
        "throwing_wrist_speed",
        "wrist_speed",
    ]

    for col in exact:
        if col in df.columns and numeric_count(df[col]) > 0:
            return col

    scored = []

    for col in df.columns:
        low = col.lower()

        if "wrist" not in low:
            continue

        if "speed" not in low and "velocity" not in low:
            continue

        if any(x in low for x in ["valid", "outlier", "candidate"]):
            continue

        n = numeric_count(df[col])
        if n == 0:
            continue

        score = 0
        if "smooth" in low:
            score += 5
        if "relative" in low:
            score += 4
        if "body" in low:
            score += 4
        if "norm" in low:
            score += 3
        if "throwing" in low:
            score += 2

        scored.append((score, n, col))

    if not scored:
        raise RuntimeError(
            "wrist speed 컬럼 자동 탐지 실패. --speed_col로 직접 지정하세요."
        )

    scored.sort(reverse=True)
    return scored[0][2]


def detect_valid_col(
    df: pd.DataFrame,
    explicit: str | None = None,
) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"valid column 없음: {explicit}")
        return explicit

    exact = [
        "wrist_speed_valid",
        "wrist_speed_is_valid",
        "throwing_wrist_speed_valid",
        "wrist_valid",
        "wrist_motion_valid",
    ]

    for col in exact:
        if col in df.columns:
            return col

    scored = []

    for col in df.columns:
        low = col.lower()

        if "wrist" in low and "valid" in low:
            mask = bool_series(df[col])
            if mask.any():
                score = 2 if "speed" in low else 1
                scored.append((score, int(mask.sum()), col))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][2]


def triangular_score(
    x: float,
    low: float,
    target: float,
    high: float,
) -> float:
    if x < low or x > high:
        return 0.0

    if x <= target:
        denom = max(target - low, 1e-8)
        return float((x - low) / denom)

    denom = max(high - target, 1e-8)
    return float((high - x) / denom)


def local_peak_mask(
    values: pd.Series,
    radius: int = 1,
) -> pd.Series:
    """
    단순 local maximum detector.
    양옆 radius 안의 값보다 크거나 같으면 peak.
    """
    v = pd.to_numeric(values, errors="coerce")
    out = pd.Series(False, index=v.index)

    for pos in range(len(v)):
        value = v.iloc[pos]

        if not np.isfinite(value):
            continue

        lo = max(0, pos - radius)
        hi = min(len(v), pos + radius + 1)

        window = v.iloc[lo:hi].dropna()

        if window.empty:
            continue

        if value >= float(window.max()):
            out.iloc[pos] = True

    return out


# ============================================================
# Detector
# ============================================================

def detect_final_release_v2(
    df: pd.DataFrame,
    fc_frame: int,
    preliminary_anchor_frame: int | None,
    speed_col: str | None = None,
    valid_col: str | None = None,
    search_min_sec: float = 0.02,
    search_max_sec: float = 0.30,
    normal_min_sec: float = 0.06,
    normal_max_sec: float = 0.20,
    early_anchor_sec: float = 0.06,
    fallback_min_sec: float = 0.08,
    fallback_target_sec: float = 0.12,
    fallback_max_sec: float = 0.18,
):
    if "frame" not in df.columns:
        raise ValueError("입력 CSV에 frame 컬럼이 없습니다.")

    fps = estimate_fps(df)

    detected_speed_col = detect_speed_col(
        df,
        explicit=speed_col,
    )

    detected_valid_col = detect_valid_col(
        df,
        explicit=valid_col,
    )

    frame = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    speed = pd.to_numeric(
        df[detected_speed_col],
        errors="coerce",
    )

    valid_mask = None

    if detected_valid_col is not None:
        valid_mask = bool_series(
            df[detected_valid_col]
        )

    search_start = int(
        round(
            fc_frame
            +
            search_min_sec
            *
            fps
        )
    )

    search_end = int(
        round(
            fc_frame
            +
            search_max_sec
            *
            fps
        )
    )

    search_mask = (
        frame.notna()
        &
        (frame >= search_start)
        &
        (frame <= search_end)
    )

    if not search_mask.any():
        raise RuntimeError(
            "FC 이후 Release 탐색 구간이 비어 있습니다."
        )

    normal_start = int(
        round(
            fc_frame
            +
            normal_min_sec
            *
            fps
        )
    )

    normal_end = int(
        round(
            fc_frame
            +
            normal_max_sec
            *
            fps
        )
    )

    normal_mask = (
        frame.notna()
        &
        (frame >= normal_start)
        &
        (frame <= normal_end)
    )

    # --------------------------------------------------------
    # preliminary anchor 상태
    # --------------------------------------------------------

    anchor_gap_sec = None
    anchor_is_early = False
    anchor_is_normal = False

    if preliminary_anchor_frame is not None:
        anchor_gap_sec = (
            preliminary_anchor_frame
            -
            fc_frame
        ) / fps

        anchor_is_early = (
            anchor_gap_sec
            <
            early_anchor_sec
        )

        anchor_is_normal = (
            normal_min_sec
            <= anchor_gap_sec
            <= normal_max_sec
        )

    # --------------------------------------------------------
    # validated peak in normal timing window
    # --------------------------------------------------------

    validated_peak_frame = None
    validated_peak_speed = None

    if valid_mask is not None:
        vm = (
            normal_mask
            &
            valid_mask
            &
            speed.notna()
        )

        if vm.any():
            idx = speed.loc[vm].idxmax()

            validated_peak_frame = int(
                frame.loc[idx]
            )

            validated_peak_speed = float(
                speed.loc[idx]
            )

    # --------------------------------------------------------
    # raw peak in normal timing window
    # --------------------------------------------------------

    raw_peak_frame = None
    raw_peak_speed = None

    rm = (
        normal_mask
        &
        speed.notna()
    )

    if rm.any():
        idx = speed.loc[rm].idxmax()

        raw_peak_frame = int(
            frame.loc[idx]
        )

        raw_peak_speed = float(
            speed.loc[idx]
        )

    # --------------------------------------------------------
    # Decision path A: normal preliminary anchor
    # --------------------------------------------------------

    if (
        preliminary_anchor_frame is not None
        and anchor_is_normal
        and not anchor_is_early
    ):
        # validated peak이 있으면 validated trajectory 우선
        if validated_peak_frame is not None:
            release_frame = validated_peak_frame
            release_speed = validated_peak_speed
            source = "validated_peak_normal"
            confidence = "high"

            reason = (
                "preliminary anchor lies in normal post-FC timing and "
                "validated wrist-speed peak is available"
            )

        elif raw_peak_frame is not None:
            release_frame = raw_peak_frame
            release_speed = raw_peak_speed
            source = "raw_peak_normal_fallback"
            confidence = "medium"

            reason = (
                "preliminary anchor timing is plausible but validated peak "
                "is unavailable; raw wrist-speed peak used"
            )

        else:
            release_frame = int(
                preliminary_anchor_frame
            )

            release_speed = np.nan
            source = "preliminary_anchor_only"
            confidence = "low"

            reason = (
                "no usable post-FC speed peak; preliminary anchor retained"
            )

        candidates_df = pd.DataFrame()

    # --------------------------------------------------------
    # Decision path B: suspiciously early anchor / no anchor
    # --------------------------------------------------------

    else:
        fallback_start = int(
            round(
                fc_frame
                +
                fallback_min_sec
                *
                fps
            )
        )

        fallback_end = int(
            round(
                fc_frame
                +
                fallback_max_sec
                *
                fps
            )
        )

        fallback_mask = (
            frame.notna()
            &
            (frame >= fallback_start)
            &
            (frame <= fallback_end)
            &
            speed.notna()
        )

        if not fallback_mask.any():
            raise RuntimeError(
                "Extreme fallback Release 탐색 구간에 유효 wrist speed가 없습니다."
            )

        temp = pd.DataFrame(
            {
                "frame":
                    frame.loc[fallback_mask].astype(int),

                "speed":
                    speed.loc[fallback_mask].astype(float),
            }
        )

        temp["gap_sec_from_fc"] = (
            temp["frame"]
            -
            fc_frame
        ) / fps

        local_mask = local_peak_mask(
            temp["speed"],
            radius=1,
        )

        peaks = temp.loc[
            local_mask
        ].copy()

        if peaks.empty:
            peaks = temp.copy()

        # speed를 fallback window 내부 0~1 정규화
        s_min = float(
            temp["speed"].min()
        )

        s_max = float(
            temp["speed"].max()
        )

        denom = max(
            s_max - s_min,
            1e-8,
        )

        peaks["speed_score"] = (
            peaks["speed"]
            -
            s_min
        ) / denom

        peaks["timing_score"] = peaks[
            "gap_sec_from_fc"
        ].apply(
            lambda x:
            triangular_score(
                float(x),
                fallback_min_sec,
                fallback_target_sec,
                fallback_max_sec,
            )
        )

        # Rogers 같은 케이스는 raw peak가 FC 직후 너무 일찍 생길 수 있으므로,
        # plausible window 안에서는 timing prior를 충분히 반영한다.
        peaks["combined_score"] = (
            1.0 * peaks["speed_score"]
            +
            1.4 * peaks["timing_score"]
        )

        chosen = (
            peaks
            .sort_values(
                [
                    "combined_score",
                    "timing_score",
                    "speed",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        release_frame = int(
            chosen["frame"]
        )

        release_speed = float(
            chosen["speed"]
        )

        source = "raw_local_peak_timing_fallback"
        confidence = "medium"

        if anchor_is_early:
            reason = (
                "preliminary anchor is suspiciously early after FC; "
                "raw local wrist-speed peaks in a plausible post-FC window "
                "were rescored with a timing prior"
            )
        else:
            reason = (
                "preliminary anchor unavailable or outside normal timing; "
                "raw local wrist-speed peaks were rescored with a timing prior"
            )

        candidates_df = peaks.sort_values(
            "combined_score",
            ascending=False,
        )

    release_gap_frames = int(
        release_frame
        -
        fc_frame
    )

    release_gap_sec = (
        release_gap_frames
        /
        fps
    )

    summary = {
        "fps":
            float(fps),

        "speed_column":
            detected_speed_col,

        "valid_mask_column":
            detected_valid_col,

        "fc_frame":
            int(fc_frame),

        "preliminary_anchor_frame":
            (
                int(preliminary_anchor_frame)
                if preliminary_anchor_frame is not None
                else None
            ),

        "preliminary_anchor_gap_sec":
            (
                float(anchor_gap_sec)
                if anchor_gap_sec is not None
                else None
            ),

        "preliminary_anchor_is_early":
            bool(anchor_is_early),

        "search_start_frame":
            int(search_start),

        "search_end_frame":
            int(search_end),

        "normal_window_start_frame":
            int(normal_start),

        "normal_window_end_frame":
            int(normal_end),

        "validated_peak_frame":
            validated_peak_frame,

        "validated_peak_speed":
            validated_peak_speed,

        "raw_peak_frame":
            raw_peak_frame,

        "raw_peak_speed":
            raw_peak_speed,

        "final_release_frame":
            int(release_frame),

        "final_release_speed":
            (
                float(release_speed)
                if np.isfinite(release_speed)
                else None
            ),

        "fc_to_release_frames":
            int(release_gap_frames),

        "fc_to_release_sec":
            float(release_gap_sec),

        "release_source":
            source,

        "release_confidence":
            confidence,

        "reason":
            reason,

        "note":
            (
                "Wrist-motion based automatic release candidate; "
                "not direct ball-separation ground truth."
            ),
    }

    debug = pd.DataFrame(
        {
            "frame":
                frame,

            "wrist_speed":
                speed,

            "wrist_speed_valid":
                (
                    valid_mask
                    if valid_mask is not None
                    else False
                ),

            "in_release_search_window":
                search_mask,

            "in_normal_release_window":
                normal_mask,
        }
    )

    return (
        debug,
        candidates_df,
        summary,
    )


# ============================================================
# Graph
# ============================================================

def save_graph(
    debug: pd.DataFrame,
    summary: dict,
    output_path: Path,
):
    fig = plt.figure(
        figsize=(12, 6)
    )

    ax = fig.add_subplot(111)

    ax.plot(
        debug["frame"],
        debug["wrist_speed"],
        label="Raw wrist speed",
    )

    valid = debug[
        "wrist_speed_valid"
    ].astype(bool)

    if valid.any():
        ax.plot(
            debug["frame"],
            debug["wrist_speed"].where(valid),
            label="Validated wrist speed",
        )

    ax.axvspan(
        summary["normal_window_start_frame"],
        summary["normal_window_end_frame"],
        alpha=0.08,
        label="Normal post-FC release window",
    )

    ax.axvline(
        summary["fc_frame"],
        linestyle="--",
        label=f"FC {summary['fc_frame']}",
    )

    if summary["preliminary_anchor_frame"] is not None:
        ax.axvline(
            summary["preliminary_anchor_frame"],
            linestyle=":",
            label=(
                f"Preliminary anchor "
                f"{summary['preliminary_anchor_frame']}"
            ),
        )

    if summary["validated_peak_frame"] is not None:
        ax.axvline(
            summary["validated_peak_frame"],
            linestyle=":",
            label=(
                f"Validated peak "
                f"{summary['validated_peak_frame']}"
            ),
        )

    if summary["raw_peak_frame"] is not None:
        ax.axvline(
            summary["raw_peak_frame"],
            linestyle=":",
            label=(
                f"Raw peak "
                f"{summary['raw_peak_frame']}"
            ),
        )

    ax.axvline(
        summary["final_release_frame"],
        linestyle="-.",
        label=(
            f"Final Release "
            f"{summary['final_release_frame']}"
        ),
    )

    ax.set_title(
        "Sports2D Final Release Detection v2"
    )

    ax.set_xlabel(
        "Frame"
    )

    ax.set_ylabel(
        summary["speed_column"]
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--fc_frame",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--preliminary_anchor_frame",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--speed_col",
        default=None,
    )

    parser.add_argument(
        "--valid_col",
        default=None,
    )

    parser.add_argument(
        "--search_min_sec",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--search_max_sec",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--normal_min_sec",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--normal_max_sec",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--early_anchor_sec",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--fallback_min_sec",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--fallback_target_sec",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--fallback_max_sec",
        type=float,
        default=0.18,
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 CSV 없음: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    debug, candidates, summary = (
        detect_final_release_v2(
            df=df,
            fc_frame=args.fc_frame,
            preliminary_anchor_frame=args.preliminary_anchor_frame,
            speed_col=args.speed_col,
            valid_col=args.valid_col,
            search_min_sec=args.search_min_sec,
            search_max_sec=args.search_max_sec,
            normal_min_sec=args.normal_min_sec,
            normal_max_sec=args.normal_max_sec,
            early_anchor_sec=args.early_anchor_sec,
            fallback_min_sec=args.fallback_min_sec,
            fallback_target_sec=args.fallback_target_sec,
            fallback_max_sec=args.fallback_max_sec,
        )
    )

    debug_path = (
        output_dir
        /
        "final_release_v2_debug.csv"
    )

    candidates_path = (
        output_dir
        /
        "final_release_v2_candidates.csv"
    )

    summary_path = (
        output_dir
        /
        "final_release_v2_summary.json"
    )

    graph_path = (
        output_dir
        /
        "final_release_v2.png"
    )

    debug.to_csv(
        debug_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidates.to_csv(
        candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        summary_path,
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
        debug,
        summary,
        graph_path,
    )

    print()
    print(
        "===== Sports2D Final Release Detection v2 ====="
    )

    print(
        f"Input CSV               : {input_path}"
    )

    print(
        f"FPS                     : {summary['fps']:.2f}"
    )

    print(
        f"Speed column            : {summary['speed_column']}"
    )

    print(
        f"Valid mask column       : {summary['valid_mask_column']}"
    )

    print()

    print(
        f"FC frame                : {summary['fc_frame']}"
    )

    print(
        f"Preliminary Anchor      : "
        f"{summary['preliminary_anchor_frame']}"
    )

    if summary["preliminary_anchor_gap_sec"] is not None:
        print(
            f"Anchor - FC             : "
            f"{summary['preliminary_anchor_gap_sec']:.4f} sec"
        )

    print(
        f"Anchor early?           : "
        f"{summary['preliminary_anchor_is_early']}"
    )

    print()

    print(
        f"Validated peak          : "
        f"{summary['validated_peak_frame']}"
    )

    print(
        f"Raw peak                : "
        f"{summary['raw_peak_frame']}"
    )

    print()

    print(
        f"Final Release           : "
        f"Frame {summary['final_release_frame']}"
    )

    print(
        f"FC -> Release           : "
        f"{summary['fc_to_release_frames']} frames "
        f"({summary['fc_to_release_sec']:.4f} sec)"
    )

    print(
        f"Release source          : "
        f"{summary['release_source']}"
    )

    print(
        f"Release confidence      : "
        f"{summary['release_confidence']}"
    )

    print(
        f"Reason                  : "
        f"{summary['reason']}"
    )

    print()
    print(
        "※ Final Release는 wrist-motion 기반 자동 후보이며 "
        "공의 실제 분리 순간을 직접 측정한 값은 아닙니다."
    )

    print()
    print(
        f"Candidates              : {candidates_path}"
    )

    print(
        f"Debug                   : {debug_path}"
    )

    print(
        f"Summary                 : {summary_path}"
    )

    print(
        f"Graph                   : {graph_path}"
    )


if __name__ == "__main__":
    main()
