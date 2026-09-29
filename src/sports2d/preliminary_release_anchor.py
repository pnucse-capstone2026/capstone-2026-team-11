#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def estimate_fps(df):
    for col in ["timestamp", "timestamp_sec", "time"]:
        if col in df.columns:
            t = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if len(dt):
                fps = 1.0 / np.nanmedian(dt)
                if np.isfinite(fps) and fps > 0:
                    return float(fps)
    return 30.0


def bool_series(s):
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t", "valid"])
    )


def detect_speed_col(df, explicit=None):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"speed column 없음: {explicit}")
        return explicit

    exact = [
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
    for c in exact:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().any():
                return c

    scored = []
    for c in df.columns:
        low = c.lower()
        if "wrist" not in low:
            continue
        if "speed" not in low and "velocity" not in low:
            continue
        if "valid" in low or "outlier" in low:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if not v.notna().any():
            continue
        score = 0
        if "norm" in low:
            score += 5
        if "body" in low:
            score += 4
        if "relative" in low:
            score += 2
        if "throwing" in low:
            score += 2
        scored.append((score, int(v.notna().sum()), c))

    if not scored:
        raise RuntimeError(
            "wrist speed 컬럼 자동 탐지 실패. --speed_col로 직접 지정하세요."
        )

    scored.sort(reverse=True)
    return scored[0][2]


def detect_valid_col(df, explicit=None):
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
    for c in exact:
        if c in df.columns:
            return c

    scored = []
    for c in df.columns:
        low = c.lower()
        if "wrist" in low and "valid" in low:
            m = bool_series(df[c])
            if m.any():
                score = 2 if "speed" in low else 1
                scored.append((score, int(m.sum()), c))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][2]


def peak(df, speed, mask=None):
    valid = speed.notna() & np.isfinite(speed.to_numpy(float))
    if mask is not None:
        valid = valid & mask
    if not valid.any():
        return None

    idx = speed.loc[valid].idxmax()
    return {
        "frame": int(df.loc[idx, "frame"]),
        "speed": float(speed.loc[idx]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--speed_col", default=None)
    p.add_argument("--valid_col", default=None)
    p.add_argument("--disagreement_sec", type=float, default=0.08)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if "frame" not in df.columns:
        raise ValueError("frame 컬럼이 없습니다.")

    fps = estimate_fps(df)
    speed_col = detect_speed_col(df, args.speed_col)
    valid_col = detect_valid_col(df, args.valid_col)

    speed = pd.to_numeric(df[speed_col], errors="coerce")
    raw_peak = peak(df, speed)

    if raw_peak is None:
        raise RuntimeError("raw wrist-speed peak를 찾지 못했습니다.")

    valid_mask = None
    validated_peak = None

    if valid_col is not None:
        valid_mask = bool_series(df[valid_col])
        validated_peak = peak(df, speed, valid_mask)

    if validated_peak is None:
        anchor = raw_peak
        source = "raw_peak_no_validation_mask"
        confidence = "medium"
        disagreement_frames = None
        disagreement_sec = None
        reason = "validation mask unavailable; raw peak used"
    else:
        disagreement_frames = abs(
            raw_peak["frame"] - validated_peak["frame"]
        )
        disagreement_sec = disagreement_frames / fps

        if disagreement_sec <= args.disagreement_sec:
            anchor = validated_peak
            source = "validated_peak"
            confidence = "high"
            reason = "raw and validated peaks are temporally consistent"
        else:
            anchor = raw_peak
            source = "raw_peak_fallback"
            confidence = "medium"
            reason = (
                "validated peak disagrees strongly with raw peak; "
                "raw peak used as preliminary anchor"
            )

    summary = {
        "fps": fps,
        "speed_column": speed_col,
        "valid_mask_column": valid_col,
        "raw_peak_frame": raw_peak["frame"],
        "raw_peak_speed": raw_peak["speed"],
        "validated_peak_frame": (
            validated_peak["frame"] if validated_peak else None
        ),
        "validated_peak_speed": (
            validated_peak["speed"] if validated_peak else None
        ),
        "peak_disagreement_frames": disagreement_frames,
        "peak_disagreement_sec": disagreement_sec,
        "disagreement_threshold_sec": args.disagreement_sec,
        "preliminary_release_anchor_frame": anchor["frame"],
        "preliminary_release_anchor_speed": anchor["speed"],
        "anchor_source": source,
        "anchor_confidence": confidence,
        "reason": reason,
        "is_final_release": False,
        "note": (
            "Preliminary wrist-motion anchor for FC search; "
            "not final ball release."
        ),
    }

    summary_path = out / "preliminary_release_anchor_v1.json"
    graph_path = out / "preliminary_release_anchor_v1.png"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    frames = pd.to_numeric(df["frame"], errors="coerce")

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(frames, speed, label="Raw wrist speed")

    if valid_mask is not None:
        ax.plot(
            frames,
            speed.where(valid_mask),
            label="Validated wrist speed",
        )

    ax.axvline(
        raw_peak["frame"],
        linestyle=":",
        label=f"Raw peak {raw_peak['frame']}",
    )

    if validated_peak is not None:
        ax.axvline(
            validated_peak["frame"],
            linestyle="--",
            label=f"Validated peak {validated_peak['frame']}",
        )

    ax.axvline(
        anchor["frame"],
        linestyle="-.",
        label=f"Preliminary anchor {anchor['frame']}",
    )

    ax.set_title("Sports2D Preliminary Release Anchor v1")
    ax.set_xlabel("Frame")
    ax.set_ylabel(speed_col)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(graph_path, dpi=160)
    plt.close(fig)

    print()
    print("===== Sports2D Preliminary Release Anchor v1 =====")
    print(f"Input CSV               : {inp}")
    print(f"FPS                     : {fps:.2f}")
    print(f"Speed column            : {speed_col}")
    print(f"Valid mask column       : {valid_col}")
    print()
    print(
        f"Raw peak                : "
        f"Frame {raw_peak['frame']} ({raw_peak['speed']:.3f})"
    )

    if validated_peak is not None:
        print(
            f"Validated peak          : "
            f"Frame {validated_peak['frame']} "
            f"({validated_peak['speed']:.3f})"
        )
        print(
            f"Peak disagreement       : "
            f"{disagreement_frames} frames "
            f"({disagreement_sec:.4f} sec)"
        )

    print()
    print(
        f"Preliminary Anchor      : "
        f"Frame {anchor['frame']}"
    )
    print(f"Anchor source           : {source}")
    print(f"Anchor confidence       : {confidence}")
    print(f"Reason                  : {reason}")
    print()
    print("※ 최종 Ball Release가 아니라 FC 탐색용 anchor입니다.")
    print(f"Summary                 : {summary_path}")
    print(f"Graph                   : {graph_path}")


if __name__ == "__main__":
    main()
