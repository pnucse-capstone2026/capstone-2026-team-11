#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOOT_PARTS = ["ankle", "heel", "big_toe", "small_toe"]


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


def find_xy_cols(df, joint):
    for sx, sy in [("_x_clean", "_y_clean"), ("_x", "_y")]:
        xcol, ycol = joint + sx, joint + sy
        if xcol in df.columns and ycol in df.columns:
            return xcol, ycol
    return None, None


def safe_median(s):
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.median()) if not v.empty else np.nan


def build_foot_center(df, stride_side):
    xs, ys, used = [], [], []
    for part in FOOT_PARTS:
        joint = f"{stride_side}_{part}"
        xcol, ycol = find_xy_cols(df, joint)
        if xcol is None:
            continue
        xs.append(pd.to_numeric(df[xcol], errors="coerce"))
        ys.append(pd.to_numeric(df[ycol], errors="coerce"))
        used.append(joint)
    if not xs:
        raise RuntimeError(f"{stride_side} foot 좌표를 찾지 못했습니다.")
    return (
        pd.concat(xs, axis=1).median(axis=1, skipna=True),
        pd.concat(ys, axis=1).median(axis=1, skipna=True),
        used,
    )


def estimate_body_scale(df):
    for col in ["body_scale_px", "reference_body_scale_px", "body_scale"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                med = float(s.dropna().median())
                if np.isfinite(med) and med > 0:
                    return s.fillna(med)

    joints = {}
    for joint in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
        xcol, ycol = find_xy_cols(df, joint)
        if xcol is not None:
            joints[joint] = (
                pd.to_numeric(df[xcol], errors="coerce"),
                pd.to_numeric(df[ycol], errors="coerce"),
            )

    if len(joints) == 4:
        lsx, lsy = joints["left_shoulder"]
        rsx, rsy = joints["right_shoulder"]
        lhx, lhy = joints["left_hip"]
        rhx, rhy = joints["right_hip"]
        smx, smy = (lsx + rsx) / 2, (lsy + rsy) / 2
        hmx, hmy = (lhx + rhx) / 2, (lhy + rhy) / 2
        shoulder_width = np.sqrt((lsx-rsx)**2 + (lsy-rsy)**2)
        torso_len = np.sqrt((smx-hmx)**2 + (smy-hmy)**2)
        scale = shoulder_width + torso_len
        med = float(pd.to_numeric(scale, errors="coerce").dropna().median())
        if np.isfinite(med) and med > 0:
            return pd.Series(scale, index=df.index).fillna(med)

    return pd.Series(np.ones(len(df)), index=df.index)


def timing_score(offset_sec, min_offset, target_offset, max_offset):
    if offset_sec < min_offset or offset_sec > max_offset:
        return 0.0
    if offset_sec <= target_offset:
        return float((offset_sec - min_offset) / max(target_offset - min_offset, 1e-8))
    return float((max_offset - offset_sec) / max(max_offset - target_offset, 1e-8))


def detect_fc(df, throwing_side, anchor_frame,
              smooth_window=3, pre_window_sec=0.10, post_window_sec=0.10,
              min_offset_sec=-0.22, target_offset_sec=-0.10, max_offset_sec=0.04):
    fps = estimate_fps(df)
    if not (min_offset_sec < target_offset_sec < max_offset_sec):
        raise ValueError("min_offset_sec < target_offset_sec < max_offset_sec 이어야 합니다.")

    stride_side = "left" if throwing_side == "right" else "right"
    frames = pd.to_numeric(df["frame"], errors="coerce")
    scale = estimate_body_scale(df).astype(float)
    foot_x_raw, foot_y_raw, used = build_foot_center(df, stride_side)
    foot_x = foot_x_raw.rolling(smooth_window, center=True, min_periods=1).median()
    foot_y = foot_y_raw.rolling(smooth_window, center=True, min_periods=1).median()
    dx, dy = foot_x.diff(), foot_y.diff()
    speed = np.sqrt(dx**2 + dy**2) / scale
    downward = dy / scale

    vf = frames.dropna()
    if vf.empty:
        raise RuntimeError("유효 frame 번호가 없습니다.")

    search_start = max(int(vf.min()), int(round(anchor_frame + min_offset_sec * fps)))
    search_end = min(int(vf.max()), int(round(anchor_frame + max_offset_sec * fps)))
    if search_start >= search_end:
        raise RuntimeError(f"FC 탐색 구간 오류: {search_start} ~ {search_end}")

    mask = frames.notna() & (frames >= search_start) & (frames <= search_end)
    search_speed = speed.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
    if len(search_speed) < 4:
        raise RuntimeError("탐색 구간의 유효 foot speed가 너무 적습니다.")

    p20 = float(search_speed.quantile(.20))
    p35 = float(search_speed.quantile(.35))
    p60 = float(search_speed.quantile(.60))
    stable_thr = max(p20, min(p35, p60 * .65))
    moving_thr = max(p60, stable_thr * 1.35)

    pre_n = max(2, int(round(pre_window_sec * fps)))
    post_n = max(2, int(round(post_window_sec * fps)))
    rows = []

    for idx in np.flatnonzero(mask.to_numpy()):
        if idx - pre_n < 0 or idx + post_n >= len(df):
            continue
        frame = int(frames.iloc[idx])
        offset_frames = frame - anchor_frame
        offset_sec = offset_frames / fps
        if offset_sec < min_offset_sec or offset_sec > max_offset_sec:
            continue

        prev_med = safe_median(speed.iloc[idx-pre_n:idx])
        post_med = safe_median(speed.iloc[idx:idx+post_n])
        if not (np.isfinite(prev_med) and np.isfinite(post_med)):
            continue
        ratio = prev_med / max(post_med, 1e-8)
        drop = prev_med - post_med
        if drop <= 0 or ratio < 1.10:
            continue

        t = timing_score(offset_sec, min_offset_sec, target_offset_sec, max_offset_sec)
        moving_score = float(np.clip(prev_med / max(moving_thr, 1e-8), 0, 2))
        stable_score = float(np.clip(stable_thr / max(post_med, 1e-8), 0, 2))
        ratio_score = float(np.clip(ratio / 1.5, 0, 2))
        drop_score = float(np.clip(drop / max(stable_thr, 1e-8), 0, 2))
        down_med = safe_median(downward.iloc[idx-pre_n:idx])
        down_bonus = .35 if np.isfinite(down_med) and down_med > 0 else 0.0

        score = 1.4*moving_score + 1.6*stable_score + 1.2*ratio_score + 1.0*drop_score + 1.3*t + down_bonus
        rows.append({
            "frame": frame,
            "index": int(idx),
            "offset_frames_from_anchor": int(offset_frames),
            "offset_sec_from_anchor": float(offset_sec),
            "timing_score": float(t),
            "score": float(score),
            "prev_speed_median": float(prev_med),
            "post_speed_median": float(post_med),
            "speed_ratio": float(ratio),
            "speed_drop": float(drop),
        })

    if not rows:
        raise RuntimeError("Preliminary anchor 주변에서 foot-lock 후보를 찾지 못했습니다.")

    cand = pd.DataFrame(rows)
    qualified = cand[(cand["speed_ratio"] >= 1.15) & (cand["post_speed_median"] <= moving_thr)].copy()
    if qualified.empty:
        qualified = cand.copy()
    qualified["target_offset_error"] = (qualified["offset_sec_from_anchor"] - target_offset_sec).abs()
    chosen = qualified.sort_values(["score", "target_offset_error", "frame"], ascending=[False, True, True]).iloc[0]
    fc_frame = int(chosen["frame"])

    debug = pd.DataFrame({
        "frame": frames,
        "foot_x_raw": foot_x_raw,
        "foot_y_raw": foot_y_raw,
        "foot_x_smooth": foot_x,
        "foot_y_smooth": foot_y,
        "body_scale": scale,
        "foot_speed_body_per_frame": speed,
        "foot_downward_body_per_frame": downward,
        "in_anchor_search_window": mask,
    })

    summary = {
        "fps": float(fps),
        "throwing_side": throwing_side,
        "stride_side": stride_side,
        "used_foot_joints": used,
        "preliminary_anchor_frame": int(anchor_frame),
        "min_offset_sec": float(min_offset_sec),
        "target_offset_sec": float(target_offset_sec),
        "max_offset_sec": float(max_offset_sec),
        "search_start_frame": int(search_start),
        "search_end_frame": int(search_end),
        "adaptive_stable_threshold": float(stable_thr),
        "adaptive_moving_threshold": float(moving_thr),
        "front_foot_contact_frame": fc_frame,
        "fc_offset_frames_from_anchor": int(fc_frame - anchor_frame),
        "fc_offset_sec_from_anchor": float((fc_frame - anchor_frame) / fps),
        "chosen_score": float(chosen["score"]),
        "chosen_timing_score": float(chosen["timing_score"]),
        "chosen_speed_ratio": float(chosen["speed_ratio"]),
        "chosen_speed_drop": float(chosen["speed_drop"]),
        "method": "preliminary anchor window + foot-lock quality + weak timing prior; lock frame = FC",
    }
    return debug, cand.sort_values("score", ascending=False), summary


def save_graph(debug, summary, path):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = pd.to_numeric(debug["frame"], errors="coerce")
    y = pd.to_numeric(debug["foot_speed_body_per_frame"], errors="coerce")
    ax.plot(x, y, label="Stride-foot speed (body/frame)")
    ax.axhline(summary["adaptive_stable_threshold"], linestyle="--", label="Stable threshold")
    ax.axhline(summary["adaptive_moving_threshold"], linestyle=":", label="Moving threshold")
    ax.axvspan(summary["search_start_frame"], summary["search_end_frame"], alpha=.08, label="Search window")
    target_frame = summary["preliminary_anchor_frame"] + int(round(summary["target_offset_sec"] * summary["fps"]))
    ax.axvline(target_frame, linestyle=":", label=f"Target offset {target_frame}")
    ax.axvline(summary["front_foot_contact_frame"], linestyle="--", label=f"Estimated FC {summary['front_foot_contact_frame']}")
    ax.axvline(summary["preliminary_anchor_frame"], linestyle="-.", label=f"Preliminary anchor {summary['preliminary_anchor_frame']}")
    ax.set_title("Sports2D FC v1.5 - Preliminary Anchor + Foot Lock")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized foot speed (body/frame)")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--throwing_side", required=True, choices=["left", "right"])
    p.add_argument("--preliminary_anchor_frame", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--smooth_window", type=int, default=3)
    p.add_argument("--pre_window_sec", type=float, default=.10)
    p.add_argument("--post_window_sec", type=float, default=.10)
    p.add_argument("--min_offset_sec", type=float, default=-.22)
    p.add_argument("--target_offset_sec", type=float, default=-.10)
    p.add_argument("--max_offset_sec", type=float, default=.04)
    a = p.parse_args()

    inp = Path(a.input)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(inp)
    if "frame" not in df.columns:
        raise ValueError("frame 컬럼이 없습니다.")

    debug, candidates, summary = detect_fc(
        df, a.throwing_side, a.preliminary_anchor_frame,
        a.smooth_window, a.pre_window_sec, a.post_window_sec,
        a.min_offset_sec, a.target_offset_sec, a.max_offset_sec,
    )

    debug_path = out / "fc_v1_5_debug.csv"
    cand_path = out / "fc_v1_5_candidates.csv"
    summary_path = out / "fc_v1_5_summary.json"
    graph_path = out / "fc_v1_5_anchor_based.png"
    debug.to_csv(debug_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(cand_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    save_graph(debug, summary, graph_path)

    print("\n===== Sports2D Front Foot Contact v1.5 =====")
    print(f"Input CSV               : {inp}")
    print(f"FPS                     : {summary['fps']:.2f}")
    print(f"Throwing side           : {summary['throwing_side']}")
    print(f"Stride side             : {summary['stride_side']}")
    print("Foot joints             : " + ", ".join(summary["used_foot_joints"]))
    print()
    print(f"Preliminary Anchor      : {summary['preliminary_anchor_frame']}")
    print(f"Search window           : {summary['search_start_frame']} ~ {summary['search_end_frame']}")
    print(f"Offset range            : {summary['min_offset_sec']:.3f} ~ {summary['max_offset_sec']:.3f} sec")
    print(f"Target offset           : {summary['target_offset_sec']:.3f} sec")
    print(f"Adaptive stable thr     : {summary['adaptive_stable_threshold']:.6f} body/frame")
    print(f"Adaptive moving thr     : {summary['adaptive_moving_threshold']:.6f} body/frame")
    print()
    print(f"Estimated FC            : Frame {summary['front_foot_contact_frame']}")
    print(f"FC - Anchor             : {summary['fc_offset_frames_from_anchor']} frames ({summary['fc_offset_sec_from_anchor']:.4f} sec)")
    print(f"Timing Score            : {summary['chosen_timing_score']:.3f}")
    print(f"Speed Ratio             : {summary['chosen_speed_ratio']:.3f}")
    print(f"Speed Drop              : {summary['chosen_speed_drop']:.6f}")
    print()
    print(f"Candidates              : {cand_path}")
    print(f"Debug                   : {debug_path}")
    print(f"Summary                 : {summary_path}")
    print(f"Graph                   : {graph_path}")
    print("\n※ preliminary anchor 오차를 허용하며 lock frame 자체를 FC로 사용합니다.")


if __name__ == "__main__":
    main()
