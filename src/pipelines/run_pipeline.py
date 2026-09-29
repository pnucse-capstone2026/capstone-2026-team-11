#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canonical Pitch Analysis Pipeline

Steps
-----
1) Sports2D Stage 2
2) Sports2D Stage 3
3) Build analysis_events.json
4) Motion Normalization
5) Feature Extraction
6) Analysis Overlay

The pipeline uses pitch_id as the single naming key, so exceptions such as
kershaw_02 require no special-case code.

Existing outputs are reused by default. If a selected upstream step is actually
regenerated, all selected downstream steps are automatically regenerated too,
so stale dependent outputs cannot be reused. Use --force to rerun every
selected step regardless of existing outputs.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PITCHES_CSV = PROJECT_ROOT / "configs" / "pitches.csv"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_videos"

SRC_CORE = PROJECT_ROOT / "src" / "core"
SRC_PIPELINES = PROJECT_ROOT / "src" / "pipelines"

SPORTS2D_DIR = PROJECT_ROOT / "sports2d"
SPORTS2D_INPUT_DIR = SPORTS2D_DIR / "input"
SPORTS2D_STAGE2_ROOT = SPORTS2D_DIR / "stage2"
SPORTS2D_STAGE3_ROOT = SPORTS2D_DIR / "stage3"

PIPELINE_OUTPUT_ROOT = PROJECT_ROOT / "pipeline_outputs"


STEP_NAMES = {
    1: "Sports2D Stage 2",
    2: "Sports2D Stage 3",
    3: "Build Analysis Events",
    4: "Motion Normalization",
    5: "Feature Extraction",
    6: "Analysis Overlay",
}


def normalize_text(value, default=""):
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def normalize_optional_int(value):
    text = normalize_text(value, "")

    if text == "":
        return None

    try:
        return int(round(float(text)))
    except (TypeError, ValueError):
        return None


def is_enabled(value):
    return normalize_text(value, "1").lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def load_pitches():
    if not PITCHES_CSV.exists():
        raise FileNotFoundError(
            f"pitches.csv not found: {PITCHES_CSV}"
        )

    rows = []

    with open(
        PITCHES_CSV,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not is_enabled(row.get("enabled", "1")):
                continue

            pitch_id = normalize_text(
                row.get("pitch_id")
            )

            if not pitch_id:
                continue

            row["pitch_id"] = pitch_id
            row["pitcher_name"] = normalize_text(
                row.get("pitcher_name")
            )
            row["throwing_side"] = normalize_text(
                row.get("throwing_side")
            ).lower()
            row["video_file"] = normalize_text(
                row.get("video_file"),
                f"{pitch_id}.mp4",
            )

            for key in (
                "manual_knee_lift_frame",
                "manual_fc_frame",
                "manual_release_frame",
            ):
                row[key] = normalize_optional_int(
                    row.get(key)
                )

            rows.append(row)

    return rows


def get_paths(pitch):
    pitch_id = pitch["pitch_id"]

    stage2_dir = SPORTS2D_STAGE2_ROOT / pitch_id
    stage2_final = (
        stage2_dir
        / "pose_csv"
        / f"{pitch_id}_sports2d_wrist_motion_validated.csv"
    )

    stage3_dir = SPORTS2D_STAGE3_ROOT / pitch_id
    stage3_summary = (
        stage3_dir
        / "stage3_events_final_summary.json"
    )

    output_dir = PIPELINE_OUTPUT_ROOT / pitch_id
    pose_output_dir = output_dir / "pose_csv"
    graph_output_dir = (
        output_dir
        / "graphs"
        / "motion_normalization"
    )
    video_output_dir = output_dir / "videos"

    events_json = output_dir / "analysis_events.json"

    motion_normalized = (
        pose_output_dir
        / f"{pitch_id}_motion_normalized.csv"
    )

    feature_csv = (
        output_dir
        / f"{pitch_id}_features.csv"
    )

    overlay_video = (
        video_output_dir
        / f"{pitch_id}_analysis_overlay.mp4"
    )

    video_path = (
        RAW_VIDEO_DIR
        / pitch["video_file"]
    )

    roi_json = (
        SPORTS2D_INPUT_DIR
        / f"{pitch_id}_roi.json"
    )

    return {
        "stage2_dir": stage2_dir,
        "stage2_final": stage2_final,
        "stage3_dir": stage3_dir,
        "stage3_summary": stage3_summary,
        "output_dir": output_dir,
        "events_json": events_json,
        "motion_normalized": motion_normalized,
        "graph_output_dir": graph_output_dir,
        "feature_csv": feature_csv,
        "overlay_video": overlay_video,
        "video_path": video_path,
        "roi_json": roi_json,
    }


def find_stage2_input(pitch_id):
    """
    Look only inside sports2d/input to avoid accidentally reusing
    a derived CSV from another stage.
    """
    candidates = [
        SPORTS2D_INPUT_DIR
        / f"{pitch_id}_sports2d_pose.csv",

        SPORTS2D_INPUT_DIR
        / f"{pitch_id}_pose.csv",

        SPORTS2D_INPUT_DIR
        / f"{pitch_id}.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = []

    for pattern in (
        f"{pitch_id}*sports2d*pose*.csv",
        f"{pitch_id}*pose*.csv",
    ):
        matches.extend(
            SPORTS2D_INPUT_DIR.glob(pattern)
        )

    matches = sorted(
        {
            p
            for p in matches
            if p.is_file()
        },
        key=lambda p: (
            len(p.name),
            p.name.lower(),
        ),
    )

    return matches[0] if matches else None


def pretty_command(cmd):
    return " ".join(
        f'"{str(x)}"'
        if " " in str(x)
        else str(x)
        for x in cmd
    )


def run_command(cmd, title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(pretty_command(cmd))
    print()

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{title} failed "
            f"(return code={proc.returncode})"
        )


def require_file(path, label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    return path


def can_reuse(path, force):
    return (
        Path(path).exists()
        and not force
    )


def run_stage2(
    pitch,
    paths,
    force=False,
):
    output_path = paths["stage2_final"]

    if can_reuse(output_path, force):
        print(
            f"[REUSE] Stage 2: {output_path}"
        )
        return False

    stage2_input = find_stage2_input(
        pitch["pitch_id"]
    )

    if stage2_input is None:
        raise FileNotFoundError(
            "Sports2D Stage 2 input CSV could not be found.\n"
            f"Expected a converted Sports2D pose CSV under: "
            f"{SPORTS2D_INPUT_DIR}\n"
            f"Pitch ID: {pitch['pitch_id']}\n"
            "If Stage 2 already exists, run from step 2."
        )

    script = require_file(
        SRC_PIPELINES
        / "run_sports2d_stage2.py",
        "Stage 2 runner",
    )

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(stage2_input),
        "--pitch_id",
        pitch["pitch_id"],
        "--throwing_side",
        pitch["throwing_side"],
        "--core_dir",
        str(SRC_CORE),
        "--output_root",
        str(SPORTS2D_STAGE2_ROOT),
    ]

    run_command(
        cmd,
        "STEP 1/6 - Sports2D Stage 2",
    )

    require_file(
        output_path,
        "Stage 2 final wrist-motion CSV",
    )

    return True


def run_stage3(
    pitch,
    paths,
    force=False,
):
    output_path = paths["stage3_summary"]

    if can_reuse(output_path, force):
        print(
            f"[REUSE] Stage 3: {output_path}"
        )
        return False

    stage2_final = require_file(
        paths["stage2_final"],
        "Stage 2 final wrist-motion CSV",
    )

    script = require_file(
        SRC_PIPELINES
        / "run_sports2d_stage3.py",
        "Stage 3 runner",
    )

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(stage2_final),
        "--throwing_side",
        pitch["throwing_side"],
        "--label",
        pitch["pitch_id"],
        "--output_dir",
        str(paths["stage3_dir"]),
    ]

    run_command(
        cmd,
        "STEP 2/6 - Sports2D Stage 3",
    )

    require_file(
        output_path,
        "Stage 3 summary",
    )

    return True


def run_build_events(
    pitch,
    paths,
    force=False,
    use_manual_overrides=False,
):
    output_path = paths["events_json"]

    if can_reuse(output_path, force):
        print(
            f"[REUSE] Analysis Events: {output_path}"
        )
        return False

    require_file(
        paths["stage3_summary"],
        "Stage 3 summary",
    )

    script = require_file(
        SRC_PIPELINES
        / "build_analysis_events.py",
        "analysis_events builder",
    )

    cmd = [
        sys.executable,
        str(script),
        "--pitch",
        pitch["pitch_id"],
        "--stage3_root",
        str(SPORTS2D_STAGE3_ROOT),
        "--output_root",
        str(PIPELINE_OUTPUT_ROOT),
    ]

    if use_manual_overrides:
        cmd.append(
            "--use-manual-overrides"
        )

    run_command(
        cmd,
        "STEP 3/6 - Build Analysis Events",
    )

    require_file(
        output_path,
        "analysis_events.json",
    )

    return True


def run_normalization(
    pitch,
    paths,
    force=False,
):
    output_path = paths[
        "motion_normalized"
    ]

    if can_reuse(output_path, force):
        print(
            f"[REUSE] Normalization: {output_path}"
        )
        return False

    stage2_final = require_file(
        paths["stage2_final"],
        "Stage 2 final wrist-motion CSV",
    )

    events_json = require_file(
        paths["events_json"],
        "analysis_events.json",
    )

    script = require_file(
        SRC_CORE
        / "motion_normalization.py",
        "Motion normalization script",
    )

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(stage2_final),
        "--events",
        str(events_json),
        "--output",
        str(output_path),
        "--graph_dir",
        str(paths["graph_output_dir"]),
    ]

    run_command(
        cmd,
        "STEP 4/6 - Motion Normalization",
    )

    require_file(
        output_path,
        "Motion-normalized CSV",
    )

    return True


def run_feature_extraction(
    pitch,
    paths,
    force=False,
):
    output_path = paths["feature_csv"]

    if can_reuse(output_path, force):
        print(
            f"[REUSE] Features: {output_path}"
        )
        return False

    motion_csv = require_file(
        paths["motion_normalized"],
        "Motion-normalized CSV",
    )

    events_json = require_file(
        paths["events_json"],
        "analysis_events.json",
    )

    script = require_file(
        SRC_CORE
        / "feature_extraction.py",
        "Feature extraction script",
    )

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(motion_csv),
        "--events",
        str(events_json),
        "--output",
        str(output_path),
    ]

    run_command(
        cmd,
        "STEP 5/6 - Feature Extraction",
    )

    require_file(
        output_path,
        "Feature CSV",
    )

    return True


def run_overlay(
    pitch,
    paths,
    force=False,
):
    output_path = paths["overlay_video"]

    if can_reuse(output_path, force):
        print(
            f"[REUSE] Overlay: {output_path}"
        )
        return False

    video_path = require_file(
        paths["video_path"],
        "Original video",
    )

    motion_csv = require_file(
        paths["motion_normalized"],
        "Motion-normalized CSV",
    )

    events_json = require_file(
        paths["events_json"],
        "analysis_events.json",
    )

    script = require_file(
        SRC_CORE
        / "analysis_overlay.py",
        "Analysis overlay script",
    )

    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(video_path),
        "--input",
        str(motion_csv),
        "--events",
        str(events_json),
        "--output",
        str(output_path),
    ]

    if paths["roi_json"].is_file():
        cmd.extend(["--roi-json", str(paths["roi_json"])])

    run_command(
        cmd,
        "STEP 6/6 - Analysis Overlay",
    )

    require_file(
        output_path,
        "Overlay video",
    )

    return True


def print_pitch_header(
    pitch,
    paths,
):
    print()
    print()
    print("#" * 76)
    print("CANONICAL PITCH ANALYSIS")
    print("#" * 76)
    print(
        f"Pitch ID      : "
        f"{pitch['pitch_id']}"
    )
    print(
        f"Pitcher       : "
        f"{pitch['pitcher_name']}"
    )
    print(
        f"Throwing Side : "
        f"{pitch['throwing_side']}"
    )
    print(
        f"Video         : "
        f"{paths['video_path']}"
    )
    print(
        f"ROI JSON      : "
        f"{paths['roi_json']}"
    )
    print(
        f"Output Root   : "
        f"{paths['output_dir']}"
    )
    print("#" * 76)


def run_pitch(
    pitch,
    from_step=1,
    to_step=6,
    force=False,
    use_manual_overrides=False,
):
    paths = get_paths(pitch)

    print_pitch_header(
        pitch,
        paths,
    )

    if pitch["throwing_side"] not in {
        "left",
        "right",
    }:
        raise ValueError(
            f"Invalid throwing_side for "
            f"{pitch['pitch_id']}: "
            f"{pitch['throwing_side']}"
        )

    paths["output_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = [
        (1, run_stage2),
        (2, run_stage3),
        (3, run_build_events),
        (4, run_normalization),
        (5, run_feature_extraction),
        (6, run_overlay),
    ]

    upstream_changed = False

    for step_number, func in steps:
        if not (
            from_step
            <= step_number
            <= to_step
        ):
            continue

        effective_force = (
            force
            or upstream_changed
        )

        print()
        print(
            f"[PIPELINE] "
            f"{step_number}/6 "
            f"{STEP_NAMES[step_number]}"
        )

        if upstream_changed and not force:
            print(
                "[DEPENDENCY] "
                "Upstream output changed -> "
                "rerunning this step."
            )

        if step_number == 3:
            step_executed = func(
                pitch,
                paths,
                force=effective_force,
                use_manual_overrides=
                    use_manual_overrides,
            )
        else:
            step_executed = func(
                pitch,
                paths,
                force=effective_force,
            )

        if step_executed:
            upstream_changed = True

    print()
    print(
        f"[COMPLETE] {pitch['pitch_id']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Sports2D pitch-analysis pipeline"
        )
    )

    parser.add_argument(
        "--pitch",
        default=None,
        help=(
            "Run one pitch_id, e.g. wheeler_01 or kershaw_02. "
            "If omitted, all enabled pitches are processed."
        ),
    )

    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=range(1, 7),
    )

    parser.add_argument(
        "--to-step",
        type=int,
        default=6,
        choices=range(1, 7),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun selected steps even when expected outputs exist."
        ),
    )

    parser.add_argument(
        "--use-manual-overrides",
        action="store_true",
        help=(
            "Apply manual FC/release values from pitches.csv as final "
            "canonical events. Without this flag they remain references."
        ),
    )

    args = parser.parse_args()

    if args.from_step > args.to_step:
        raise ValueError(
            "--from-step must be <= --to-step"
        )

    pitches = load_pitches()

    if args.pitch:
        pitches = [
            pitch
            for pitch in pitches
            if pitch["pitch_id"]
            == args.pitch
        ]

        if not pitches:
            raise ValueError(
                f"pitch_id not found in pitches.csv: "
                f"{args.pitch}"
            )

    print()
    print("=" * 76)
    print(
        "Canonical Baseball Pitch Analysis Pipeline"
    )
    print("=" * 76)
    print(
        f"Target Pitches : {len(pitches)}"
    )
    print(
        f"Steps          : "
        f"{args.from_step} -> {args.to_step}"
    )
    print(
        f"Force          : {args.force}"
    )
    print(
        f"Manual Override: "
        f"{args.use_manual_overrides}"
    )
    print("=" * 76)

    success = []
    failed = []

    for pitch in pitches:
        pitch_id = pitch["pitch_id"]

        try:
            run_pitch(
                pitch,
                from_step=args.from_step,
                to_step=args.to_step,
                force=args.force,
                use_manual_overrides=
                    args.use_manual_overrides,
            )

            success.append(pitch_id)

        except Exception as exc:
            failed.append(
                (
                    pitch_id,
                    str(exc),
                )
            )

            print()
            print(
                f"[FAILED] {pitch_id}"
            )
            print(
                f"Reason: {exc}"
            )

    print()
    print("=" * 76)
    print("BATCH SUMMARY")
    print("=" * 76)
    print(
        f"Success : {len(success)}"
    )
    print(
        f"Failed  : {len(failed)}"
    )

    if success:
        print(
            "Success pitches: "
            + ", ".join(success)
        )

    if failed:
        print(
            "Failed pitches:"
        )

        for pitch_id, reason in failed:
            print(
                f" - {pitch_id}: "
                f"{reason}"
            )

    print("=" * 76)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
