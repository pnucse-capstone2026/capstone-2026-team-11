#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Build canonical analysis_events.json files from:
- configs/pitches.csv
- Sports2D Stage 3 integrated summary

Default behavior:
- Knee Lift: manual reference if present, otherwise lead-knee height before FC
- FC: Sports2D Stage 3 result
- Release: Sports2D Stage 3 result
- Arm Slot: Sports2D Stage 3 result
- manual FC / release values are stored as validation references, not overrides.

Examples
--------
Single pitch:
python src\pipelines\build_analysis_events.py ^
--pitch wheeler_01

All enabled pitches with available Stage 3 summaries:
python src\pipelines\build_analysis_events.py

Optional:
--use-manual-overrides
    manual_fc_frame / manual_release_frame이 있으면 최종 event에 반영한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PITCHES_CSV = (
    PROJECT_ROOT / "configs" / "pitches.csv"
)

DEFAULT_STAGE3_ROOT = (
    PROJECT_ROOT / "sports2d_stage3"
)

DEFAULT_STAGE2_ROOT = PROJECT_ROOT / "sports2d" / "stage2"

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "pipeline_outputs"
)


def normalize_optional_int(value: Any):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return int(round(float(text)))
    except (TypeError, ValueError):
        return None


def normalize_optional_float(value: Any):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any, default=None):
    if value is None:
        return default

    text = str(value).strip()

    if text == "":
        return default

    return text


def normalize_confidence(value: Any):
    text = normalize_text(value, "unknown")
    text = str(text).lower()

    allowed = {
        "high",
        "medium",
        "low",
        "manual",
        "unknown",
    }

    return text if text in allowed else "unknown"


def standardize_release_source(value: Any):
    """
    Stage 3의 내부 알고리즘 source 이름을 canonical source로 바꾼다.
    세부 알고리즘 버전명은 analysis_events.json에 노출하지 않는다.
    """
    text = normalize_text(value, "unknown")

    if text == "unknown":
        return "unknown"

    text = str(text).lower().strip()

    # 현재 Stage 3에서 나오는 대표 source들.
    mapping = {
        "validated_peak": "sports2d_validated_peak",
        "raw_peak_fallback": "sports2d_raw_peak_fallback",
        "raw_peak_normal_fallback": "sports2d_raw_peak_fallback",
        "raw_timing_fallback": "sports2d_raw_timing_fallback",
    }

    if text in mapping:
        return mapping[text]

    if text.startswith("sports2d_"):
        return text

    # 미래 source도 정보는 보존하되 Sports2D 결과임을 명시.
    return "sports2d_" + text


def load_pitches(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"pitches.csv를 찾을 수 없습니다: {path}"
        )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            enabled = str(
                row.get("enabled", "1")
            ).strip().lower()

            if enabled not in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }:
                continue

            pitch_id = normalize_text(
                row.get("pitch_id")
            )

            if pitch_id is None:
                continue

            rows.append(row)

    return rows


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"JSON 파일을 찾을 수 없습니다: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def detect_knee_lift(stage2_csv: Path, throwing_side: str,
                     fc_frame: int | None, fps: float | None) -> int | None:
    """Find the highest lead knee relative to its hip before foot contact.

    Sports2D pixel Y increases downwards. We normalize by torso length to
    reduce apparent scale changes, then use a five-frame median to avoid a
    single misplaced knee becoming the detected event. If the required pose
    joints are missing, callers retain an explicit missing-event error.
    """
    side = str(throwing_side).strip().lower()
    if not stage2_csv.is_file() or fc_frame is None or side not in ("left", "right"):
        return None
    lead = "left" if side == "right" else "right"
    fps = fps if fps and fps > 0 else 60.0
    start = max(0, fc_frame - round(2.0 * fps))
    end = fc_frame - max(3, round(0.08 * fps))
    if end <= start:
        return None

    def coordinate(row: dict, part: str) -> tuple[float, float] | None:
        if str(row.get(f"{part}_invalid", "")).lower() in ("true", "1"):
            return None
        try:
            x = float(row[f"{part}_x_clean"])
            y = float(row[f"{part}_y_clean"])
            if not all(map(math.isfinite, (x, y))):
                return None
            return x, y
        except (KeyError, TypeError, ValueError):
            return None

    samples = []
    with stage2_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = normalize_optional_int(row.get("frame"))
            if frame is None or not start <= frame <= end:
                continue
            hip, knee = coordinate(row, f"{lead}_hip"), coordinate(row, f"{lead}_knee")
            shoulders = [coordinate(row, f"{s}_shoulder") for s in ("left", "right")]
            hips = [coordinate(row, f"{s}_hip") for s in ("left", "right")]
            if hip is None or knee is None or any(p is None for p in shoulders + hips):
                continue
            shoulder_mid = tuple((shoulders[0][j] + shoulders[1][j]) / 2 for j in (0, 1))
            hip_mid = tuple((hips[0][j] + hips[1][j]) / 2 for j in (0, 1))
            torso = math.dist(shoulder_mid, hip_mid)
            if torso > 1e-6:
                samples.append((frame, (hip[1] - knee[1]) / torso))

    if len(samples) < 5:
        return None
    # Smooth only consecutive valid frames so gaps cannot bridge unrelated poses.
    candidates = []
    for index in range(2, len(samples) - 2):
        window = samples[index-2:index+3]
        if window[-1][0] - window[0][0] == 4:
            candidates.append((sorted(value for _, value in window)[2], samples[index][0]))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def build_analysis_events(
    pitch: dict,
    stage3: dict,
    use_manual_overrides: bool = False,
    auto_knee_frame: int | None = None,
):
    pitch_id = normalize_text(
        pitch.get("pitch_id")
    )

    pitcher_name = normalize_text(
        pitch.get("pitcher_name"),
        ""
    )

    throwing_side = normalize_text(
        pitch.get("throwing_side"),
        stage3.get("throwing_side"),
    )

    manual_knee = normalize_optional_int(
        pitch.get("manual_knee_lift_frame")
    )

    manual_fc = normalize_optional_int(
        pitch.get("manual_fc_frame")
    )

    manual_release = normalize_optional_int(
        pitch.get("manual_release_frame")
    )

    # ---------------------------------------------------------
    # Automatic Sports2D results
    # ---------------------------------------------------------

    auto_fc = normalize_optional_int(
        stage3.get("front_foot_contact_frame")
    )

    auto_release = normalize_optional_int(
        stage3.get("final_release_frame")
    )

    release_confidence = normalize_confidence(
        stage3.get("final_release_confidence")
    )

    release_source = standardize_release_source(
        stage3.get("final_release_source")
    )

    # ---------------------------------------------------------
    # Canonical final events
    # ---------------------------------------------------------

    # Reference pitchers keep the reviewed frame; new uploads receive a
    # pose-derived candidate before FC. Never invent a frame on missing data.
    knee_frame = manual_knee if manual_knee is not None else auto_knee_frame
    knee_event = {
        "frame": knee_frame,
        "source": (
            "manual_reference" if manual_knee is not None else
            "sports2d_lead_knee_height" if auto_knee_frame is not None else "unknown"
        ),
        "confidence": (
            "manual" if manual_knee is not None else
            "medium" if auto_knee_frame is not None else "unknown"
        ),
    }

    fc_frame = auto_fc
    fc_source = (
        "sports2d"
        if auto_fc is not None
        else "unknown"
    )
    fc_confidence = "unknown"

    release_frame = auto_release
    final_release_source = release_source
    final_release_confidence = release_confidence

    if use_manual_overrides:
        if manual_fc is not None:
            fc_frame = manual_fc
            fc_source = "manual_override"
            fc_confidence = "manual"

        if manual_release is not None:
            release_frame = manual_release
            final_release_source = "manual_override"
            final_release_confidence = "manual"

    arm_slot_category = normalize_text(
        stage3.get("arm_slot")
    )

    arm_slot_angle = normalize_optional_float(
        stage3.get("arm_slot_angle_deg")
    )

    arm_slot_confidence = normalize_confidence(
        stage3.get("arm_slot_confidence")
    )

    fps = normalize_optional_float(
        stage3.get("fps")
    )

    payload = {
        "pitch_id": pitch_id,
        "pitcher_name": pitcher_name,
        "throwing_side": throwing_side,

        "events": {
            "knee_lift": knee_event,

            "front_foot_contact": {
                "frame": fc_frame,
                "source": fc_source,
                "confidence": fc_confidence,
            },

            "release": {
                "frame": release_frame,
                "source": final_release_source,
                "confidence": final_release_confidence,
            },
        },

        "arm_slot": {
            "category": arm_slot_category,
            "angle_deg": arm_slot_angle,
            "confidence": arm_slot_confidence,
            "source": (
                "sports2d_2d_projected"
                if arm_slot_category is not None
                else "unknown"
            ),
        },

        "reference": {
            "knee_lift_frame": manual_knee,
            "front_foot_contact_frame": manual_fc,
            "release_frame": manual_release,
            "arm_slot_label": normalize_text(
                pitch.get("arm_slot")
            ),
        },

        "automatic": {
            "knee_lift_frame": auto_knee_frame,
            "front_foot_contact_frame": auto_fc,
            "release_frame": auto_release,
            "release_source": release_source,
            "release_confidence": release_confidence,
            "arm_slot_category": arm_slot_category,
            "arm_slot_angle_deg": arm_slot_angle,
            "arm_slot_confidence": arm_slot_confidence,
        },

        "metadata": {
            "fps": fps,
            "coordinate_system": "single_camera_2d",
            "release_definition": "wrist_motion_based_candidate",
            "arm_slot_definition": "2d_projected_shoulder_to_wrist",
            "manual_overrides_applied": bool(
                use_manual_overrides
            ),
        },
    }

    return payload


def validate_payload(payload: dict):
    errors = []

    events = payload.get("events", {})

    knee = (
        events
        .get("knee_lift", {})
        .get("frame")
    )

    fc = (
        events
        .get("front_foot_contact", {})
        .get("frame")
    )

    release = (
        events
        .get("release", {})
        .get("frame")
    )

    if knee is None:
        errors.append(
            "knee_lift.frame 없음"
        )

    if fc is None:
        errors.append(
            "front_foot_contact.frame 없음"
        )

    if release is None:
        errors.append(
            "release.frame 없음"
        )

    if (
        knee is not None
        and fc is not None
        and knee >= fc
    ):
        errors.append(
            f"이벤트 순서 오류: Knee Lift {knee} >= FC {fc}"
        )

    if (
        fc is not None
        and release is not None
        and fc >= release
    ):
        errors.append(
            f"이벤트 순서 오류: FC {fc} >= Release {release}"
        )

    return errors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pitch",
        default=None,
        help="특정 pitch_id만 생성. 생략하면 enabled 전체.",
    )

    parser.add_argument(
        "--pitches_csv",
        default=str(DEFAULT_PITCHES_CSV),
    )

    parser.add_argument(
        "--stage3_root",
        default=str(DEFAULT_STAGE3_ROOT),
    )

    parser.add_argument("--stage2_root", default=str(DEFAULT_STAGE2_ROOT))

    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )

    parser.add_argument(
        "--use-manual-overrides",
        action="store_true",
        help=(
            "manual_fc_frame/manual_release_frame이 있으면 "
            "최종 canonical event에 적용"
        ),
    )

    args = parser.parse_args()

    pitches_csv = Path(args.pitches_csv)
    stage3_root = Path(args.stage3_root)
    stage2_root = Path(args.stage2_root)
    output_root = Path(args.output_root)

    pitches = load_pitches(
        pitches_csv
    )

    if args.pitch:
        pitches = [
            p
            for p in pitches
            if normalize_text(
                p.get("pitch_id")
            ) == args.pitch
        ]

        if not pitches:
            raise ValueError(
                f"pitches.csv에서 pitch_id를 찾지 못했습니다: "
                f"{args.pitch}"
            )

    success = []
    skipped = []
    failed = []

    for pitch in pitches:
        pitch_id = normalize_text(
            pitch.get("pitch_id")
        )

        stage3_path = (
            stage3_root
            / pitch_id
            / "stage3_events_final_summary.json"
        )

        if not stage3_path.exists():
            print(
                f"[SKIP] {pitch_id}: "
                f"Stage 3 summary 없음"
            )
            skipped.append(pitch_id)
            continue

        try:
            stage3 = read_json(
                stage3_path
            )

            fc_frame = normalize_optional_int(stage3.get("front_foot_contact_frame"))
            stage2_csv = (stage2_root / pitch_id / "pose_csv"
                          / f"{pitch_id}_sports2d_wrist_motion_validated.csv")
            auto_knee = (None if normalize_optional_int(pitch.get("manual_knee_lift_frame")) is not None
                         else detect_knee_lift(stage2_csv, pitch.get("throwing_side", ""),
                                               fc_frame, normalize_optional_float(stage3.get("fps"))))

            payload = build_analysis_events(
                pitch=pitch,
                stage3=stage3,
                use_manual_overrides=
                    args.use_manual_overrides,
                auto_knee_frame=auto_knee,
            )

            errors = validate_payload(
                payload
            )

            if errors:
                print(
                    f"[ERROR] {pitch_id}: "
                    + " / ".join(errors)
                )
                failed.append(pitch_id)
                continue

            output_path = (
                output_root
                / pitch_id
                / "analysis_events.json"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    payload,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"[DONE] {pitch_id}"
            )
            print(
                f"       Knee Lift : "
                f"{payload['events']['knee_lift']['frame']}"
            )
            print(
                f"       FC        : "
                f"{payload['events']['front_foot_contact']['frame']}"
            )
            print(
                f"       Release   : "
                f"{payload['events']['release']['frame']}"
            )
            print(
                f"       Arm Slot  : "
                f"{payload['arm_slot']['category']}"
            )
            print(
                f"       Output    : "
                f"{output_path}"
            )

            success.append(pitch_id)

        except Exception as exc:
            print(
                f"[ERROR] {pitch_id}: {exc}"
            )
            failed.append(pitch_id)

    print()
    print("=" * 64)
    print("analysis_events.json BUILD SUMMARY")
    print("=" * 64)
    print(f"Success : {len(success)}")
    print(f"Skipped : {len(skipped)}")
    print(f"Failed  : {len(failed)}")

    if skipped:
        print(
            "Skipped : "
            + ", ".join(skipped)
        )

    if failed:
        print(
            "Failed  : "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
