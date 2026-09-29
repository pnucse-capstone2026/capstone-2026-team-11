"""Animated 2D throwing-arm and lead-leg comparisons for Streamlit."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .form_comparison import _body_scale, _coordinate, _side, _value


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _arm_at(row: pd.Series, side: str) -> list[tuple[float, float]]:
    shoulder = _coordinate(row, side + "Shoulder")
    scale = _body_scale(row)
    return [
        ((_coordinate(row, side + joint)[0] - shoulder[0]) / scale,
         (shoulder[1] - _coordinate(row, side + joint)[1]) / scale)
        for joint in ("Shoulder", "Elbow", "Wrist")
    ]


def _lead_leg_at(row: pd.Series, throwing_side: str) -> list[tuple[float, float]]:
    stride = "R" if throwing_side == "L" else "L"
    hip = _coordinate(row, stride + "Hip")
    scale = _body_scale(row)
    return [
        ((_coordinate(row, stride + joint)[0] - hip[0]) / scale,
         (hip[1] - _coordinate(row, stride + joint)[1]) / scale)
        for joint in ("Hip", "Knee", "Ankle")
    ]


def _load_pitch(motion_csv: Path, features_csv: Path) -> dict:
    data = pd.read_csv(motion_csv)
    if "frame" not in data:
        raise ValueError(f"{motion_csv.name}: frame 열이 없습니다.")
    data = data.dropna(subset=["frame"]).drop_duplicates("frame").set_index("frame").sort_index()
    feature = pd.read_csv(features_csv).iloc[0]
    side = _side(_value(feature, "throwing_side"))
    knee = int(feature["knee_lift_frame"])
    fc = int(feature["front_foot_contact_frame"])
    release = int(feature["release_frame"])
    if not knee < fc < release:
        raise ValueError("Knee Lift < FC < Release 프레임 순서가 필요합니다.")
    return {"data": data, "side": side, "release": release, "knee": knee}


def make_arm_comparison_gif(
    motion_a: Path, features_a: Path, label_a: str,
    motion_b: Path, features_b: Path, label_b: str,
    *, window: int = 10,
) -> bytes:
    """Animate matched frame offsets around each pose-based release candidate.

    The end frame is a pose-based release candidate. Camera perspective and
    left/right marker errors are not corrected. This GIF is a visual comparison.
    """
    return _make_gif(motion_a, features_a, label_a, motion_b, features_b, label_b,
                     window=window, mode="arm")


def make_leg_comparison_gif(
    motion_a: Path, features_a: Path, label_a: str,
    motion_b: Path, features_b: Path, label_b: str,
    *, window: int = 20,
) -> bytes:
    """Animate the stride-side hip, knee and ankle around the Knee Lift reference."""
    return _make_gif(motion_a, features_a, label_a, motion_b, features_b, label_b,
                     window=window, mode="leg")


def _make_gif(
    motion_a: Path, features_a: Path, label_a: str,
    motion_b: Path, features_b: Path, label_b: str,
    *, window: int, mode: str,
) -> bytes:
    pitches = [_load_pitch(motion_a, features_a), _load_pitch(motion_b, features_b)]
    if window < 1 or window > 60:
        raise ValueError("window는 1~60 프레임이어야 합니다.")
    samples = []
    offsets = range(-window, window + 1)
    for offset in offsets:
        frame_data = []
        for pitch in pitches:
            index = pitch["release" if mode == "arm" else "knee"] + offset
            if index not in pitch["data"].index:
                frame_data.append((None, index, False))
                continue
            row = pitch["data"].loc[index]
            try:
                arm = (_arm_at(row, pitch["side"]) if mode == "arm"
                       else _lead_leg_at(row, pitch["side"]))
            except ValueError:
                arm = None
            valid = bool(row.get("throwing_elbow_angle_valid", True)) if mode == "arm" else True
            frame_data.append((arm, int(index), valid))
        samples.append(frame_data)

    all_points = [point for sample in samples for arm, _, _ in sample if arm for point in arm]
    if not all_points:
        raise ValueError("유효한 관절 좌표를 찾지 못했습니다.")
    extent = max(0.8, max(abs(v) for point in all_points for v in point) * 1.14)
    size = (760, 550)
    origin = (330, 282)
    scale = 185 / extent
    def screen(point: tuple[float, float]) -> tuple[int, int]:
        return round(origin[0] + point[0] * scale), round(origin[1] - point[1] * scale)

    frames = []
    colors = ("#ed755f", "#1aa7bd")
    for offset, sample in zip(offsets, samples):
        im = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(im)
        title = "Throwing arm motion" if mode == "arm" else "Lead leg lift motion"
        event = "Release candidate" if mode == "arm" else "Knee Lift reference"
        phase = ("AUTO CANDIDATE" if mode == "arm" else "REFERENCE") if offset == 0 else ("BEFORE" if offset < 0 else "AFTER")
        draw.text((32, 18), title, fill="#222", font=_font(22))
        draw.text((32, 47), f"{event} {offset:+d} frames  |  {phase}",
                  fill="#666", font=_font(15))
        for tick in (-1, -.5, 0, .5, 1):
            if abs(tick) > extent:
                continue
            xx, yy = screen((tick, tick))
            draw.line((xx, 83, xx, 477), fill="#ebebeb", width=1)
            draw.line((65, yy, 540, yy), fill="#ebebeb", width=1)
        draw.text((175, 505), "Horizontal position / torso length", fill="#555", font=_font(14))
        draw.text((17, 270), "Height", fill="#555", font=_font(14))
        for (arm, frame, valid), label, color, y in zip(sample, (label_a, label_b), colors, (214, 252)):
            draw.ellipse((560, y + 2, 576, y + 18), fill=color)
            draw.text((583, y), label, fill="#333", font=_font(16))
            draw.text((583, y + 19), f"frame {frame}" + (" *" if not valid else ""),
                      fill="#777", font=_font(13))
            if arm is None:
                continue
            points = [screen(point) for point in arm]
            draw.line(points, fill=color, width=7, joint="curve")
            for x, yy in points:
                draw.ellipse((x - 6, yy - 6, x + 6, yy + 6), fill=color)
        if mode == "arm":
            draw.text((550, 324), "* Elbow angle flagged", fill="#777", font=_font(12))
        else:
            draw.text((550, 324), "Hip - Knee - Ankle", fill="#777", font=_font(12))
        draw.text((550, 346), "2D projection only", fill="#777", font=_font(12))
        frames.append(im)
    buffer = BytesIO()
    # Pause at the candidate without treating it as visually verified release.
    sequence = frames[:window + 1] + [frames[window]] * 5 + frames[window + 1:] + [frames[-1]] * 3
    sequence[0].save(buffer, format="GIF", save_all=True, append_images=sequence[1:],
                     duration=160, loop=0, optimize=True)
    return buffer.getvalue()
