"""Synchronized 2D lead-leg and progressive knee-height animations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .form_comparison import _body_scale, _coordinate, _side, _value


SIZE = (620, 450)
BLUE = "#1678b8"
ORANGE = "#ee7750"


def _font(size: int):
    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _save(frames: list[Image.Image], duration: int) -> bytes:
    out = BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    return out.getvalue()


def make_knee_lift_gifs(motion_csv: Path, features_csv: Path, *, max_frames: int = 72,
                        duration_ms: int = 130) -> tuple[bytes, bytes]:
    """Draw the timeline through the release candidate; sample for responsive GIFs.

    Height is (lead hip image y - lead knee image y) / shoulder-to-hip length.
    It is a projected 2D relative height, not distance above the ground.
    """
    df = pd.read_csv(motion_csv).dropna(subset=["frame"])
    if df.empty:
        raise ValueError("정규화 Pose CSV가 비어 있습니다.")
    df = df.drop_duplicates("frame").sort_values("frame")
    info = pd.read_csv(features_csv).iloc[0]
    side = _side(_value(info, "throwing_side"))
    lead = "R" if side == "L" else "L"
    knee_frame = int(float(_value(info, "knee_lift_frame", "maximum_knee_lift_frame")))
    release_frame = int(float(_value(
        info, "release_candidate_frame", "release_frame", "pose_release_candidate_frame"
    )))
    df = df.loc[pd.to_numeric(df["frame"], errors="coerce") <= release_frame]
    if df.empty:
        raise ValueError("릴리스 후보 이전의 유효한 프레임이 없습니다.")

    rows = []
    for _, row in df.iterrows():
        try:
            hip, knee, ankle = [_coordinate(row, lead + joint) for joint in ("Hip", "Knee", "Ankle")]
            scale = _body_scale(row)
        except ValueError:
            continue
        if not np.isfinite(scale) or scale <= 0:
            continue
        points = [((p[0] - hip[0]) / scale, (hip[1] - p[1]) / scale) for p in (hip, knee, ankle)]
        rows.append((int(row["frame"]), points, points[1][1]))
    if len(rows) < 2:
        raise ValueError("다리 관절이 유효한 프레임이 부족합니다.")

    # Keep the available range through the release candidate and the Knee Lift frame.
    pick = set(np.linspace(0, len(rows) - 1, min(max_frames, len(rows)), dtype=int).tolist())
    pick.add(min(range(len(rows)), key=lambda i: abs(rows[i][0] - knee_frame)))
    indices = sorted(pick)
    first, last = rows[0][0], rows[-1][0]
    heights = [v[2] for v in rows]
    low, high = min(heights), max(heights)
    pad = max(0.12, (high - low) * 0.12)
    low -= pad
    high += pad
    extent = max(0.85, max(abs(v) for _, pts, _ in rows for p in pts for v in p) * 1.12)
    leg_frames, graph_frames = [], []
    regular = _font(17)
    bold = _font(23)

    def px(frame):
        return 78 + (frame - first) / max(1, last - first) * 505

    def py(height):
        return 374 - (height - low) / (high - low) * 272

    for index in indices:
        frame, points, height = rows[index]
        progress = (frame - first) / max(1, last - first)
        leg = Image.new("RGB", SIZE, "white")
        d = ImageDraw.Draw(leg)
        d.text((25, 18), "Lead leg motion", font=bold, fill="#202833")
        d.text((25, 51), f"Frame {frame}  |  Lead {lead} leg", font=regular, fill="#546171")
        d.line((70, 371, 555, 371), fill="#bcc6d0", width=2)
        ox, oy = 310, 192
        scale = 145 / extent
        xy = [(round(ox + x * scale), round(oy - y * scale)) for x, y in points]
        d.line(xy, fill=ORANGE, width=9, joint="curve")
        for x, y in xy:
            d.ellipse((x-9, y-9, x+9, y+9), fill=ORANGE)
        d.text((25, 401), "2D projected pose  |  hip aligned", font=regular, fill="#667381")
        d.rounded_rectangle((24, 384, 596, 390), radius=3, fill="#edf0f2")
        d.rounded_rectangle((24, 384, 24 + round(572*progress), 390), radius=3, fill=ORANGE)
        leg_frames.append(leg)

        graph = Image.new("RGB", SIZE, "white")
        g = ImageDraw.Draw(graph)
        g.text((25, 18), "Lead knee height over time", font=bold, fill="#202833")
        g.text((25, 51), "Relative to hip / torso length", font=regular, fill="#546171")
        for tick in (low, (low+high)/2, high):
            y = round(py(tick))
            g.line((78, y, 583, y), fill="#e9edf2", width=1)
            g.text((13, y-9), f"{tick:.1f}", font=_font(13), fill="#6d7987")
        zero = py(0)
        if 100 <= zero <= 374:
            g.line((78, round(zero), 583, round(zero)), fill="#9fabb7", width=1)
        event_x = round(px(knee_frame))
        if first <= knee_frame <= frame:
            g.line((event_x, 96, event_x, 374), fill="#eeb0a0", width=2)
            g.text((min(event_x + 6, 490), 101), "Knee Lift", font=_font(14), fill="#a04b35")
        seen = [(round(px(f)), round(py(h))) for f, _, h in rows[:index+1]]
        if len(seen) >= 2:
            g.line(seen, fill=BLUE, width=4, joint="curve")
        x, y = seen[-1]
        g.ellipse((x-6, y-6, x+6, y+6), fill=ORANGE)
        g.line((x, 96, x, 374), fill="#e9a392", width=2)
        g.line((78, 374, 583, 374), fill="#8c9aa9", width=2)
        g.text((80, 398), f"First {first}F", font=_font(14), fill="#536070")
        g.text((475, 398), f"Last {last}F", font=_font(14), fill="#536070")
        g.text((250, 398), f"Current {frame}F", font=_font(14), fill="#a04b35")
        graph_frames.append(graph)

    return _save(leg_frames, duration_ms), _save(graph_frames, duration_ms)
