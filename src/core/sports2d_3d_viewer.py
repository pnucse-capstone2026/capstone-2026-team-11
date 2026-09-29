"""Sports2D TRC viewer primitives extracted from the original standalone viewer.

The TRC depth is a display estimate, not measured 3D position.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

SKELETON_CONNECTIONS = [
    # 몸통
    ("Head", "Neck"),
    ("Neck", "LShoulder"),
    ("Neck", "RShoulder"),
    ("LShoulder", "RShoulder"),
    ("Neck", "Hip"),
    ("LShoulder", "LHip"),
    ("RShoulder", "RHip"),
    ("LHip", "RHip"),

    # 왼팔
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),

    # 오른팔
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),

    # 왼다리
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),

    # 오른다리
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),

    # 왼발
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("LAnkle", "LSmallToe"),
    ("LHeel", "LBigToe"),

    # 오른발
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
    ("RAnkle", "RSmallToe"),
    ("RHeel", "RBigToe"),
]


JOINT_ALIASES = {
    "Head": ["Head"],
    "Neck": ["Neck"],
    "Hip": ["Hip", "MidHip", "Pelvis"],

    "LShoulder": ["LShoulder", "LeftShoulder"],
    "RShoulder": ["RShoulder", "RightShoulder"],
    "LElbow": ["LElbow", "LeftElbow"],
    "RElbow": ["RElbow", "RightElbow"],
    "LWrist": ["LWrist", "LeftWrist"],
    "RWrist": ["RWrist", "RightWrist"],

    "LHip": ["LHip", "LeftHip"],
    "RHip": ["RHip", "RightHip"],
    "LKnee": ["LKnee", "LeftKnee"],
    "RKnee": ["RKnee", "RightKnee"],
    "LAnkle": ["LAnkle", "LeftAnkle"],
    "RAnkle": ["RAnkle", "RightAnkle"],

    "LHeel": ["LHeel", "LeftHeel"],
    "RHeel": ["RHeel", "RightHeel"],
    "LBigToe": ["LBigToe", "LeftBigToe"],
    "RBigToe": ["RBigToe", "RightBigToe"],
    "LSmallToe": ["LSmallToe", "LeftSmallToe"],
    "RSmallToe": ["RSmallToe", "RightSmallToe"],
}

BODY_LABELS = {
    "LShoulder": "L 어깨", "RShoulder": "R 어깨",
    "LHip": "L 골반", "RHip": "R 골반",
}


# ------------------------------------------------------------
# TRC 읽기
# ------------------------------------------------------------

def _split_line(line: str):
    if "\t" in line:
        return [x.strip() for x in line.rstrip("\n").split("\t")]
    return line.strip().split()


def read_trc(path: str | Path):
    """
    일반적인 OpenSim TRC 형식 파서.
    marker header 2줄 구조를 처리한다.
    """
    path = Path(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    if len(lines) < 6:
        raise ValueError("TRC 파일 형식이 너무 짧습니다.")

    header_idx = None

    for i, line in enumerate(lines):
        if "Frame#" in line and "Time" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("TRC에서 Frame#/Time header를 찾지 못했습니다.")

    marker_line = _split_line(lines[header_idx])
    coord_line = _split_line(lines[header_idx + 1])

    marker_names = []
    current_marker = None

    # TRC 첫 두 column은 Frame#, Time
    for token in marker_line[2:]:
        token = token.strip()

        if token:
            current_marker = token
            marker_names.append(current_marker)

    if not marker_names:
        raise ValueError("TRC에서 marker 이름을 찾지 못했습니다.")

    # 데이터는 header 다음 한 줄 뒤부터 시작
    data_start = header_idx + 2

    rows = []

    for line in lines[data_start:]:
        if not line.strip():
            continue

        parts = _split_line(line)

        # 충분한 숫자가 없으면 skip
        if len(parts) < 2 + len(marker_names) * 3:
            continue

        try:
            frame = int(float(parts[0]))
            time = float(parts[1])
        except ValueError:
            continue

        row = {
            "frame": frame,
            "time": time,
        }

        idx = 2

        for marker in marker_names:
            xyz = parts[idx:idx + 3]
            idx += 3

            try:
                x = float(xyz[0]) if xyz[0] != "" else np.nan
                y = float(xyz[1]) if xyz[1] != "" else np.nan
                z = float(xyz[2]) if xyz[2] != "" else np.nan
            except Exception:
                x = y = z = np.nan

            row[f"{marker}_x"] = x
            row[f"{marker}_y"] = y
            row[f"{marker}_z"] = z

        rows.append(row)

    if not rows:
        raise ValueError("TRC에서 데이터 row를 읽지 못했습니다.")

    df = pd.DataFrame(rows)

    return df, marker_names


# ------------------------------------------------------------
# marker / joint 매핑
# ------------------------------------------------------------

def normalize_name(s: str):
    return "".join(ch.lower() for ch in s if ch.isalnum())


def resolve_joint_map(marker_names: list[str]):
    normalized = {
        normalize_name(name): name
        for name in marker_names
    }

    result = {}

    for canonical, aliases in JOINT_ALIASES.items():
        found = None

        for alias in aliases:
            key = normalize_name(alias)
            if key in normalized:
                found = normalized[key]
                break

        if found is None:
            for marker in marker_names:
                n = normalize_name(marker)
                if any(
                    n.endswith(normalize_name(alias))
                    for alias in aliases
                ):
                    found = marker
                    break

        if found is not None:
            result[canonical] = found

    return result


def get_xyz(row, marker_name):
    """
    TRC 원본 좌표를 읽는다.
    Sports2D/OpenSim 계열에서는 일반적으로 Y가 수직축이고
    Z는 깊이축에 가깝다.
    """
    vals = []

    for axis in ["x", "y", "z"]:
        col = f"{marker_name}_{axis}"
        if col not in row.index:
            return None
        vals.append(row[col])

    if not all(np.isfinite(v) for v in vals):
        return None

    return tuple(float(v) for v in vals)


def to_display_xyz(xyz, floor_y=0.0):
    """
    Plotly 표시 좌표:
      display X = TRC X
      display Y = TRC Z (깊이)
      display Z = TRC Y - floor_y (수직)
    """
    x, y, z = xyz
    return (
        float(x),
        float(z),
        float(y - floor_y),
    )


def estimate_floor_y(df, marker_names):
    """
    발 관련 marker의 Y 좌표에서 낮은 쪽 값을 이용해 지면을 추정.
    Sports2D/OpenSim meter TRC의 Y를 수직축으로 가정한다.
    """
    foot_tokens = (
        "ankle",
        "heel",
        "bigtoe",
        "smalltoe",
        "toe",
    )

    values = []

    for marker in marker_names:
        n = normalize_name(marker)

        if not any(token in n for token in foot_tokens):
            continue

        col = f"{marker}_y"

        if col not in df.columns:
            continue

        arr = pd.to_numeric(
            df[col],
            errors="coerce",
        ).dropna().to_numpy()

        if len(arr):
            values.extend(arr.tolist())

    if not values:
        # fallback: 모든 marker의 Y 중 1 percentile
        for marker in marker_names:
            col = f"{marker}_y"
            if col in df.columns:
                arr = pd.to_numeric(
                    df[col],
                    errors="coerce",
                ).dropna().to_numpy()
                if len(arr):
                    values.extend(arr.tolist())

    if not values:
        return 0.0

    # outlier에 덜 민감하게 2 percentile 사용
    return float(np.nanpercentile(values, 2.0))


# ------------------------------------------------------------
# 시각화
# ------------------------------------------------------------

def compute_global_bounds(df, marker_names, floor_y):
    """
    화면 표시 좌표 기준 범위 계산.
    display X = TRC X
    display Y = TRC Z
    display Z = TRC Y - floor
    """
    xs, ys, zs = [], [], []

    for marker in marker_names:
        xcol = f"{marker}_x"
        ycol = f"{marker}_y"
        zcol = f"{marker}_z"

        if not all(c in df.columns for c in [xcol, ycol, zcol]):
            continue

        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        z = pd.to_numeric(df[zcol], errors="coerce")

        valid = x.notna() & y.notna() & z.notna()

        if valid.any():
            xs.extend(x[valid].tolist())
            ys.extend(z[valid].tolist())               # depth
            zs.extend((y[valid] - floor_y).tolist())   # vertical

    if not xs or not ys or not zs:
        return {
            "x": (-1, 1),
            "y": (-1, 1),
            "z": (0, 2),
        }

    def pad_range(arr, pad_ratio=0.08):
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        span = hi - lo

        if span <= 1e-9:
            span = 1.0

        pad = span * pad_ratio
        return lo - pad, hi + pad

    xr = pad_range(xs)
    yr = pad_range(ys)
    zr = pad_range(zs)

    # 지면이 화면 아래에 오도록 vertical lower bound는 0 근처로 정리
    z_lo = min(0.0, zr[0])
    zr = (z_lo, zr[1])

    return {
        "x": xr,
        "y": yr,
        "z": zr,
    }


def make_ground_trace(bounds):
    """
    z=0 평면에 얇은 지면 grid를 표시한다.
    """
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]

    xx = np.linspace(x0, x1, 12)
    yy = np.linspace(y0, y1, 12)

    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros_like(X)

    return go.Surface(
        x=X,
        y=Y,
        z=Z,
        showscale=False,
        opacity=0.18,
        hoverinfo="skip",
        name="Ground",
    )


def default_camera():
    """
    사람을 똑바로 세워 보는 기본 시점.
    turntable dragmode와 함께 쓰면 좌우 회전 + 줌 중심으로 조작 가능.
    """
    return dict(
        eye=dict(x=1.7, y=1.7, z=1.15),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0.15),
        projection=dict(type="orthographic"),
    )


def make_frame_figure(
    df,
    frame_number,
    joint_map,
    marker_names,
    bounds,
    floor_y,
    marker_size=2.5,
    line_width=4.0,
    fc_frame=None,
    release_frame=None,
    arm_slot=None,
    unreliable_arm_frames=None,
    throwing_side=None,
):
    frame_rows = df[df["frame"] == frame_number]

    if frame_rows.empty:
        row = df.iloc[
            (df["frame"] - frame_number).abs().argmin()
        ]
        actual_frame = int(row["frame"])
    else:
        row = frame_rows.iloc[0]
        actual_frame = int(frame_number)

    joint_points = {}

    for canonical, marker in joint_map.items():
        # Sports2D can misplace the fast moving elbow/wrist near release.
        # Omit only that arm's uncertain joints instead of drawing a misleading 3D pose.
        if (unreliable_arm_frames and actual_frame in unreliable_arm_frames
                and throwing_side in ("L", "R")
                and canonical in (f"{throwing_side}Elbow", f"{throwing_side}Wrist")):
            continue
        xyz = get_xyz(row, marker)
        if xyz is not None:
            joint_points[canonical] = to_display_xyz(
                xyz,
                floor_y=floor_y,
            )

    traces = []

    # skeleton lines
    for a, b in SKELETON_CONNECTIONS:
        if a not in joint_points or b not in joint_points:
            continue

        p1 = joint_points[a]
        p2 = joint_points[b]

        traces.append(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(width=line_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # marker points
    marker_x = []
    marker_y = []
    marker_z = []
    marker_text = []
    marker_labels = []

    for canonical, xyz in joint_points.items():
        marker_x.append(xyz[0])
        marker_y.append(xyz[1])
        marker_z.append(xyz[2])
        marker_text.append(canonical)
        marker_labels.append(BODY_LABELS.get(canonical, ""))

    traces.append(
        go.Scatter3d(
            x=marker_x,
            y=marker_y,
            z=marker_z,
            mode="markers+text",
            marker=dict(size=marker_size),
            text=marker_labels,
            hovertext=marker_text,
            textposition="top center",
            textfont=dict(size=11, color="#e35d39"),
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        )
    )

    event_text = []

    if fc_frame is not None and actual_frame == fc_frame:
        event_text.append("Front Foot Contact")

    if release_frame is not None and actual_frame == release_frame:
        event_text.append("Release Candidate")

    if arm_slot:
        event_text.append(f"Arm Slot: {arm_slot}")

    subtitle = " | ".join(event_text)

    title = f"Frame {actual_frame}"

    if subtitle:
        title += f" — {subtitle}"

    traces.insert(
        0,
        make_ground_trace(bounds),
    )

    fig = go.Figure(
        data=traces
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            xaxis=dict(
                title="X",
                range=list(bounds["x"]),
                showspikes=False,
            ),
            yaxis=dict(
                title="Y",
                range=list(bounds["y"]),
                showspikes=False,
            ),
            zaxis=dict(
                title="Z",
                range=list(bounds["z"]),
                showspikes=False,
            ),
            aspectmode="data",
            dragmode="turntable",
            camera=default_camera(),
        ),
        uirevision="sports2d-camera-v5-1",
    )

    return fig


def make_animation_figure(
    df,
    joint_map,
    marker_names,
    bounds,
    floor_y,
    start_frame,
    end_frame,
    frame_step,
    marker_size=2.5,
    line_width=4.0,
    fc_frame=None,
    release_frame=None,
    arm_slot=None,
    unreliable_arm_frames=None,
    throwing_side=None,
):
    selected_frames = list(
        range(
            int(start_frame),
            int(end_frame) + 1,
            int(frame_step),
        )
    )

    if not selected_frames:
        selected_frames = [int(start_frame)]

    def traces_for_frame(frame_number):
        frame_rows = df[df["frame"] == frame_number]

        if frame_rows.empty:
            row = df.iloc[
                (df["frame"] - frame_number).abs().argmin()
            ]
            actual_frame = int(row["frame"])
        else:
            row = frame_rows.iloc[0]
            actual_frame = int(frame_number)

        joint_points = {}

        for canonical, marker in joint_map.items():
            # Preserve valid joints and suppress only unreliable throwing-arm
            # joints; the fixed TRC depth cannot repair an occluded 2D limb.
            if (unreliable_arm_frames and actual_frame in unreliable_arm_frames
                    and throwing_side in ("L", "R")
                    and canonical in (f"{throwing_side}Elbow", f"{throwing_side}Wrist")):
                continue
            xyz = get_xyz(row, marker)
            if xyz is not None:
                joint_points[canonical] = to_display_xyz(
                    xyz,
                    floor_y=floor_y,
                )

        x_lines, y_lines, z_lines = [], [], []

        for a, b in SKELETON_CONNECTIONS:
            if a not in joint_points or b not in joint_points:
                continue

            p1 = joint_points[a]
            p2 = joint_points[b]

            x_lines += [p1[0], p2[0], None]
            y_lines += [p1[1], p2[1], None]
            z_lines += [p1[2], p2[2], None]

        marker_x = []
        marker_y = []
        marker_z = []
        marker_text = []
        marker_labels = []

        for canonical, xyz in joint_points.items():
            marker_x.append(xyz[0])
            marker_y.append(xyz[1])
            marker_z.append(xyz[2])
            marker_text.append(canonical)
            marker_labels.append(BODY_LABELS.get(canonical, ""))

        line_trace = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(width=line_width),
            hoverinfo="skip",
            showlegend=False,
        )

        point_trace = go.Scatter3d(
            x=marker_x,
            y=marker_y,
            z=marker_z,
            mode="markers+text",
            marker=dict(size=marker_size),
            text=marker_labels,
            hovertext=marker_text,
            textposition="top center",
            textfont=dict(size=11, color="#e35d39"),
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        )

        title = f"Frame {actual_frame}"

        flags = []

        if fc_frame is not None and actual_frame == fc_frame:
            flags.append("FC")

        if release_frame is not None and actual_frame == release_frame:
            flags.append("Release")

        if flags:
            title += " — " + " / ".join(flags)

        return [line_trace, point_trace], title

    initial_data, initial_title = traces_for_frame(
        selected_frames[0]
    )

    animation_frames = []

    slider_steps = []

    for frame_number in selected_frames:
        data, title = traces_for_frame(frame_number)

        animation_frames.append(
            go.Frame(
                data=data,
                traces=[1, 2],
                name=str(frame_number),
            )
        )

        slider_steps.append(
            {
                "args": [
                    [str(frame_number)],
                    {
                        "frame": {
                            "duration": 0,
                            "redraw": True,
                        },
                        "mode": "immediate",
                        "transition": {
                            "duration": 0
                        },
                    },
                ],
                "label": str(frame_number),
                "method": "animate",
            }
        )

    fig = go.Figure(
        data=[
            make_ground_trace(bounds),
            *initial_data,
        ],
        frames=animation_frames,
    )

    fig.update_layout(
        title="3D Pitching Motion",
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            xaxis=dict(
                title="X",
                range=list(bounds["x"]),
                showspikes=False,
            ),
            yaxis=dict(
                title="Y",
                range=list(bounds["y"]),
                showspikes=False,
            ),
            zaxis=dict(
                title="Z",
                range=list(bounds["z"]),
                showspikes=False,
            ),
            aspectmode="data",
            dragmode="turntable",
            camera=default_camera(),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.10,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 35,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 0
                                },
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0
                                },
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Frame: "
                },
                "pad": {
                    "t": 45
                },
                "steps": slider_steps,
            }
        ],
        uirevision="sports2d-camera-v5-1",
    )

    if arm_slot:
        fig.add_annotation(
            text=f"Arm Slot: {arm_slot}",
            x=1,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right",
        )

    return fig



# ------------------------------------------------------------
# Camera-persistent HTML animation
# ------------------------------------------------------------

def render_live_zoom_animation(
    fig,
    height=760,
):
    """
    Streamlit의 st.plotly_chart 대신 독립 HTML/JS Plotly를 사용한다.

    핵심:
    - animation frame마다 Plotly.animate/redraw를 하지 않는다.
    - line/marker trace의 x/y/z 값만 Plotly.restyle로 갱신한다.
    - scene.camera에는 손대지 않는다.

    따라서 Play 중에도 사용자가 휠로 zoom하거나 좌우 회전한 값이
    다음 프레임에서 덮어써지지 않는다.
    """
    fig_dict = fig.to_plotly_json()

    data_json = json.dumps(
        fig_dict.get("data", []),
        ensure_ascii=False,
    )

    layout_json = json.dumps(
        fig_dict.get("layout", {}),
        ensure_ascii=False,
    )

    frames = fig_dict.get("frames", [])

    # 필요한 animation data만 JS-friendly 구조로 정리
    compact_frames = []

    for fr in frames:
        compact_frames.append(
            {
                "name": fr.get("name"),
                "data": fr.get("data", []),
                "traces": fr.get("traces", [1, 2]),
            }
        )

    frames_json = json.dumps(
        compact_frames,
        ensure_ascii=False,
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: Arial, sans-serif;
  }}

  #controls {{
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 4px 0 8px 0;
  }}

  button {{
    border: 1px solid #777;
    border-radius: 6px;
    padding: 6px 12px;
    cursor: pointer;
  }}

  #frameSlider {{
    flex: 1;
    min-width: 240px;
  }}

  #frameLabel {{
    min-width: 90px;
    font-size: 14px;
  }}

  #plot {{
    width: 100%;
    height: {height - 60}px;
  }}
</style>
</head>
<body>

<div id="controls">
  <button id="playBtn">▶ Play</button>
  <button id="pauseBtn">⏸ Pause</button>
  <input id="frameSlider" type="range" min="0" max="0" value="0" step="1">
  <span id="frameLabel">Frame</span>
</div>

<div id="plot"></div>

<script>
const initialData = {data_json};
const initialLayout = {layout_json};
const frames = {frames_json};

const gd = document.getElementById("plot");
const slider = document.getElementById("frameSlider");
const label = document.getElementById("frameLabel");

let playing = false;
let playTimer = null;
let currentIndex = 0;

// Streamlit 내부 iframe에서는 responsive가 중요하다.
const config = {{
  responsive: true,
  displaylogo: false,
  scrollZoom: true
}};

// Plotly 자체 animation control은 제거.
// 아래 JS control만 쓴다.
if (initialLayout.updatemenus) delete initialLayout.updatemenus;
if (initialLayout.sliders) delete initialLayout.sliders;

Plotly.newPlot(
  gd,
  initialData,
  initialLayout,
  config
).then(() => {{
  slider.max = Math.max(0, frames.length - 1);

  if (frames.length > 0) {{
    label.textContent = "Frame: " + frames[0].name;
  }}
}});

async function showFrame(index) {{
  if (!frames.length) return;

  index = Math.max(
    0,
    Math.min(index, frames.length - 1)
  );

  currentIndex = index;

  const fr = frames[index];

  const update = {{}};

  // line trace
  if (fr.data[0]) {{
    update.x = [fr.data[0].x, fr.data[1] ? fr.data[1].x : []];
    update.y = [fr.data[0].y, fr.data[1] ? fr.data[1].y : []];
    update.z = [fr.data[0].z, fr.data[1] ? fr.data[1].z : []];

    if (fr.data[1]) {{
      update.text = [null, fr.data[1].text || []];
      update.hovertext = [null, fr.data[1].hovertext || []];
    }}
  }}

  // ground가 trace 0, skeleton line/marker가 1/2
  await Plotly.restyle(
    gd,
    update,
    [1, 2]
  );

  slider.value = index;
  label.textContent = "Frame: " + fr.name;
}}

function stopPlay() {{
  playing = false;

  if (playTimer !== null) {{
    clearTimeout(playTimer);
    playTimer = null;
  }}
}}

function playNext() {{
  if (!playing) return;

  showFrame(currentIndex).then(() => {{
    currentIndex++;

    if (currentIndex >= frames.length) {{
      currentIndex = 0;
    }}

    if (playing) {{
      // 약 30fps. trace 좌표만 갱신하므로 재생 중에도 zoom/rotate 가능.
      playTimer = setTimeout(playNext, 33);
    }}
  }});
}}

document.getElementById("playBtn").addEventListener("click", () => {{
  if (playing) return;

  playing = true;
  playNext();
}});

document.getElementById("pauseBtn").addEventListener("click", () => {{
  stopPlay();
}});

slider.addEventListener("input", () => {{
  stopPlay();
  showFrame(parseInt(slider.value));
}});
</script>

</body>
</html>
"""

    components.html(
        html,
        height=height,
        scrolling=False,
    )
