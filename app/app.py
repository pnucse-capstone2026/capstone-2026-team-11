from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import math

# ============================================================
# Project Paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if APP_DIR.name.lower() == "app" else APP_DIR

SRC_DIR = ROOT / "src"
CONFIG_DIR = ROOT / "configs"
RAW_VIDEO_DIR = ROOT / "data" / "raw_videos"
FEATURE_DIR = ROOT / "data" / "features"
PIPELINE_OUTPUT_DIR = ROOT / "pipeline_outputs"
SPORTS2D_DIR = ROOT / "sports2d"
PITCHES_CSV = CONFIG_DIR / "pitches.csv"
RUN_PIPELINE = SRC_DIR / "pipelines" / "run_pipeline.py"
PREPARE_UPLOAD = SRC_DIR / "pipelines" / "prepare_uploaded_sports2d.py"
MANUAL_ROI_CROP = SRC_DIR / "sports2d" / "manual_roi_crop.py"
DATA_DIR = ROOT / "data"

KNEE_LIFT_TEMPLATE_IMAGE = ROOT / "assets" / "pitcher_knee_lift.png"
STRIDE_KNEE_TEMPLATE_IMAGE = ROOT / "assets" / "pitcher_FC_stride_knee.png"
ARM_ANGLE_TEMPLATE_NORMAL = ROOT / "assets" / "pitcher_arm_angle.png"
ARM_ANGLE_TEMPLATE_UNDERHAND = ROOT / "assets" / "pitcher_arm_angle_under.png"
CARD_DISPLAY_WIDTH = 430

# 기준 투수 Arm Angle.
# MLB 기준 투수는 Feature CSV의 2D 추정값보다 아래 기준값을 우선 사용합니다.
# 커쇼/휠러처럼 Statcast 직접값이 아닌 항목은 source에서 명시합니다.
REFERENCE_ARM_ANGLES = {
    "kershaw_02": {
        "angle": 65.0,
        "source": "Historical Estimate",
        "reference": "2016–2017 영상 시기 추정",
    },
    "verlander_01": {
        "angle": 43.0,
        "source": "Nearest Statcast",
        "reference": "2020 Statcast (영상은 2019년경)",
    },
    "cole_01": {
        "angle": 40.0,
        "source": "Statcast Representative",
        "reference": "38–43° 범위 대표값",
    },
    "sale_01": {
        "angle": 12.0,
        "source": "Statcast Representative",
        "reference": "8–15° 범위 대표값",
    },
    "wheeler_01": {
        "angle": 28.0,
        "source": "Estimated from Video",
        "reference": "영상 자세 기반 임시 추정값",
    },
    "rogers_01": {
        "angle": -61.0,
        "source": "Statcast Representative",
        "reference": "-64–-59° 범위 대표값",
    },
}

# ============================================================
# Streamlit
# ============================================================

st.set_page_config(
    page_title="투구 동작 분석 | Pitching Motion Analysis",
    page_icon="⚾",
    layout="wide",
)

st.markdown("""
<style>
/* 전체 페이지 폭 제한 + 가운데 정렬 */
.main .block-container {
    max-width: 1280px;
    margin-left: auto;
    margin-right: auto;
    padding-left: 3.5rem;
    padding-right: 3.5rem;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* 너무 큰 모니터에서도 중앙에 모이게 */
@media (min-width: 1600px) {
    .main .block-container {
        max-width: 1360px;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .app-subtitle {
        color: #9aa4b2;
        font-size: 0.98rem;
        margin-top: -0.25rem;
        margin-bottom: 0.8rem;
    }
    .section-note {
        border: 1px solid rgba(49, 130, 206, 0.25);
        background: rgba(49, 130, 206, 0.10);
        border-radius: 0.65rem;
        padding: 0.9rem 1rem;
        margin-top: 0.5rem;
    }
    @media (max-width: 900px) {
        .app-subtitle {
            font-size: 0.90rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

ARM_SLOT_OPTIONS = [
    "unknown",
    "high_overhand",
    "overhand",
    "high_three_quarter",
    "three_quarter",
    "low_three_quarter",
    "sidearm",
    "submarine",
]

ARM_SLOT_LABELS = {
    "unknown": "미분류 (unknown)",
    "high_overhand": "하이 오버핸드",
    "overhand": "오버핸드",
    "high_three_quarter": "하이 쓰리쿼터",
    "three_quarter": "쓰리쿼터",
    "low_three_quarter": "로우 쓰리쿼터",
    "sidearm": "사이드암",
    "submarine": "서브마린",
}


# 사용자 업로드 투수의 Arm Slot 카드를 그리기 위한 "대표 방향".
# 실제 측정 Arm Angle이 아니며, 카드의 방향선을 시각적으로 배치하는 용도로만 사용.
ARM_SLOT_VISUAL_ANGLES = {
    # 시각화용 대표 방향일 뿐 측정 각도가 아닙니다.
    "high_overhand": 82.0,
    "overhand": 72.0,
    "high_three_quarter": 55.0,
    "three_quarter": 45.0,
    "low_three_quarter": 28.0,
    "sidearm": 5.0,
    "submarine": -35.0,
}


def get_mlb_logo_source() -> str | None:
    local_candidates = [
        ROOT / "assets" / "mlb_logo.png",
        ROOT / "assets" / "mlb_logo.jpg",
        ROOT / "assets" / "mlb_logo.jpeg",
        ROOT / "assets" / "mlb_logo.webp",
        ROOT / "assets" / "mlb_logo.svg",
        ROOT / "web" / "assets" / "mlb_logo.png",
    ]
    for path in local_candidates:
        if path.exists():
            return str(path)

    # 로컬 파일이 없으면 원격 이미지 사용
    return "https://www.mlbstatic.com/team-logos/league-on-dark/1.svg"


def safe_pitch_id(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "uploaded_pitch"


def read_manifest() -> pd.DataFrame:
    if not PITCHES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PITCHES_CSV, encoding="utf-8-sig")


def upsert_manifest_row(row: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if PITCHES_CSV.exists():
        with open(PITCHES_CSV, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    else:
        fieldnames = []
        rows = []

    required = [
        "enabled",
        "pitch_id",
        "pitcher_name",
        "throwing_side",
        "arm_slot",
        "video_file",
        "manual_knee_lift_frame",
        "manual_fc_frame",
        "manual_release_frame",
    ]

    for name in required:
        if name not in fieldnames:
            fieldnames.append(name)

    normalized = {name: "" for name in fieldnames}
    normalized.update(row)

    replaced = False
    for i, old in enumerate(rows):
        if str(old.get("pitch_id", "")).strip() == str(row["pitch_id"]).strip():
            merged = {name: old.get(name, "") for name in fieldnames}
            merged.update(normalized)
            rows[i] = merged
            replaced = True
            break

    if not replaced:
        rows.append(normalized)

    with open(PITCHES_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    pitch_id: str,
    from_step: int = 1,
    to_step: int = 6,
    force: bool = False,
) -> tuple[bool, str]:
    if not RUN_PIPELINE.exists():
        return False, f"run_pipeline.py를 찾을 수 없습니다: {RUN_PIPELINE}"

    command = [
        sys.executable,
        str(RUN_PIPELINE),
        "--pitch",
        pitch_id,
        "--from-step",
        str(int(from_step)),
        "--to-step",
        str(int(to_step)),
    ]
    if force:
        command.append("--force")

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (result.stdout or "")
    if result.stderr:
        output += "\n\n[stderr]\n" + result.stderr

    return result.returncode == 0, output


def prepare_uploaded_video(pitch_id: str, video_path: Path, force: bool = False,
                           reuse_trc: bool = False) -> tuple[bool, str]:
    if not PREPARE_UPLOAD.is_file():
        return False, f"Sports2D 업로드 준비 스크립트가 없습니다: {PREPARE_UPLOAD}"
    command = [
        sys.executable, str(PREPARE_UPLOAD),
        "--pitch-id", pitch_id, "--video", str(video_path),
        "--roi-json", str(SPORTS2D_DIR / "input" / f"{pitch_id}_roi.json"),
    ]
    if force:
        command.append("--force")
    if reuse_trc:
        command.append("--reuse-trc")
    result = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    log = (result.stdout or "") + ("\n[stderr]\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, log


def select_pitcher_roi(pitch_id: str, video_path: Path) -> tuple[bool, str]:
    if not MANUAL_ROI_CROP.is_file():
        return False, f"투수 영역 선택 스크립트가 없습니다: {MANUAL_ROI_CROP}"
    crop = SPORTS2D_DIR / "cropped" / f"{pitch_id}_roi.mp4"
    roi_json = SPORTS2D_DIR / "input" / f"{pitch_id}_roi.json"
    command = [
        sys.executable, str(MANUAL_ROI_CROP),
        "--input", str(video_path), "--output", str(crop),
        "--json", str(roi_json),
    ]
    result = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    log = (result.stdout or "") + ("\n[stderr]\n" + result.stderr if result.stderr else "")
    return result.returncode == 0 and roi_json.is_file() and crop.is_file(), log


def discover_feature_files() -> dict[str, Path]:
    found: dict[str, Path] = {}

    if FEATURE_DIR.exists():
        for path in FEATURE_DIR.glob("*_features.csv"):
            if path.name == "sample_features.csv":
                continue
            pitch_id = path.stem.removesuffix("_features")
            found[pitch_id] = path

    if PIPELINE_OUTPUT_DIR.exists():
        for pitch_dir in PIPELINE_OUTPUT_DIR.iterdir():
            if not pitch_dir.is_dir():
                continue
            path = pitch_dir / f"{pitch_dir.name}_features.csv"
            if path.exists():
                found[pitch_dir.name] = path

    return dict(sorted(found.items()))

def find_feature_csv_for_pitch(pitch_id: str) -> Path | None:
    candidates = [
        FEATURE_DIR / f"{pitch_id}_features.csv",
        PIPELINE_OUTPUT_DIR / pitch_id / f"{pitch_id}_features.csv",
        PIPELINE_OUTPUT_DIR / pitch_id / "features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None

def canonical_stage2_ready(pitch_id: str) -> bool:
    path = (
        SPORTS2D_DIR
        / "stage2"
        / pitch_id
        / "pose_csv"
        / f"{pitch_id}_sports2d_wrist_motion_validated.csv"
    )
    return path.exists()



def load_feature_row(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def first_value(row: dict, candidates: list[str]):
    for key in candidates:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def fmt_number(value, digits=2, suffix="") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def fmt_frame(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"프레임 {int(float(value))}"
    except (TypeError, ValueError):
        return str(value)


def fmt_confidence(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    text = str(value)
    mapping = {
        "high": "높음 (HIGH)",
        "medium": "보통 (MEDIUM)",
        "low": "낮음 (LOW)",
    }
    return mapping.get(text.lower(), text)


def display_pitch_label(pitch_id: str, path: Path) -> str:
    row = load_feature_row(path)
    pitcher_name = first_value(row, ["pitcher_name", "Pitcher", "pitcher"])
    if pitcher_name is None:
        return pitch_id
    return str(pitcher_name)


def selection_index_map(feature_files: dict[str, Path]) -> tuple[list[str], dict[str, str]]:
    ids = list(feature_files.keys())
    names = [display_pitch_label(pid, feature_files[pid]) for pid in ids]
    labels = [name if names.count(name) == 1 else f"{name} · {pid}" for name, pid in zip(names, ids)]
    label_to_id = {label: pid for label, pid in zip(labels, ids)}
    return labels, label_to_id


def show_feature_dashboard(feature_path: Path) -> None:
    row = load_feature_row(feature_path)

    if not row:
        st.warning("Feature CSV가 비어 있습니다.")
        return

    knee_frame = first_value(row, [
        "knee_lift_frame", "maximum_knee_lift_frame", "lead_leg_lift_frame"
    ])
    fc_frame = first_value(row, [
        "front_foot_contact_frame", "fc_frame", "foot_contact_frame"
    ])
    release_frame = first_value(row, [
        "release_candidate_frame", "release_frame", "pose_release_candidate_frame"
    ])
    release_conf = first_value(row, [
        "release_confidence", "release_candidate_confidence"
    ])

    fc_release = first_value(row, [
        "fc_to_release_time_s",
        "fc_to_release_time",
        "fc_release_time_sec",
        "fc_to_release_sec",
    ])

    c1, c2, c3 = st.columns(3)
    c1.metric("최대 리드 레그 리프트\n(Maximum Lead-Leg Lift)", fmt_frame(knee_frame))
    c2.metric("앞발 착지\n(Front Foot Contact)", fmt_frame(fc_frame))
    c3.metric("Pose-based Release Candidate", fmt_frame(release_frame))

    st.metric("FC → Release 후보", fmt_number(fc_release, 3, " s"))

    if release_conf is not None:
        st.caption(f"Release Candidate 신뢰도: **{fmt_confidence(release_conf)}**")


def _safe_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def get_stride_knee_angle_fc_value(feature_path: Path) -> float | None:
    row = load_feature_row(feature_path)
    value = first_value(
        row,
        [
            "stride_knee_angle_front_foot_contact_deg",
            "stride_knee_angle_foot_contact_deg",
            "stride_knee_angle_fc",
            "stride_knee_fc",
            "stride_knee_angle_at_fc",
        ],
    )
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None

def get_knee_lift_frame_value(feature_path: Path) -> int | None:
    row = load_feature_row(feature_path)
    value = first_value(
        row,
        [
            "knee_lift_frame",
            "maximum_knee_lift_frame",
            "lead_leg_lift_frame",
        ],
    )
    try:
        if value is None or pd.isna(value):
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def get_lead_leg_lift_angle_value(feature_path: Path) -> float | None:
    row = load_feature_row(feature_path)
    value = first_value(
        row,
        [
            "lead_leg_lift_angle_2d_deg",
            "lead_leg_lift_angle_deg",
            "lead_leg_lift_deg",
            "lead_leg_lift_angle",
            "knee_lift_angle_deg",
        ],
    )
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def describe_lead_leg_lift_angle(angle_deg: float) -> str:
    if angle_deg >= 75:
        return "키킹 동작에서 앞다리를 매우 높게 들어 올린 형태"
    elif angle_deg >= 60:
        return "키킹 동작에서 앞다리를 비교적 높게 들어 올린 형태"
    elif angle_deg >= 45:
        return "키킹 동작에서 앞다리를 보통 높이로 들어 올린 형태"
    else:
        return "키킹 동작에서 앞다리를 비교적 낮게 들어 올린 형태"


def render_knee_lift_card(
    angle_deg: float,
    template_path: Path = KNEE_LIFT_TEMPLATE_IMAGE,
):
    if not template_path.exists():
        raise FileNotFoundError(f"템플릿 이미지를 찾지 못했습니다: {template_path}")

    image = Image.open(template_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    # 사용자가 지정한 기준점: pitcher_knee_lift = (538, 695)
    origin = (538, 695)
    lead_leg_end = (428, 598)
    support_leg_end = (325, 925)

    _draw_line(draw, origin, lead_leg_end, width=22, fill=(18, 24, 28))
    _draw_line(draw, lead_leg_end, support_leg_end, width=22, fill=(18, 24, 28))

    for pt in [origin, lead_leg_end, support_leg_end]:
        _draw_dot(draw, pt, r=15, fill=(18, 24, 28), outline=(240, 240, 240))

    subtitle_font = _safe_font(24, bold=False)
    value_font = _safe_font(72, bold=True)
    angle_font = _safe_font(28, bold=True)

    draw.text((250, 90), "Maximum Lead-Leg Lift", font=subtitle_font, fill=(111, 126, 130))
    draw.line((470, 130, 620, 130), fill=(122, 198, 185), width=3)

    # 시각적 각도 arc는 왼쪽 꼭지점(lead_leg_end)에 표시
    _draw_angle_arc(draw, lead_leg_end, origin, support_leg_end, radius=84, fill=(18, 24, 28), width=8)
    draw.text((325, 720), f"{angle_deg:.1f}°", font=angle_font, fill=(18, 24, 28))
    draw.text((690, 1070), f"{angle_deg:.1f}°", font=value_font, fill=(18, 32, 38))

    return image.convert("RGB")


@st.cache_data(show_spinner=False)
def generate_knee_lift_card_image(
    pitch_id: str,
    feature_path_str: str,
    template_path_str: str | None = None,
) -> bytes | None:
    feature_path = Path(feature_path_str)
    template_path = Path(template_path_str) if template_path_str else KNEE_LIFT_TEMPLATE_IMAGE

    angle_deg = get_lead_leg_lift_angle_value(feature_path)
    if angle_deg is None:
        return None

    image = render_knee_lift_card(
        angle_deg=angle_deg,
        template_path=template_path,
    )

    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def show_knee_lift_visual_card(
    pitch_id: str,
    feature_path: Path,
    template_path: Path = KNEE_LIFT_TEMPLATE_IMAGE,
) -> None:
    st.subheader("최대 리드 레그 리프트")
    st.caption("키킹 동작에서 앞다리를 들어 올린 정도를 2D 각도로 표시합니다.")

    image_bytes = generate_knee_lift_card_image(
        pitch_id=pitch_id,
        feature_path_str=str(feature_path),
        template_path_str=str(template_path),
    )

    if image_bytes is None:
        st.info("Lead-Leg Lift 각도 값을 찾지 못했습니다.")
        return

    left_spacer, center_col, right_spacer = st.columns([2.35, 2.3, 2.35])
    with center_col:
        st.image(image_bytes, width=CARD_DISPLAY_WIDTH)


def describe_stride_knee_angle(angle_deg: float) -> str:
    if angle_deg >= 165:
        return "착지 순간 앞무릎이 매우 펴진 상태"
    elif angle_deg >= 150:
        return "착지 순간 앞무릎이 비교적 펴진 상태"
    elif angle_deg >= 135:
        return "착지 순간 앞무릎이 약간 굽혀진 상태"
    else:
        return "착지 순간 앞무릎이 비교적 많이 굽혀진 상태"

def _draw_dot(draw, xy, r=14, fill=(18, 32, 38), outline=(255, 255, 255)):
    x, y = xy
    draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline, width=2)

def _draw_line(draw, p1, p2, width=18, fill=(20, 20, 20)):
    draw.line([p1, p2], fill=fill, width=width)

def _draw_leader(draw, p1, p2, width=2, fill=(70, 70, 70)):
    draw.line([p1, p2], fill=fill, width=width)

def _draw_angle_arc(draw, knee, hip, ankle, radius=60, fill=(0, 0, 0), width=8):
    kx, ky = knee
    angle1 = math.degrees(math.atan2(hip[1] - ky, hip[0] - kx))
    angle2 = math.degrees(math.atan2(ankle[1] - ky, ankle[0] - kx))
    a1 = angle1 % 360
    a2 = angle2 % 360
    diff = (a2 - a1) % 360
    if diff > 180:
        start = a2
        end = a1
    else:
        start = a1
        end = a2
    bbox = (kx-radius, ky-radius, kx+radius, ky+radius)
    draw.arc(bbox, start=start, end=end, fill=fill, width=width)

def render_stride_knee_card(
    angle_deg: float,
    template_path: Path = STRIDE_KNEE_TEMPLATE_IMAGE,
):
    if not template_path.exists():
        raise FileNotFoundError(f"템플릿 이미지를 찾지 못했습니다: {template_path}")

    base = Image.open(template_path).convert("RGBA")
    w, h = base.size

    canvas = Image.new("RGBA", (w, h), (245, 245, 245, 255))
    canvas.alpha_composite(base)
    draw = ImageDraw.Draw(canvas)

    # 사용자가 지정한 기준점: pitcher_FC_stride_knee = (487, 775)
    # 이 좌표는 스트라이드 엉덩이(hip) 기준입니다.
    hip = (487, 775)
    knee = (597, 945)
    ankle = (647, 1200)

    _draw_line(draw, hip, knee, width=20, fill=(18, 24, 28))
    _draw_line(draw, knee, ankle, width=20, fill=(18, 24, 28))

    for pt in [hip, knee, ankle]:
        _draw_dot(draw, pt, r=14, fill=(18, 24, 28), outline=(240, 240, 240))

    _draw_angle_arc(draw, knee, hip, ankle, radius=80, fill=(18, 24, 28), width=8)

    subtitle_font = _safe_font(24, bold=False)
    label_font = _safe_font(20, bold=True)
    label_sub_font = _safe_font(17, bold=False)
    value_font = _safe_font(72, bold=True)
    angle_font = _safe_font(28, bold=True)

    draw.text((210, 90), "Stride Knee Angle @ Front Foot Contact", font=subtitle_font, fill=(111, 126, 130))
    draw.line((470, 130, 620, 130), fill=(122, 198, 185), width=3)

    _draw_leader(draw, hip, (610, 765))
    draw.text((620, 740), "스트라이드 엉덩이", font=label_font, fill=(25, 40, 45))
    draw.text((620, 772), "Stride Hip", font=label_sub_font, fill=(105, 115, 120))

    _draw_leader(draw, knee, (715, 965))
    draw.text((725, 940), "스트라이드 무릎", font=label_font, fill=(25, 40, 45))
    draw.text((725, 972), "Stride Knee", font=label_sub_font, fill=(105, 115, 120))

    _draw_leader(draw, ankle, (720, 1235))
    draw.text((730, 1210), "스트라이드 발목", font=label_font, fill=(25, 40, 45))
    draw.text((730, 1242), "Stride Ankle", font=label_sub_font, fill=(105, 115, 120))

    draw.text((645, 1070), f"{angle_deg:.1f}°", font=angle_font, fill=(18, 24, 28))
    draw.text((760, 1070), f"{angle_deg:.1f}°", font=value_font, fill=(18, 32, 38))

    return canvas.convert("RGB")

@st.cache_data(show_spinner=False)
def generate_stride_knee_card_image(
    pitch_id: str,
    feature_path_str: str,
    template_path_str: str | None = None,
) -> bytes | None:
    feature_path = Path(feature_path_str)
    template_path = Path(template_path_str) if template_path_str else STRIDE_KNEE_TEMPLATE_IMAGE

    angle_deg = get_stride_knee_angle_fc_value(feature_path)
    if angle_deg is None:
        return None

    image = render_stride_knee_card(
        angle_deg=angle_deg,
        template_path=template_path,
    )

    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def show_stride_knee_visual_card(
    pitch_id: str,
    feature_path: Path,
    template_path: Path = STRIDE_KNEE_TEMPLATE_IMAGE,
) -> None:
    st.subheader("자세 시각화 카드")
    st.caption(
        "FC(앞발 착지) 시점의 스트라이드 무릎 각도를 "
        "시각화 카드로 표시합니다."
    )

    image_bytes = generate_stride_knee_card_image(
        pitch_id=pitch_id,
        feature_path_str=str(feature_path),
        template_path_str=str(template_path),
    )

    if image_bytes is None:
        st.info("스트라이드 무릎 각도 값을 찾지 못했습니다.")
        return

    left_spacer, center_col, right_spacer = st.columns([2, 3, 2])

    with center_col:
        st.image(
            image_bytes,
            width=CARD_DISPLAY_WIDTH,
        )


def render_card_section_title(title: str) -> None:
    st.markdown(
        f"""
        <div style="font-size:1.02rem;font-weight:800;color:#fafafa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:0.45rem;line-height:1.2;">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def show_visual_cards_row(
    pitch_id: str,
    feature_path: Path,
) -> None:
    st.subheader("자세 시각화 카드")
    st.caption("핵심 동작 지표와 자동 Arm Slot 분류를 카드 형태로 보여줍니다.")

    knee_img = generate_knee_lift_card_image(
        pitch_id=pitch_id,
        feature_path_str=str(feature_path),
        template_path_str=str(KNEE_LIFT_TEMPLATE_IMAGE),
    )
    stride_img = generate_stride_knee_card_image(
        pitch_id=pitch_id,
        feature_path_str=str(feature_path),
        template_path_str=str(STRIDE_KNEE_TEMPLATE_IMAGE),
    )
    arm_slot_info = get_pitcher_arm_slot_info(
        pitch_id,
        feature_path,
    )

    cols = st.columns(3)

    with cols[0]:
        render_card_section_title("최대 리드 레그 리프트")
        if knee_img is not None:
            st.image(knee_img, use_container_width=True)
            lift_angle = get_lead_leg_lift_angle_value(feature_path)
            if lift_angle is not None:
                st.caption(
                    f"리드 레그 리프트 각도: {lift_angle:.1f}° · "
                    f"{describe_lead_leg_lift_angle(lift_angle)}"
                )
        else:
            st.info("Lead-Leg Lift 각도 값을 찾지 못했습니다.")

    with cols[1]:
        render_card_section_title("FC 시 스트라이드 무릎 각도")
        if stride_img is not None:
            st.image(stride_img, use_container_width=True)
            stride_angle = get_stride_knee_angle_fc_value(feature_path)
            if stride_angle is not None:
                st.caption(
                    f"앞발 착지 시점 스트라이드 무릎 각도: {stride_angle:.1f}° · "
                    f"{describe_stride_knee_angle(stride_angle)}"
                )
        else:
            st.info("스트라이드 무릎 각도 값을 찾지 못했습니다.")

    with cols[2]:
        render_card_section_title("투구 팔 슬롯")

        if arm_slot_info is None:
            st.info("자동 Arm Slot 분류 결과를 찾지 못했습니다.")
        else:
            image_bytes = generate_arm_slot_card_image(
                pitch_id=pitch_id,
            )

            if image_bytes is not None:
                st.image(image_bytes, use_container_width=True)
            else:
                st.info("Arm Slot 카드를 생성하지 못했습니다.")

            label = arm_slot_info["label"]
            confidence = arm_slot_info.get("confidence")

            if confidence:
                st.caption(
                    f"Arm Slot: {label} · 신뢰도: {fmt_confidence(confidence)}"
                )
            else:
                st.caption(f"Arm Slot: {label}")



def lead_knee_angle_change(pitch_id: str, feature_path: Path) -> float | None:
    """Projected stride-knee extension from FC to the release candidate.

    Suppress the comparison if either event's lead knee was flagged invalid.
    The result is a 2D change, not a measured 3D joint rotation.
    """
    row = load_feature_row(feature_path)
    fc_angle = first_value(row, ["stride_knee_angle_front_foot_contact_deg"])
    release_angle = first_value(row, ["stride_knee_angle_release_deg"])
    fc = _safe_int(first_value(row, ["front_foot_contact_frame", "fc_frame"]))
    release = _safe_int(first_value(row, ["release_candidate_frame", "release_frame"]))
    side = str(first_value(row, ["stride_side"]) or "").lower()
    motion_path = find_normalized_csv(pitch_id)
    if (fc is None or release is None or release <= fc or side not in ("left", "right")
            or motion_path is None):
        return None
    try:
        change = float(release_angle) - float(fc_angle)
        if not np.isfinite(change):
            return None
        motion = pd.read_csv(motion_path)
        invalid_col = f"{side}_knee_invalid"
        if invalid_col not in motion.columns:
            return None
        for frame in (fc, release):
            matches = motion.loc[pd.to_numeric(motion["frame"], errors="coerce") == frame]
            if matches.empty or str(matches.iloc[0][invalid_col]).strip().lower() in ("true", "1", "1.0"):
                return None
        return change
    except (OSError, TypeError, ValueError, KeyError, pd.errors.EmptyDataError):
        return None


def show_coaching_insights(
    pitch_id: str,
    feature_path: Path,
) -> None:
    """
    단일 카메라 2D 분석 결과를 '교정 판정'이 아니라
    반복 투구에서 확인할 포인트로 정리한다.
    """
    row = load_feature_row(feature_path)
    if not row:
        return

    arm_info = get_pitcher_arm_slot_info(
        pitch_id,
        feature_path,
    )

    fc_release = first_value(
        row,
        [
            "fc_to_release_time_s",
            "fc_to_release_time",
            "fc_release_time_sec",
            "fc_to_release_sec",
        ],
    )

    knee_change = lead_knee_angle_change(pitch_id, feature_path)

    lift_angle = first_value(
        row,
        [
            "lead_leg_lift_angle_2d_deg",
            "lead_leg_lift_angle_deg",
            "lead_leg_lift_deg",
            "lead_leg_lift_angle",
        ],
    )

    insights = []

    if arm_info is not None:
        slot_label = arm_info["label"]
        confidence = arm_info.get("confidence")
        value = slot_label
        if confidence:
            value += f" · {fmt_confidence(confidence)}"

        insights.append(
            (
                "Arm Slot",
                value,
                "반복 투구에서 같은 Arm Slot 분류가 유지되는지 확인해 "
                "팔 경로의 일관성을 살펴볼 수 있습니다.",
            )
        )

    try:
        if fc_release is not None and not pd.isna(fc_release):
            value = float(fc_release)
            insights.append(
                (
                    "FC → Release Timing",
                    f"{value:.3f} s",
                    "앞발 착지 이후 릴리스까지 걸리는 시간입니다. "
                    "동일 투수의 여러 투구에서 이 구간의 편차를 비교해 "
                    "전달 타이밍의 일관성을 확인할 수 있습니다.",
                )
            )
    except (TypeError, ValueError):
        pass

    if knee_change is not None:
        insights.append((
            "FC → Release 앞무릎 변화",
            f"{knee_change:+.1f}°",
            "착지부터 릴리스 후보까지의 2D 무릎 각도 변화입니다. "
            "+는 영상에서 더 펴짐, −는 더 굽힘을 뜻합니다. "
            "촬영 각도와 릴리스 후보 시점에 영향을 받습니다.",
        ))

    try:
        if lift_angle is not None and not pd.isna(lift_angle):
            value = float(lift_angle)
            insights.append(
                (
                    "Lead-Leg Lift",
                    f"{value:.1f}°",
                    "키킹 동작에서 앞다리를 들어 올린 정도입니다. "
                    "다른 투구와 비교해 리프트 높이와 동작 패턴의 반복성을 확인할 수 있습니다.",
                )
            )
    except (TypeError, ValueError):
        pass

    if not insights:
        return

    st.subheader("코칭 인사이트")
    st.caption(
        "현재 분석값을 바탕으로 반복 투구에서 확인할 포인트를 정리합니다. "
        "단일 카메라 2D 분석이므로 특정 자세를 좋거나 나쁘다고 판정하지 않습니다."
    )

    # 화면이 너무 길어지지 않도록 최대 4개
    insights = insights[:4]
    cols = st.columns(len(insights))

    for col, (title, value, note) in zip(cols, insights):
        with col:
            st.markdown(
                f"""
                <div style="
                    min-height:220px;
                    border:1px solid color-mix(in srgb, var(--text-color) 20%, transparent);
                    border-radius:14px;
                    padding:18px 16px;
                    background:var(--secondary-background-color);
                    color:var(--text-color);
                ">
                    <div style="
                        color:color-mix(in srgb, var(--text-color) 66%, transparent);
                        font-size:0.82rem;
                        margin-bottom:0.45rem;
                    ">{title}</div>
                    <div style="
                        color:var(--text-color);
                        font-size:1.38rem;
                        font-weight:800;
                        margin-bottom:0.8rem;
                    ">{value}</div>
                    <div style="
                        color:color-mix(in srgb, var(--text-color) 82%, transparent);
                        font-size:0.88rem;
                        line-height:1.55;
                    ">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )




# ============================================================
# Interactive 3D FC Coaching View
# ============================================================

COACHING_SKELETON_CONNECTIONS = [
    ("Head", "Neck"), ("Neck", "LShoulder"), ("Neck", "RShoulder"),
    ("LShoulder", "RShoulder"), ("Neck", "Hip"),
    ("LShoulder", "LHip"), ("RShoulder", "RHip"), ("LHip", "RHip"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"), ("LHeel", "LBigToe"),
    ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"), ("RHeel", "RBigToe"),
]

COACHING_JOINT_ALIASES = {
    "Head": ["Head"], "Neck": ["Neck"], "Hip": ["Hip", "MidHip", "Pelvis"],
    "LShoulder": ["LShoulder", "LeftShoulder"],
    "RShoulder": ["RShoulder", "RightShoulder"],
    "LElbow": ["LElbow", "LeftElbow"], "RElbow": ["RElbow", "RightElbow"],
    "LWrist": ["LWrist", "LeftWrist"], "RWrist": ["RWrist", "RightWrist"],
    "LHip": ["LHip", "LeftHip"], "RHip": ["RHip", "RightHip"],
    "LKnee": ["LKnee", "LeftKnee"], "RKnee": ["RKnee", "RightKnee"],
    "LAnkle": ["LAnkle", "LeftAnkle"], "RAnkle": ["RAnkle", "RightAnkle"],
    "LHeel": ["LHeel", "LeftHeel"], "RHeel": ["RHeel", "RightHeel"],
    "LBigToe": ["LBigToe", "LeftBigToe"], "RBigToe": ["RBigToe", "RightBigToe"],
}


def _normalize_joint_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _find_trc_for_pitch(pitch_id: str) -> Path | None:
    roots = [ROOT / "sports2d_results", ROOT / "sports2d"]
    patterns = [
        f"**/{pitch_id}/**/*_m_person00.trc",
        f"**/{pitch_id}/**/*_m_*.trc",
        f"**/{pitch_id}/**/*.trc",
        f"**/*{pitch_id}*_m_person00.trc",
        f"**/*{pitch_id}*_m_*.trc",
    ]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(root.glob(pattern))
        if found:
            break
    if not found:
        return None
    return sorted(set(found), key=lambda p: (0 if "_m_" in p.name else 1, len(str(p))))[0]


@st.cache_data(show_spinner=False)
def _read_trc_for_coaching(path_str: str):
    path = Path(path_str)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = next((i for i, line in enumerate(lines) if "Frame#" in line and "Time" in line), None)
    if header_idx is None or header_idx + 2 >= len(lines):
        raise ValueError("TRC header를 찾지 못했습니다.")

    def split_line(line: str):
        return [x.strip() for x in line.rstrip("\n").split("\t")] if "\t" in line else line.strip().split()

    marker_line = split_line(lines[header_idx])
    marker_names = [token.strip() for token in marker_line[2:] if token.strip()]
    rows = []
    for line in lines[header_idx + 2:]:
        if not line.strip():
            continue
        parts = split_line(line)
        if len(parts) < 2 + len(marker_names) * 3:
            continue
        try:
            frame = int(float(parts[0])); time = float(parts[1])
        except Exception:
            continue
        row = {"frame": frame, "time": time}
        idx = 2
        for marker in marker_names:
            xyz = parts[idx:idx+3]; idx += 3
            for axis, value in zip("xyz", xyz):
                try: row[f"{marker}_{axis}"] = float(value) if value != "" else np.nan
                except Exception: row[f"{marker}_{axis}"] = np.nan
        rows.append(row)
    if not rows:
        raise ValueError("TRC 데이터 row를 읽지 못했습니다.")
    return pd.DataFrame(rows), marker_names


def _resolve_coaching_joint_map(marker_names):
    normalized = {_normalize_joint_name(name): name for name in marker_names}
    result = {}
    for canonical, aliases in COACHING_JOINT_ALIASES.items():
        found = None
        for alias in aliases:
            if _normalize_joint_name(alias) in normalized:
                found = normalized[_normalize_joint_name(alias)]; break
        if found is None:
            for marker in marker_names:
                nm = _normalize_joint_name(marker)
                if any(nm.endswith(_normalize_joint_name(alias)) for alias in aliases):
                    found = marker; break
        if found is not None:
            result[canonical] = found
    return result


def _trc_xyz(row, marker):
    try:
        vals = tuple(float(row[f"{marker}_{a}"]) for a in "xyz")
    except Exception:
        return None
    return vals if all(np.isfinite(v) for v in vals) else None


def _estimate_trc_floor_y(df, marker_names):
    vals = []
    for marker in marker_names:
        nm = _normalize_joint_name(marker)
        if not any(token in nm for token in ("ankle", "heel", "toe")):
            continue
        col = f"{marker}_y"
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            if len(arr): vals.extend(arr.tolist())
    return float(np.nanpercentile(vals, 2.0)) if vals else 0.0


def _to_3d_display(xyz, floor_y):
    x, y, z = xyz
    return float(x), float(z), float(y - floor_y)


def _nearest_trc_row(df, frame_number):
    exact = df.loc[df["frame"] == int(frame_number)]
    if not exact.empty:
        return exact.iloc[0]
    idx = (pd.to_numeric(df["frame"], errors="coerce") - int(frame_number)).abs().idxmin()
    return df.loc[idx]


def _build_fc_3d_coaching_figure(pitch_id: str, feature_path: Path):
    trc_path = _find_trc_for_pitch(pitch_id)
    if trc_path is None:
        return None
    try:
        df, marker_names = _read_trc_for_coaching(str(trc_path))
    except Exception:
        return None

    row = load_feature_row(feature_path)
    fc_frame = first_value(row, ["front_foot_contact_frame", "fc_frame", "foot_contact_frame"])
    knee_frame = first_value(row, ["knee_lift_frame", "maximum_knee_lift_frame", "lead_leg_lift_frame"])
    throwing_side = str(first_value(row, ["throwing_side"]) or "").strip().lower()
    try: fc_frame = int(round(float(fc_frame)))
    except Exception: return None
    try: knee_frame = int(round(float(knee_frame)))
    except Exception: knee_frame = max(int(df["frame"].min()), fc_frame - 20)

    joint_map = _resolve_coaching_joint_map(marker_names)
    floor_y = _estimate_trc_floor_y(df, marker_names)
    fc_row = _nearest_trc_row(df, fc_frame)
    knee_row = _nearest_trc_row(df, knee_frame)

    fc_points, knee_points = {}, {}
    for canonical, marker in joint_map.items():
        xyz = _trc_xyz(fc_row, marker)
        if xyz is not None: fc_points[canonical] = _to_3d_display(xyz, floor_y)
        xyz = _trc_xyz(knee_row, marker)
        if xyz is not None: knee_points[canonical] = _to_3d_display(xyz, floor_y)

    traces = []
    for a, b in COACHING_SKELETON_CONNECTIONS:
        if a in fc_points and b in fc_points:
            p1, p2 = fc_points[a], fc_points[b]
            traces.append(go.Scatter3d(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], mode="lines", line=dict(width=4,color="#374151"), hoverinfo="skip", showlegend=False))

    if fc_points:
        traces.append(go.Scatter3d(x=[p[0] for p in fc_points.values()], y=[p[1] for p in fc_points.values()], z=[p[2] for p in fc_points.values()], mode="markers", marker=dict(size=4,color="#374151"), text=list(fc_points.keys()), hovertemplate="%{text}<extra></extra>", showlegend=False))

    stride_prefix = "R" if throwing_side == "left" else "L"

    def add_guide(p1, p2, name, color, width=9):
        if p1 is None or p2 is None: return
        traces.append(go.Scatter3d(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], mode="lines+markers", line=dict(width=width,color=color), marker=dict(size=7,color=color), name=name, hovertemplate=name+"<extra></extra>"))

    add_guide(
        fc_points.get("LShoulder"),
        fc_points.get("RShoulder"),
        "Shoulder Line",
        "#00AEEF",
        12,
    )
    add_guide(
        fc_points.get("LHip"),
        fc_points.get("RHip"),
        "Hip Line",
        "#D946EF",
        12,
    )
    add_guide(
        fc_points.get(f"{stride_prefix}Heel"),
        fc_points.get(f"{stride_prefix}BigToe"),
        "Front Foot",
        "#FACC15",
        12,
    )

    pts = list(fc_points.values()) or [(-1,-1,0),(1,1,2)]
    xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    x0,x1=min(xs),max(xs); y0,y1=min(ys),max(ys); z0,z1=min(zs),max(zs)
    xp=max(.3,(x1-x0)*.35); yp=max(.3,(y1-y0)*.35)
    xx=np.linspace(x0-xp,x1+xp,2); yy=np.linspace(y0-yp,y1+yp,2); X,Y=np.meshgrid(xx,yy); Z=np.zeros_like(X)
    traces.insert(0, go.Surface(x=X,y=Y,z=Z,showscale=False,opacity=.10,colorscale=[[0,"#d8dde3"],[1,"#d8dde3"]],hoverinfo="skip",name="Ground"))

    fig=go.Figure(data=traces)
    fig.update_layout(title=f"FC 3D Coaching View · Frame {fc_frame}", margin=dict(l=0,r=0,t=50,b=0), legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0), scene=dict(xaxis=dict(title="X",range=[x0-.4,x1+.4],showspikes=False), yaxis=dict(title="Depth",range=[y0-.4,y1+.4],showspikes=False), zaxis=dict(title="Height",range=[min(0,z0-.15),z1+.35],showspikes=False), aspectmode="data", dragmode="orbit", camera=dict(eye=dict(x=1.8,y=1.7,z=1.15),up=dict(x=0,y=0,z=1),projection=dict(type="orthographic"))), uirevision=f"fc-coaching-{pitch_id}")
    return fig

def _safe_int(value, default=None):
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _collect_trc_points_for_frame(df, joint_map, frame_number, floor_y):
    row = _nearest_trc_row(df, frame_number)
    points = {}
    for canonical, marker in joint_map.items():
        xyz = _trc_xyz(row, marker)
        if xyz is not None:
            points[canonical] = _to_3d_display(xyz, floor_y)
    return points


def _infer_stride_prefix_from_motion(
    knee_points: dict,
    fc_points: dict,
    throwing_side: str,
):
    """
    Infer the actual stride/front foot from Knee Lift -> FC motion.

    Default baseball convention:
      left-handed pitcher  -> right stride foot
      right-handed pitcher -> left stride foot

    Some single-camera Sports2D/TRC outputs can occasionally swap left/right
    labels for a specific clip. To keep the coaching guide on the foot that
    actually strides, compare ankle displacement from Knee Lift to FC.

    The expected side is retained unless the opposite ankle shows clearly
    larger motion.
    """
    expected = "R" if throwing_side == "left" else "L"
    other = "L" if expected == "R" else "R"

    def motion(prefix):
        p0 = knee_points.get(f"{prefix}Ankle")
        p1 = fc_points.get(f"{prefix}Ankle")

        if p0 is None or p1 is None:
            return None

        try:
            return float(
                np.sqrt(
                    (p1[0] - p0[0]) ** 2
                    + (p1[1] - p0[1]) ** 2
                    + (p1[2] - p0[2]) ** 2
                )
            )
        except Exception:
            return None

    expected_motion = motion(expected)
    other_motion = motion(other)

    if expected_motion is None and other_motion is None:
        return expected, "throwing_side_default"

    if expected_motion is None:
        return other, "ankle_motion_fallback"

    if other_motion is None:
        return expected, "throwing_side_default"

    if (
        other_motion > expected_motion * 1.35
        and (other_motion - expected_motion) > 0.05
    ):
        return other, "ankle_motion_override"

    return expected, "throwing_side_default"


def _pitch_direction_from_hip_motion(start_points: dict, end_points: dict):
    """Estimate horizontal travel direction from pelvis midpoint motion."""
    def _mid(points):
        l = points.get("LHip")
        r = points.get("RHip")
        if l is None or r is None:
            return None
        return ((l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0)

    p0 = _mid(start_points)
    p1 = _mid(end_points)
    if p0 is None or p1 is None:
        return None

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-6:
        return None
    return dx / norm, dy / norm


def _release_anchored_lr_and_direction(
    df,
    joint_map,
    release_frame: int,
    floor_y: float,
    throwing_side: str,
    knee_points: dict,
):
    """
    Release-anchored GLOBAL left/right mapping.

    The pitch direction is estimated first from pelvis travel from Knee Lift
    to Release.  Then the raw TRC arm whose wrist is farther toward the
    catcher at Release is treated as the throwing arm.  Because the true
    throwing side is known from metadata, this gives one global raw->anatomical
    L/R mapping.  The SAME mapping is then used for both arms and legs.

    This is intentionally different from the previous version, which corrected
    arms and selected the FC foot independently.  That could produce a left arm
    with a right-foot label (or vice versa) when the whole TRC skeleton was
    mirrored/swapped for a clip.
    """
    frames = []
    for f in range(max(0, int(release_frame) - 3), int(release_frame) + 1):
        pts = _collect_trc_points_for_frame(df, joint_map, f, floor_y)
        if pts:
            frames.append((f, pts))

    release_points = (
        frames[-1][1]
        if frames
        else _collect_trc_points_for_frame(df, joint_map, release_frame, floor_y)
    )

    # 1) Pitch direction from body/pelvis travel.
    pitch_dir = _pitch_direction_from_hip_motion(
        knee_points,
        release_points,
    )
    direction_source = "pelvis_knee_to_release"

    # Fallback: shoulder midpoint travel over Release-3F..Release.
    if pitch_dir is None and len(frames) >= 2:
        def shoulder_mid(pts):
            l = pts.get("LShoulder")
            r = pts.get("RShoulder")
            if l is None or r is None:
                return None
            return (
                (l[0] + r[0]) / 2.0,
                (l[1] + r[1]) / 2.0,
                (l[2] + r[2]) / 2.0,
            )

        a = shoulder_mid(frames[0][1])
        b = shoulder_mid(frames[-1][1])
        if a is not None and b is not None:
            dx, dy = b[0] - a[0], b[1] - a[1]
            norm = float(np.hypot(dx, dy))
            if norm > 1e-6:
                pitch_dir = (dx / norm, dy / norm)
                direction_source = "shoulder_mid_release_window"

    # 2) Identify the raw throwing arm at/near Release.
    #    Primary rule: wrist farther toward catcher.
    #    Secondary rule: larger Release-window wrist path length.
    def _wrist_series(prefix):
        vals = []
        for f, pts in frames:
            pt = pts.get(f"{prefix}Wrist")
            if pt is not None:
                vals.append((f, pt))
        return vals

    def _path_length(series):
        if len(series) < 2:
            return np.nan
        total = 0.0
        for (_, a), (_, b) in zip(series[:-1], series[1:]):
            total += float(np.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3))))
        return total

    l_series = _wrist_series("L")
    r_series = _wrist_series("R")
    l_motion = _path_length(l_series)
    r_motion = _path_length(r_series)

    raw_throwing = None
    arm_source = None

    if pitch_dir is not None:
        ux, uy = pitch_dir
        lw = release_points.get("LWrist")
        rw = release_points.get("RWrist")
        if lw is not None and rw is not None:
            l_proj = lw[0] * ux + lw[1] * uy
            r_proj = rw[0] * ux + rw[1] * uy
            raw_throwing = "L" if l_proj >= r_proj else "R"
            arm_source = "release_wrist_forward_projection"

    if raw_throwing is None:
        if np.isfinite(l_motion) and np.isfinite(r_motion):
            raw_throwing = "L" if l_motion >= r_motion else "R"
            arm_source = "release_wrist_motion_fallback"
        else:
            raw_throwing = "L" if throwing_side == "left" else "R"
            arm_source = "handedness_fallback"

    anatomical_throwing = "L" if throwing_side == "left" else "R"
    anatomical_glove = "R" if anatomical_throwing == "L" else "L"
    raw_glove = "R" if raw_throwing == "L" else "L"

    raw_for_anatomical = {
        anatomical_throwing: raw_throwing,
        anatomical_glove: raw_glove,
    }

    global_lr_swapped = (raw_throwing != anatomical_throwing)

    return {
        "release_points": release_points,
        "pitch_dir": pitch_dir,
        "direction_source": direction_source,
        "raw_throwing_prefix": raw_throwing,
        "raw_for_anatomical": raw_for_anatomical,
        "global_lr_swapped": global_lr_swapped,
        "left_wrist_motion": l_motion,
        "right_wrist_motion": r_motion,
        "arm_source": arm_source,
    }

def _foot_center(points: dict, prefix: str):
    """Return horizontal center of available ankle/heel/toe points."""
    vals = []
    for joint in ("Ankle", "Heel", "BigToe", "SmallToe"):
        pt = points.get(f"{prefix}{joint}")
        if pt is not None:
            vals.append(pt)
    if not vals:
        return None
    return (
        float(np.mean([p[0] for p in vals])),
        float(np.mean([p[1] for p in vals])),
        float(np.mean([p[2] for p in vals])),
    )


def _select_front_foot_at_fc(fc_points: dict, pitch_dir, throwing_side: str):
    """
    Select the physical foot that is farther toward the catcher at FC.

    This does not trust TRC L/R foot labels. The anatomical stride side is
    reported from throwing handedness (LHP -> right foot, RHP -> left foot).
    """
    expected_anatomical = "R" if throwing_side == "left" else "L"

    if pitch_dir is None:
        raw_prefix = expected_anatomical
        return raw_prefix, expected_anatomical, "handedness_fallback"

    ux, uy = pitch_dir
    candidates = {}
    for prefix in ("L", "R"):
        c = _foot_center(fc_points, prefix)
        if c is not None:
            candidates[prefix] = c[0] * ux + c[1] * uy

    if not candidates:
        return expected_anatomical, expected_anatomical, "handedness_fallback"

    raw_prefix = max(candidates, key=candidates.get)
    return raw_prefix, expected_anatomical, "fc_forward_projection"



def _side_bundle_center(points: dict, prefix: str, joints):
    vals = []
    for joint in joints:
        pt = points.get(f"{prefix}{joint}")
        if pt is not None:
            vals.append(pt)

    if not vals:
        return None

    return np.array(
        [
            float(np.mean([p[0] for p in vals])),
            float(np.mean([p[1] for p in vals])),
            float(np.mean([p[2] for p in vals])),
        ],
        dtype=float,
    )


def _identify_stride_raw_at_knee(knee_points: dict, throwing_side: str):
    """
    Knee Lift anchor:
    identify which raw TRC leg is the lifted stride leg.

    LHP -> anatomical R leg is the stride leg.
    RHP -> anatomical L leg is the stride leg.

    Raw L/R labels are NOT trusted here. The leg whose knee/ankle are higher
    at Maximum Knee Lift is treated as the lifted stride leg.
    """
    expected_anatomical = "R" if throwing_side == "left" else "L"

    def lift_score(prefix):
        hip = knee_points.get(f"{prefix}Hip")
        knee = knee_points.get(f"{prefix}Knee")
        ankle = knee_points.get(f"{prefix}Ankle")

        values = []

        if knee is not None:
            if hip is not None:
                values.append(float(knee[2] - hip[2]))
            else:
                values.append(float(knee[2]))

        if ankle is not None:
            if hip is not None:
                values.append(float(ankle[2] - hip[2]))
            else:
                values.append(float(ankle[2]))

        if not values:
            return np.nan

        return float(np.mean(values))

    l_score = lift_score("L")
    r_score = lift_score("R")

    if np.isfinite(l_score) and np.isfinite(r_score):
        raw_stride = "L" if l_score >= r_score else "R"
        source = "knee_lift_height_anchor"
    else:
        raw_stride = expected_anatomical
        source = "handedness_fallback"

    raw_pivot = "R" if raw_stride == "L" else "L"
    anatomical_pivot = "L" if expected_anatomical == "R" else "R"

    return {
        expected_anatomical: raw_stride,
        anatomical_pivot: raw_pivot,
    }, source, l_score, r_score


def _identify_throwing_raw_at_release(
    df,
    joint_map,
    release_frame: int,
    floor_y: float,
    throwing_side: str,
):
    """
    Release anchor:
    identify the raw TRC throwing arm from Release-3F..Release.

    Uses elbow+wrist path length plus shoulder->wrist extension.
    The raw label is then mapped to the known anatomical throwing side.
    """
    release_points = _collect_trc_points_for_frame(
        df, joint_map, release_frame, floor_y
    )

    def path_length(prefix):
        series = []
        for f in range(max(0, int(release_frame) - 3), int(release_frame) + 1):
            pts = _collect_trc_points_for_frame(df, joint_map, f, floor_y)
            e = pts.get(f"{prefix}Elbow")
            w = pts.get(f"{prefix}Wrist")

            if e is not None and w is not None:
                series.append(
                    np.array(
                        [
                            (e[0] + w[0]) / 2.0,
                            (e[1] + w[1]) / 2.0,
                            (e[2] + w[2]) / 2.0,
                        ],
                        dtype=float,
                    )
                )

        if len(series) < 2:
            return np.nan

        return float(
            sum(
                np.linalg.norm(b - a)
                for a, b in zip(series[:-1], series[1:])
            )
        )

    def extension(prefix):
        s = release_points.get(f"{prefix}Shoulder")
        w = release_points.get(f"{prefix}Wrist")

        if s is None or w is None:
            return np.nan

        return float(
            np.linalg.norm(
                np.array(w, dtype=float)
                - np.array(s, dtype=float)
            )
        )

    scores = {}

    for prefix in ("L", "R"):
        motion = path_length(prefix)
        ext = extension(prefix)

        motion_part = motion if np.isfinite(motion) else 0.0
        ext_part = ext if np.isfinite(ext) else 0.0

        scores[prefix] = motion_part + 0.35 * ext_part

    if max(scores.values()) > 0:
        raw_throwing = max(scores, key=scores.get)
        source = "release_arm_motion_extension_anchor"
    else:
        raw_throwing = "L" if throwing_side == "left" else "R"
        source = "handedness_fallback"

    anatomical_throwing = "L" if throwing_side == "left" else "R"
    anatomical_glove = "R" if anatomical_throwing == "L" else "L"
    raw_glove = "R" if raw_throwing == "L" else "L"

    return {
        anatomical_throwing: raw_throwing,
        anatomical_glove: raw_glove,
    }, source, scores


def _track_pair_identity(
    df,
    joint_map,
    floor_y: float,
    start_frame: int,
    end_frame: int,
    start_mapping: dict,
    joints,
):
    """
    Track anatomical L/R identities frame-by-frame.

    The raw Sports2D/TRC labels may swap at intermediate frames.
    At each frame we choose normal-vs-swapped raw assignment by continuity
    with the previously tracked anatomical L/R bundle positions.

    start_mapping:
        {"L": raw_prefix_at_anchor, "R": raw_prefix_at_anchor}
    """
    step = 1 if end_frame >= start_frame else -1
    mapping = dict(start_mapping)

    anchor_points = _collect_trc_points_for_frame(
        df, joint_map, start_frame, floor_y
    )

    previous = {
        anatomical: _side_bundle_center(
            anchor_points,
            raw_prefix,
            joints,
        )
        for anatomical, raw_prefix in mapping.items()
    }

    swap_count = 0
    evaluated = 0

    for frame_no in range(
        int(start_frame) + step,
        int(end_frame) + step,
        step,
    ):
        pts = _collect_trc_points_for_frame(
            df, joint_map, frame_no, floor_y
        )

        raw_l = _side_bundle_center(
            pts, "L", joints
        )
        raw_r = _side_bundle_center(
            pts, "R", joints
        )

        prev_l = previous.get("L")
        prev_r = previous.get("R")

        if (
            raw_l is None
            or raw_r is None
            or prev_l is None
            or prev_r is None
        ):
            # Keep current identity when a frame is incomplete.
            for anatomical in ("L", "R"):
                raw_prefix = mapping.get(anatomical)
                cur = _side_bundle_center(
                    pts,
                    raw_prefix,
                    joints,
                )
                if cur is not None:
                    previous[anatomical] = cur
            continue

        cost_normal = (
            float(np.linalg.norm(prev_l - raw_l))
            + float(np.linalg.norm(prev_r - raw_r))
        )

        cost_swapped = (
            float(np.linalg.norm(prev_l - raw_r))
            + float(np.linalg.norm(prev_r - raw_l))
        )

        evaluated += 1

        if cost_swapped + 1e-9 < cost_normal:
            mapping = {
                "L": "R",
                "R": "L",
            }
            previous = {
                "L": raw_r,
                "R": raw_l,
            }
            swap_count += 1
        else:
            mapping = {
                "L": "L",
                "R": "R",
            }
            previous = {
                "L": raw_l,
                "R": raw_r,
            }

    return mapping, {
        "evaluated_frames": evaluated,
        "swap_frames": swap_count,
    }


def _pitch_direction_consistent_with_front_foot(
    pitch_dir,
    fc_points: dict,
    front_raw_prefix: str,
):
    """
    Keep the catcher arrow pointing toward the tracked anatomical front foot.

    Pelvis travel gives the initial direction. If that direction points away
    from the tracked stride foot at FC, reverse it.
    """
    front = _foot_center(
        fc_points,
        front_raw_prefix,
    )

    lhip = fc_points.get("LHip")
    rhip = fc_points.get("RHip")

    if (
        front is None
        or lhip is None
        or rhip is None
    ):
        return pitch_dir, False

    pelvis = (
        (lhip[0] + rhip[0]) / 2.0,
        (lhip[1] + rhip[1]) / 2.0,
    )

    vx = front[0] - pelvis[0]
    vy = front[1] - pelvis[1]

    norm = float(np.hypot(vx, vy))
    if norm <= 1e-6:
        return pitch_dir, False

    foot_dir = (vx / norm, vy / norm)

    if pitch_dir is None:
        return foot_dir, False

    dot = (
        pitch_dir[0] * foot_dir[0]
        + pitch_dir[1] * foot_dir[1]
    )

    if dot < 0:
        return (-pitch_dir[0], -pitch_dir[1]), True

    return pitch_dir, False




def _track_single_limb_role(
    df,
    joint_map,
    floor_y: float,
    start_frame: int,
    end_frame: int,
    start_raw_prefix: str,
    joints,
):
    """
    Track ONE physical limb role across frames, independent of raw L/R labels.

    This is intentionally different from global L/R swapping:
    - stride leg is anchored at Maximum Knee Lift, then tracked forward to FC
    - throwing arm is anchored at Release, then tracked backward to FC

    At each frame, whichever raw L/R limb is spatially closest to the
    previously tracked physical limb is chosen.
    """
    step = 1 if end_frame >= start_frame else -1
    current_raw = str(start_raw_prefix)

    start_pts = _collect_trc_points_for_frame(
        df, joint_map, start_frame, floor_y
    )
    previous = _side_bundle_center(
        start_pts,
        current_raw,
        joints,
    )

    switches = 0
    evaluated = 0
    history = [(int(start_frame), current_raw)]

    for frame_no in range(
        int(start_frame) + step,
        int(end_frame) + step,
        step,
    ):
        pts = _collect_trc_points_for_frame(
            df, joint_map, frame_no, floor_y
        )

        candidates = {}
        for prefix in ("L", "R"):
            center = _side_bundle_center(
                pts,
                prefix,
                joints,
            )
            if center is not None:
                candidates[prefix] = center

        if not candidates:
            history.append((int(frame_no), current_raw))
            continue

        if previous is None:
            if current_raw in candidates:
                chosen = current_raw
            else:
                chosen = next(iter(candidates))
        else:
            distances = {
                prefix: float(
                    np.linalg.norm(center - previous)
                )
                for prefix, center in candidates.items()
            }

            chosen = min(
                distances,
                key=distances.get,
            )

            evaluated += 1

        if chosen != current_raw:
            switches += 1

        current_raw = chosen
        previous = candidates[chosen]
        history.append((int(frame_no), current_raw))

    return current_raw, {
        "switches": switches,
        "evaluated_frames": evaluated,
        "history": history,
    }


def _raw_stride_leg_at_knee_lift(
    knee_points: dict,
):
    """
    Pick the physically lifted leg at Maximum Knee Lift.
    Raw L/R names are treated only as detector labels.
    """
    def score(prefix):
        hip = knee_points.get(f"{prefix}Hip")
        knee = knee_points.get(f"{prefix}Knee")
        ankle = knee_points.get(f"{prefix}Ankle")

        vals = []

        if hip is not None and knee is not None:
            vals.append(float(knee[2] - hip[2]))
        elif knee is not None:
            vals.append(float(knee[2]))

        if hip is not None and ankle is not None:
            vals.append(float(ankle[2] - hip[2]))
        elif ankle is not None:
            vals.append(float(ankle[2]))

        if not vals:
            return -np.inf

        return float(np.mean(vals))

    scores = {
        "L": score("L"),
        "R": score("R"),
    }

    raw_prefix = max(
        scores,
        key=scores.get,
    )

    return raw_prefix, scores


def _pitch_direction_from_tracked_front_foot(
    fc_points: dict,
    front_raw_prefix: str,
):
    """
    Catcher direction is defined from pelvis midpoint toward the physically
    tracked stride/front foot at FC.
    """
    front = _foot_center(
        fc_points,
        front_raw_prefix,
    )

    lhip = fc_points.get("LHip")
    rhip = fc_points.get("RHip")

    if (
        front is None
        or lhip is None
        or rhip is None
    ):
        return None

    pelvis = (
        (lhip[0] + rhip[0]) / 2.0,
        (lhip[1] + rhip[1]) / 2.0,
    )

    dx = float(front[0] - pelvis[0])
    dy = float(front[1] - pelvis[1])

    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return None

    return (
        dx / norm,
        dy / norm,
    )




def _joint_xyz_at_frame(df, joint_map, floor_y, frame_no, raw_prefix, joint_name):
    pts = _collect_trc_points_for_frame(df, joint_map, frame_no, floor_y)
    return pts.get(f"{raw_prefix}{joint_name}")


def _limb_bundle_center_at_frame(
    df,
    joint_map,
    floor_y,
    frame_no,
    raw_prefix,
    joints,
):
    pts = _collect_trc_points_for_frame(df, joint_map, frame_no, floor_y)
    return _side_bundle_center(pts, raw_prefix, joints)


def _leg_forward_motion_scores(
    df,
    joint_map,
    floor_y: float,
    knee_frame: int,
    fc_frame: int,
):
    """
    Determine the PHYSICAL stride leg from its whole Knee-Lift -> FC trajectory.

    We do not trust raw L/R names. The stride leg should travel much farther
    horizontally than the pivot leg between Maximum Knee Lift and FC.
    """
    scores = {}

    for prefix in ("L", "R"):
        start = _limb_bundle_center_at_frame(
            df,
            joint_map,
            floor_y,
            knee_frame,
            prefix,
            ("Knee", "Ankle", "Heel", "BigToe", "SmallToe"),
        )
        end = _limb_bundle_center_at_frame(
            df,
            joint_map,
            floor_y,
            fc_frame,
            prefix,
            ("Knee", "Ankle", "Heel", "BigToe", "SmallToe"),
        )

        if start is None or end is None:
            scores[prefix] = {
                "horizontal_displacement": -np.inf,
                "vector": None,
            }
            continue

        vec = np.array(
            [
                float(end[0] - start[0]),
                float(end[1] - start[1]),
            ],
            dtype=float,
        )

        scores[prefix] = {
            "horizontal_displacement": float(np.linalg.norm(vec)),
            "vector": vec,
        }

    valid = {
        k: v
        for k, v in scores.items()
        if np.isfinite(v["horizontal_displacement"])
    }

    if valid:
        raw_stride = max(
            valid,
            key=lambda k: valid[k]["horizontal_displacement"],
        )
    else:
        raw_stride = "L"

    return raw_stride, scores


def _throwing_arm_motion_scores(
    df,
    joint_map,
    floor_y: float,
    release_frame: int,
    window: int = 6,
):
    """
    Determine the PHYSICAL throwing arm from wrist/elbow motion near Release.

    The throwing wrist normally has the larger trajectory speed immediately
    around ball release. Raw L/R names are used only as detector labels.
    """
    scores = {}

    for prefix in ("L", "R"):
        wrist_positions = []
        elbow_positions = []

        start_f = max(
            int(pd.to_numeric(df["frame"], errors="coerce").min()),
            int(release_frame) - int(window),
        )
        end_f = int(release_frame) + 1

        for frame_no in range(start_f, end_f):
            pts = _collect_trc_points_for_frame(
                df,
                joint_map,
                frame_no,
                floor_y,
            )

            w = pts.get(f"{prefix}Wrist")
            e = pts.get(f"{prefix}Elbow")

            if w is not None:
                wrist_positions.append(np.array(w, dtype=float))
            if e is not None:
                elbow_positions.append(np.array(e, dtype=float))

        def path_length(seq):
            if len(seq) < 2:
                return 0.0
            return float(
                sum(
                    np.linalg.norm(b - a)
                    for a, b in zip(seq[:-1], seq[1:])
                )
            )

        wrist_path = path_length(wrist_positions)
        elbow_path = path_length(elbow_positions)

        scores[prefix] = {
            "wrist_path": wrist_path,
            "elbow_path": elbow_path,
            "score": wrist_path + 0.35 * elbow_path,
        }

    raw_throwing = max(
        scores,
        key=lambda k: scores[k]["score"],
    )

    return raw_throwing, scores


def _fc_followthrough_cues(df, joint_map, floor_y, fc_frame, release_frame):
    """Optional image-height cues; no glove detection or real depth inference."""
    fc = _collect_trc_points_for_frame(df, joint_map, fc_frame, floor_y)
    end = min(int(release_frame), int(fc_frame) + 8)
    frames = [
        _collect_trc_points_for_frame(df, joint_map, f, floor_y)
        for f in range(int(fc_frame) + 1, end + 1)
    ]
    rise = {}
    for side in ("L", "R"):
        base = fc.get(f"{side}Ankle")
        later = [pts[f"{side}Ankle"][2] for pts in frames if f"{side}Ankle" in pts]
        rise[side] = max(later) - base[2] if base is not None and later else None
    lifted = None
    if all(v is not None for v in rise.values()) and abs(rise["L"] - rise["R"]) >= 0.04:
        lifted = max(rise, key=rise.get)
    wrists = {s: fc.get(f"{s}Wrist") for s in ("L", "R")}
    higher = None
    if all(v is not None for v in wrists.values()) and abs(wrists["L"][2] - wrists["R"][2]) >= 0.05:
        higher = max(wrists, key=lambda s: wrists[s][2])
    return {"lifted_raw_foot": lifted, "ankle_rise": rise, "higher_raw_wrist_fc": higher}


def _canonical_identity_from_full_motion(
    df,
    joint_map,
    floor_y: float,
    knee_frame: int,
    fc_frame: int,
    release_frame: int,
    throwing_side: str,
):
    """
    Build anatomical L/R identities from the FULL pitching motion.

    Legs:
      largest Knee-Lift -> FC horizontal travel = physical stride/front leg

    Arms:
      largest wrist/elbow path near Release = physical throwing arm

    Then assign anatomical labels from known handedness:
      LHP -> throwing arm L, stride leg R
      RHP -> throwing arm R, stride leg L

    A single global L/R mapping is used only when arm and stride cues agree.
    Independent arm/leg swaps would create anatomically impossible labels.
    """
    raw_stride, leg_scores = _leg_forward_motion_scores(
        df=df,
        joint_map=joint_map,
        floor_y=floor_y,
        knee_frame=knee_frame,
        fc_frame=fc_frame,
    )

    raw_throwing, arm_scores = _throwing_arm_motion_scores(
        df=df,
        joint_map=joint_map,
        floor_y=floor_y,
        release_frame=release_frame,
        window=6,
    )
    followthrough = _fc_followthrough_cues(df, joint_map, floor_y, fc_frame, release_frame)

    anatomical_throwing = "L" if throwing_side == "left" else "R"
    anatomical_glove = "R" if anatomical_throwing == "L" else "L"

    anatomical_stride = "R" if throwing_side == "left" else "L"
    anatomical_pivot = "L" if anatomical_stride == "R" else "R"

    raw_other_arm = "R" if raw_throwing == "L" else "L"
    raw_other_leg = "R" if raw_stride == "L" else "L"

    cues_agree = raw_throwing != raw_stride
    if cues_agree:
        arm_map = {anatomical_throwing: raw_throwing, anatomical_glove: raw_other_arm}
        leg_map = {anatomical_stride: raw_stride, anatomical_pivot: raw_other_leg}
    else:
        arm_map = {"L": "L", "R": "R"}
        leg_map = {"L": "L", "R": "R"}

    return {
        "arm_map": arm_map,
        "leg_map": leg_map,
        "raw_throwing": raw_throwing,
        "raw_stride": raw_stride,
        "anatomical_throwing": anatomical_throwing,
        "anatomical_stride": anatomical_stride,
        "leg_scores": leg_scores,
        "arm_scores": arm_scores,
        "cues_agree": cues_agree,
        "followthrough": followthrough,
    }


def _pitch_identity_diagnostic(identity: dict, arm_slot: str = "") -> dict:
    """Check arm/stride consistency without inferring synthetic TRC depth."""
    arm = [float(identity["arm_scores"][s]["score"]) for s in ("L", "R")]
    leg = [float(identity["leg_scores"][s]["horizontal_displacement"]) for s in ("L", "R")]
    arm_clear = min(arm) > 0 and max(arm) / min(arm) >= 1.25
    leg_clear = min(leg) > 0 and max(leg) / min(leg) >= 1.25
    if arm_clear and leg_clear:
        status = "consistent" if identity["raw_throwing"] != identity["raw_stride"] else "conflicting"
    else:
        status = "uncertain"
    cues = identity["followthrough"]
    if cues["lifted_raw_foot"] is not None and cues["lifted_raw_foot"] != identity["raw_throwing"]:
        status = "conflicting"
    slot = str(arm_slot).strip().lower().replace("-", "").replace(" ", "")
    if slot in ("overhand", "threequarter", "threequarters"):
        if cues["higher_raw_wrist_fc"] is not None and cues["higher_raw_wrist_fc"] != identity["raw_throwing"]:
            status = "conflicting"
    return {"status": status, "followthrough": cues, "depth_orientation": "unknown"}



def _build_shoulder_opening_sequence_figure(pitch_id: str, feature_path: Path):
    trc_path = _find_trc_for_pitch(pitch_id)
    if trc_path is None:
        return None, None

    try:
        df, marker_names = _read_trc_for_coaching(str(trc_path))
    except Exception:
        return None, None

    row = load_feature_row(feature_path)

    fc_frame = first_value(row, ["front_foot_contact_frame", "fc_frame", "foot_contact_frame"])
    release_frame = first_value(
        row,
        [
            "release_frame",
            "ball_release_frame",
            "release_candidate_frame",
            "final_release_frame",
        ],
    )
    knee_frame = first_value(
        row,
        [
            "knee_lift_frame",
            "maximum_knee_lift_frame",
            "lead_leg_lift_frame",
        ],
    )
    throwing_side = str(first_value(row, ["throwing_side"]) or "").strip().lower()

    fc_frame = _safe_int(fc_frame)
    release_frame = _safe_int(release_frame)
    knee_frame = _safe_int(knee_frame)

    if fc_frame is None or release_frame is None:
        return None, None

    frame_min = int(pd.to_numeric(df["frame"], errors="coerce").min())
    frame_max = int(pd.to_numeric(df["frame"], errors="coerce").max())

    if knee_frame is None:
        knee_frame = max(frame_min, fc_frame - 20)

    joint_map = _resolve_coaching_joint_map(marker_names)
    floor_y = _estimate_trc_floor_y(df, marker_names)

    # 사용할 프레임 정의
    frame_specs = [
        ("FC-3F", max(frame_min, fc_frame - 3), "#94A3B8"),
        ("FC", fc_frame, "#2563EB"),
        ("FC+3F", min(frame_max, fc_frame + 3), "#F59E0B"),
        ("Release", min(frame_max, release_frame), "#EF4444"),
    ]

    # 중복 frame은 허용하되 label은 유지
    sequence_points = []
    for label, frame_no, color in frame_specs:
        pts = _collect_trc_points_for_frame(df, joint_map, frame_no, floor_y)
        if "LShoulder" in pts and "RShoulder" in pts:
            sequence_points.append((label, frame_no, color, pts))

    if not sequence_points:
        return None, None

    fc_points = _collect_trc_points_for_frame(df, joint_map, fc_frame, floor_y)
    knee_points = _collect_trc_points_for_frame(df, joint_map, knee_frame, floor_y)

    # --------------------------------------------------------
    # Full-motion anatomical identity model
    #
    # Instead of deciding L/R from FC or a single anchor frame, use the
    # entire relevant motion:
    #
    #   Leg identity: Knee Lift -> FC trajectory
    #   Arm identity: Release-6F -> Release trajectory
    #
    # Arms and legs get independent raw->anatomical mappings.
    # --------------------------------------------------------

    identity = _canonical_identity_from_full_motion(
        df=df,
        joint_map=joint_map,
        floor_y=floor_y,
        knee_frame=knee_frame,
        fc_frame=fc_frame,
        release_frame=release_frame,
        throwing_side=throwing_side,
    )
    diagnostic = _pitch_identity_diagnostic(identity, first_value(row, ["arm_slot", "arm_slot_category"]) or "")

    raw_for_anatomical = identity["arm_map"]
    leg_raw_for_anatomical = identity["leg_map"]

    stride_prefix = identity["anatomical_stride"]
    front_raw_prefix = leg_raw_for_anatomical.get(
        stride_prefix,
        stride_prefix,
    )

    stride_source = "full_motion_leg_identity"

    traces = []

    # 0) 바닥
    all_pts = []
    for _, _, _, pts in sequence_points:
        all_pts.extend(list(pts.values()))
    all_pts.extend(list(fc_points.values()))

    if not all_pts:
        return None, None

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    z0, z1 = min(zs), max(zs)

    xp = max(0.35, (x1 - x0) * 0.35)
    yp = max(0.35, (y1 - y0) * 0.35)

    xx = np.linspace(x0 - xp, x1 + xp, 2)
    yy = np.linspace(y0 - yp, y1 + yp, 2)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros_like(X)

    traces.append(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=0.10,
            colorscale=[[0, "#d8dde3"], [1, "#d8dde3"]],
            hoverinfo="skip",
            name="Ground",
        )
    )

    # 1) FC 스켈레톤 연하게 표시
    for a, b in COACHING_SKELETON_CONNECTIONS:
        if a in fc_points and b in fc_points:
            p1, p2 = fc_points[a], fc_points[b]
            traces.append(
                go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode="lines",
                    line=dict(width=4, color="#9CA3AF"),
                    opacity=0.35,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if fc_points:
        traces.append(
            go.Scatter3d(
                x=[p[0] for p in fc_points.values()],
                y=[p[1] for p in fc_points.values()],
                z=[p[2] for p in fc_points.values()],
                mode="markers",
                marker=dict(size=3, color="#9CA3AF"),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 2) FC 기준 가이드
    def add_guide(p1, p2, name, color, width=8, opacity=0.85):
        if p1 is None or p2 is None:
            return
        traces.append(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines+markers",
                line=dict(width=width, color=color),
                marker=dict(size=6, color=color),
                opacity=opacity,
                name=name,
                hovertemplate=name + "<extra></extra>",
            )
        )

    add_guide(
        fc_points.get("LHip"),
        fc_points.get("RHip"),
        "Hip Line @ FC",
        "#A855F7",
        width=10,
        opacity=0.90,
    )

    if diagnostic["status"] == "consistent":
        add_guide(
            fc_points.get(f"{front_raw_prefix}Heel"),
            fc_points.get(f"{front_raw_prefix}BigToe"),
            f"Front Foot @ FC ({stride_prefix})",
            "#FACC15",
            width=10,
            opacity=0.95,
        )

    front_center = _foot_center(fc_points, front_raw_prefix)
    if front_center is not None and diagnostic["status"] == "consistent":
        traces.append(go.Scatter3d(
            x=[front_center[0]], y=[front_center[1]], z=[front_center[2] + 0.05],
            mode="text", text=[f"{stride_prefix} FRONT FOOT"],
            textfont=dict(size=12, color="#CA8A04"),
            showlegend=False, hoverinfo="skip",
        ))

    # FC skeleton: anatomical Left/Right arm labels anchored at Release.
    arm_specs = [
        ("L", "#06B6D4", "Left Arm"),
        ("R", "#EC4899", "Right Arm"),
    ]

    for anatomical_prefix, color, base_name in arm_specs:
        if diagnostic["status"] != "consistent":
            continue
        raw_prefix = raw_for_anatomical.get(anatomical_prefix, anatomical_prefix)
        is_throwing = (
            (throwing_side == "left" and anatomical_prefix == "L")
            or (throwing_side == "right" and anatomical_prefix == "R")
        )
        name = base_name + (" (Throwing)" if is_throwing else "")
        segs = [
            (f"{raw_prefix}Shoulder", f"{raw_prefix}Elbow"),
            (f"{raw_prefix}Elbow", f"{raw_prefix}Wrist"),
        ]

        xs, ys, zs = [], [], []
        for a, b in segs:
            if a in fc_points and b in fc_points:
                p1, p2 = fc_points[a], fc_points[b]
                xs += [p1[0], p2[0], None]
                ys += [p1[1], p2[1], None]
                zs += [p1[2], p2[2], None]

        if xs:
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines+markers",
                line=dict(width=9, color=color),
                marker=dict(size=5, color=color),
                opacity=0.95,
                name=name,
                hovertemplate=name + "<extra></extra>",
            ))

            wrist_pt = fc_points.get(f"{raw_prefix}Wrist")
            if wrist_pt is not None:
                label = f"{anatomical_prefix} ARM"
                if is_throwing:
                    label += " · THROW"
                traces.append(go.Scatter3d(
                    x=[wrist_pt[0]], y=[wrist_pt[1]], z=[wrist_pt[2]],
                    mode="text", text=[label],
                    textfont=dict(size=12, color=color),
                    showlegend=False, hoverinfo="skip",
                ))

    # 3) Shoulder Opening Sequence
    for label, frame_no, color, pts in sequence_points:
        p1 = pts.get("LShoulder")
        p2 = pts.get("RShoulder")
        if p1 is None or p2 is None:
            continue

        traces.append(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines+markers+text",
                line=dict(width=12, color=color),
                marker=dict(size=8, color=color),
                text=[label, ""],
                textposition="top center",
                name=f"{label} ({frame_no}F)",
                hovertemplate=f"{label} · Frame {frame_no}<extra></extra>",
            )
        )

        # 어깨선 중앙점도 작게 찍어주면 회전 흐름이 보기 쉬움
        mid = (
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
            (p1[2] + p2[2]) / 2.0,
        )
        traces.append(
            go.Scatter3d(
                x=[mid[0]],
                y=[mid[1]],
                z=[mid[2]],
                mode="markers",
                marker=dict(size=5, color=color),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Projected Shoulder Line Sequence · {pitch_id}",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        scene=dict(
            xaxis=dict(title="X", range=[x0 - 0.4, x1 + 0.4], showspikes=False),
            yaxis=dict(title="Depth", range=[y0 - 0.4, y1 + 0.4], showspikes=False),
            zaxis=dict(title="Height", range=[min(0, z0 - 0.15), z1 + 0.35], showspikes=False),
            aspectmode="data",
            dragmode="orbit",
            camera=dict(
                eye=dict(x=1.7, y=1.9, z=1.0),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"),
            ),
        ),
        uirevision=f"shoulder-seq-{pitch_id}",
    )

    summary = {
        "fc_frame": fc_frame,
        "release_frame": release_frame,
        "frames": [
            (label, frame_no)
            for label, frame_no, _, _ in sequence_points
        ],

        "stride_prefix": stride_prefix,
        "front_raw_prefix": front_raw_prefix,
        "stride_source": stride_source,

        "anatomical_throwing": identity["anatomical_throwing"],
        "raw_throwing": identity["raw_throwing"],
        "anatomical_stride": identity["anatomical_stride"],
        "raw_stride": identity["raw_stride"],

        "arm_map": identity["arm_map"],
        "leg_map": identity["leg_map"],
        "arm_scores": identity["arm_scores"],
        "diagnostic": diagnostic,
        "leg_scores": {
            k: {
                "horizontal_displacement": (
                    None
                    if not np.isfinite(v["horizontal_displacement"])
                    else float(v["horizontal_displacement"])
                )
            }
            for k, v in identity["leg_scores"].items()
        },
    }

    return fig, summary


def show_shoulder_opening_sequence(pitch_id: str, feature_path: Path) -> None:
    show_interactive_3d_viewer(pitch_id, feature_path, key=f"reference_3d_{pitch_id}")
    show_shoulder_projection_cue(pitch_id, feature_path)


def show_shoulder_projection_cue(pitch_id: str, feature_path: Path) -> None:
    """Show observable shoulder span, without claiming a 3D opening angle."""
    motion_path = find_normalized_csv(pitch_id)
    if motion_path is None:
        return
    row = load_feature_row(feature_path)
    fc = _safe_int(first_value(row, ["front_foot_contact_frame", "fc_frame", "foot_contact_frame"]))
    release = _safe_int(first_value(row, ["release_candidate_frame", "release_frame", "pose_release_candidate_frame"]))
    if fc is None or release is None or release <= fc:
        return
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.core.form_comparison import _body_scale, _coordinate

    try:
        motion = pd.read_csv(motion_path)
        frames, widths = [], []
        for _, pose in motion.iterrows():
            frame = _safe_int(pose.get("frame"))
            if frame is None or not fc <= frame <= release:
                continue
            try:
                left = _coordinate(pose, "LShoulder")
                right = _coordinate(pose, "RShoulder")
                width = abs(right[0] - left[0]) / _body_scale(pose)
            except ValueError:
                continue
            if np.isfinite(width) and 0 < width < 4:
                frames.append(frame)
                widths.append(width)
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError):
        return
    if len(frames) < 3:
        return

    st.markdown("**어깨선 변화 살펴보기**")
    st.caption(
        "앞발 착지부터 릴리스 후보까지 영상에서 보이는 양쪽 어깨의 가로 간격입니다. "
        "같은 투수의 반복 투구에서 변화 시점을 확인하세요. 촬영 각도·관절 가림·좌우 라벨에 영향을 받으며, "
        "실제 어깨 열림 각도나 좋은 자세를 뜻하지 않습니다."
    )
    fig = go.Figure(go.Scatter(x=frames, y=widths, mode="lines+markers",
                               line=dict(color="#ee7750", width=3), marker=dict(size=5),
                               hovertemplate="%{x}F · 상대 간격 %{y:.2f}<extra></extra>"))
    for frame, label in ((fc, "앞발 착지"), (release, "릴리스 후보")):
        fig.add_vline(x=frame, line_dash="dash", line_color="#7f8994",
                      annotation_text=label, annotation_position="top")
    fig.update_layout(height=280, margin=dict(l=30, r=20, t=30, b=35),
                      xaxis_title="프레임", yaxis_title="보이는 어깨 폭 / 몸통 길이",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=f"shoulder_span_{pitch_id}")


@st.cache_data(show_spinner=False)
def _load_interactive_trc(path_str: str, modified_ns: int):
    return _load_sports2d_3d_module().read_trc(path_str)


@st.cache_data(show_spinner=False)
def _unreliable_release_arm_frames(motion_path_str: str, modified_ns: int,
                                   release: int, throwing_side: str) -> frozenset[int]:
    """Use source 2D quality flags only near release; never infer 3D confidence."""
    motion = pd.read_csv(motion_path_str)
    bad = set()
    side = "left" if throwing_side == "L" else "right"

    def flag(value) -> bool:
        return str(value).strip().lower() in ("true", "1", "1.0")

    for _, row in motion.iterrows():
        frame = _safe_int(row.get("frame"))
        if frame is None or abs(frame - release) > 10:
            continue
        angle_valid = row.get("throwing_elbow_angle_valid")
        invalid_joint = any(flag(row.get(f"{side}_{part}_invalid"))
                            for part in ("elbow", "wrist"))
        if ((pd.notna(angle_valid) and not flag(angle_valid)) or invalid_joint):
            bad.add(frame)
    return frozenset(bad)


def _load_sports2d_3d_module():
    module_path = SRC_DIR / "core" / "sports2d_3d_viewer.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"3D 뷰어 파일이 없습니다: {module_path}. "
            "다운로드한 파일을 이 이름으로 저장해 주세요."
        )
    module_name = "pitching_sports2d_3d_viewer"
    loaded = sys.modules.get(module_name)
    if loaded is not None and getattr(loaded, "__file__", None) == str(module_path):
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"3D 뷰어 파일을 읽을 수 없습니다: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def show_interactive_3d_viewer(pitch_id: str, feature_path: Path, key: str) -> None:
    """Embed the original TRC viewer without running its standalone Streamlit page."""
    st.subheader("3D 스켈레톤 뷰어")
    st.caption(
        "프레임을 선택하거나 재생하며 자세를 회전·확대할 수 있습니다. "
        "L/R 어깨와 골반을 표시합니다. 깊이는 단일 영상에서 추정한 표시값이며 "
        "실제 포수 방향이나 3D 회전은 판정할 수 없습니다."
    )
    trc_path = _find_trc_for_pitch(pitch_id)
    if trc_path is None:
        st.info("이 투수의 Sports2D TRC 결과가 없습니다.")
        return

    try:
        viewer = _load_sports2d_3d_module()
        df, marker_names = _load_interactive_trc(str(trc_path), trc_path.stat().st_mtime_ns)
        joints = viewer.resolve_joint_map(marker_names)
        floor = viewer.estimate_floor_y(df, marker_names)
        bounds = viewer.compute_global_bounds(df, marker_names, floor)
    except (OSError, ImportError, ValueError, KeyError) as exc:
        st.info(f"3D 뷰어를 표시할 수 없습니다: {exc}")
        return

    minimum, maximum = int(df["frame"].min()), int(df["frame"].max())
    row = load_feature_row(feature_path)
    fc = _safe_int(first_value(row, ["front_foot_contact_frame", "fc_frame", "foot_contact_frame"]))
    release = _safe_int(first_value(row, ["release_candidate_frame", "release_frame", "pose_release_candidate_frame"]))
    knee = _safe_int(first_value(row, ["knee_lift_frame", "maximum_knee_lift_frame", "lead_leg_lift_frame"]))
    default_frame = min(max(knee if knee is not None else minimum, minimum), maximum)
    side = str(first_value(row, ["throwing_side", "pitcher_hand"]) or "").upper()
    motion_path = find_normalized_csv(pitch_id)
    unreliable = frozenset()
    if motion_path is not None and release is not None and side in ("L", "R"):
        try:
            unreliable = _unreliable_release_arm_frames(
                str(motion_path), motion_path.stat().st_mtime_ns, release, side
            )
        except (OSError, ValueError, KeyError):
            pass

    frame_tab, playback_tab = st.tabs(["프레임 보기", "동작 재생"])
    with frame_tab:
        frame = st.slider("프레임", minimum, maximum, default_frame, key=f"{key}_frame")
        fig = viewer.make_frame_figure(
            df, frame, joints, marker_names, bounds, floor,
            fc_frame=fc, release_frame=release,
            unreliable_arm_frames=unreliable, throwing_side=side,
        )
        figure_col, image_col = st.columns([1.2, 1])
        with figure_col:
            st.plotly_chart(
                fig, use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True}, key=f"{key}_figure",
            )
        with image_col:
            overlay = find_analysis_video(pitch_id)
            original = find_event_video(pitch_id)
            for caption, video in (("분석 영상의 2D 관절", overlay), ("원본 영상", original)):
                if video is None or (caption == "원본 영상" and video == overlay):
                    continue
                image_rgb, actual, _ = read_video_frame(str(video), frame)
                if image_rgb is not None:
                    st.image(image_rgb, caption=f"{caption} · {actual}F", use_container_width=True)
                    if actual != frame:
                        st.caption("영상과 TRC의 프레임 범위가 다릅니다. 시점을 직접 확인하세요.")
        if frame in unreliable:
            st.caption(
                "이 프레임은 2D 투구팔 관절 검증이 실패해 팔꿈치·손목과 연결선을 "
                "3D에서 생략했습니다. 오른쪽 영상의 실제 자세를 확인하세요."
            )
    with playback_tab:
        start_default = min(max((knee or minimum) - 10, minimum), maximum)
        end_default = min(max((release or maximum) + 10, start_default), maximum)
        start, end = st.slider(
            "재생 구간", minimum, maximum, (start_default, end_default),
            key=f"{key}_range",
        )
        step = st.select_slider("프레임 간격", options=[1, 2, 3, 4, 5], value=1, key=f"{key}_step")
        if (end - start) // step + 1 > 300:
            st.info("재생 프레임이 많습니다. 구간을 줄이거나 프레임 간격을 늘려주세요.")
        else:
            animation = viewer.make_animation_figure(
                df, joints, marker_names, bounds, floor,
                start_frame=start, end_frame=end, frame_step=step,
                fc_frame=fc, release_frame=release,
                unreliable_arm_frames=unreliable, throwing_side=side,
            )
            viewer.render_live_zoom_animation(animation, height=620)
            st.caption(
                "드래그로 회전, 휠로 확대할 수 있습니다. 릴리스 후보 전후 2D 관절 검증이 "
                "실패한 프레임은 잘못된 3D 자세로 보이지 않게 투구팔 일부를 생략합니다."
            )




def find_event_video(pitch_id: str) -> Path | None:
    """
    Feature의 이벤트 frame 번호와 동일한 분석 타임라인을 가진
    영상을 우선적으로 찾는다.

    원본 영상을 우선해 오버레이에 포함된 분석용 숫자가 이벤트
    썸네일에 보이지 않게 한다. 원본이 없을 때만 분석 영상을 사용한다.
    """
    pitch_root = PIPELINE_OUTPUT_DIR / pitch_id

    raw_candidates = [RAW_VIDEO_DIR / f"{pitch_id}{ext}" for ext in (".mp4", ".mov", ".avi", ".mkv")]
    manifest = read_manifest()
    if not manifest.empty and {"pitch_id", "video_file"}.issubset(manifest.columns):
        rows = manifest.loc[manifest["pitch_id"].astype(str) == pitch_id]
        if not rows.empty:
            raw_candidates.insert(0, RAW_VIDEO_DIR / str(rows.iloc[0]["video_file"]))
    candidates = raw_candidates + [
        pitch_root / "videos" / f"{pitch_id}_analysis_overlay.mp4",
        pitch_root / "videos" / f"{pitch_id}_sample_analysis_overlay.mp4",
        pitch_root / "videos" / f"{pitch_id}_pitch_segment.mp4",
        pitch_root / "videos" / f"{pitch_id}_sample_pose.mp4",

        ROOT / "results" / "overlay" / "sample_analysis_overlay.mp4",
        ROOT / "results" / "segment" / "sample" / "pitch_segment.mp4",
        ROOT / "results" / "pose" / "sample_pose.mp4",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


@st.cache_data(show_spinner=False)
def read_video_frame(
    video_path_str: str,
    frame_number: int,
) -> tuple[np.ndarray | None, int, float]:
    """
    특정 frame을 RGB 이미지로 반환.

    Returns
    -------
    image_rgb : np.ndarray | None
    actual_frame : int
    fps : float
    """
    video_path = Path(video_path_str)

    if not video_path.exists():
        return None, frame_number, 0.0

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, frame_number, 0.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if not fps or fps <= 0:
        fps = 30.0

    if total_frames <= 0:
        cap.release()
        return None, frame_number, fps

    actual_frame = max(
        0,
        min(
            int(frame_number),
            total_frames - 1,
        ),
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        actual_frame,
    )

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None, actual_frame, fps

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    return frame_rgb, actual_frame, fps


def get_event_frames_from_feature(
    feature_path: Path,
) -> list[dict]:
    """
    Feature CSV에서 Knee Lift / FC / Release Candidate frame 추출.
    실제 프로젝트의 여러 컬럼명 버전과 호환.
    """
    row = load_feature_row(feature_path)

    knee_frame = first_value(
        row,
        [
            "knee_lift_frame",
            "maximum_knee_lift_frame",
            "lead_leg_lift_frame",
        ],
    )

    fc_frame = first_value(
        row,
        [
            "front_foot_contact_frame",
            "fc_frame",
            "foot_contact_frame",
        ],
    )

    release_frame = first_value(
        row,
        [
            "release_candidate_frame",
            "release_frame",
            "pose_release_candidate_frame",
        ],
    )

    release_conf = first_value(
        row,
        [
            "release_confidence",
            "release_candidate_confidence",
        ],
    )

    events = [
        {
            "title": "최대 리드 레그 리프트",
            "short": "Knee Lift",
            "frame": knee_frame,
            "confidence": None,
        },
        {
            "title": "앞발 착지",
            "short": "Front Foot Contact",
            "frame": fc_frame,
            "confidence": None,
        },
        {
            "title": "릴리스 후보",
            "short": "Release Candidate",
            "frame": release_frame,
            "confidence": release_conf,
        },
    ]

    return events


def show_event_frames(
    pitch_id: str,
    feature_path: Path,
) -> None:
    """
    Knee Lift / FC / Release Candidate 대표 프레임 3장을 표시.
    """
    video_path = find_event_video(
        pitch_id
    )

    if video_path is None:
        st.caption(
            "이벤트 프레임을 표시할 분석 영상을 찾지 못했습니다."
        )
        return

    events = get_event_frames_from_feature(
        feature_path
    )

    valid_events = [
        event
        for event in events
        if event["frame"] is not None
        and pd.notna(event["frame"])
    ]

    if not valid_events:
        st.caption(
            "Feature CSV에서 이벤트 frame 정보를 찾지 못했습니다."
        )
        return

    st.subheader("주요 이벤트 프레임")

    st.caption(
        "검출된 Knee Lift, Front Foot Contact, "
        "Pose-based Release Candidate 시점의 자세를 확인합니다."
    )

    cols = st.columns(3)

    for col, event in zip(
        cols,
        events,
    ):
        with col:
            st.markdown(
                f"**{event['title']}**"
            )

            if (
                event["frame"] is None
                or pd.isna(event["frame"])
            ):
                st.info(
                    f"{event['short']}\n\n프레임 정보 없음"
                )
                continue

            frame_number = int(
                float(
                    event["frame"]
                )
            )

            image, actual_frame, fps = (
                read_video_frame(
                    str(video_path),
                    frame_number,
                )
            )

            if image is None:
                st.warning(
                    f"Frame {frame_number}을 읽지 못했습니다."
                )
                continue

            st.image(
                image,
                use_container_width=True,
            )

            time_sec = (
                actual_frame / fps
                if fps > 0
                else None
            )

            if time_sec is not None:
                st.caption(
                    f"Frame **{actual_frame}** · "
                    f"{time_sec:.3f} s"
                )
            else:
                st.caption(
                    f"Frame **{actual_frame}**"
                )

            if event["confidence"] is not None:
                st.caption(
                    "Confidence: "
                    f"**{fmt_confidence(event['confidence'])}**"
                )



# ============================================================
# [CALL 1] 기준 투수 탭
# 기존:
#
#     show_feature_dashboard(feature_path)
#     show_archived_visuals(selected, feature_path)
#
# 를 아래처럼 변경하세요.
# ============================================================

# show_feature_dashboard(feature_path)
# show_event_frames(selected, feature_path)
# show_archived_visuals(selected, feature_path)


# ============================================================
# [CALL 2] 새 투수 분석 성공 후
# 기존:
#
#     show_feature_dashboard(feature_path)
#     show_archived_visuals(pitch_id, feature_path)
#
# 를 아래처럼 변경하세요.
# ============================================================

# show_feature_dashboard(feature_path)
# show_event_frames(pitch_id, feature_path)
# show_archived_visuals(pitch_id, feature_path)

def _get_numeric_feature(pitch_id: str, aliases: list[str]) -> float | None:
    path = find_feature_csv_for_pitch(pitch_id)
    if path is None:
        return None
    row = load_feature_row(path)
    value = first_value(row, aliases)
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_easy_compare_summary(
    pitch_a: str,
    pitch_b: str,
    label_a: str,
    label_b: str,
) -> dict:
    fc_rel_a = _get_numeric_feature(pitch_a, [
        "fc_to_release_time_s", "fc_to_release_time",
        "fc_release_time_sec", "fc_to_release_sec",
    ])
    fc_rel_b = _get_numeric_feature(pitch_b, [
        "fc_to_release_time_s", "fc_to_release_time",
        "fc_release_time_sec", "fc_to_release_sec",
    ])

    sentences = []
    cards = []

    if fc_rel_a is not None and fc_rel_b is not None:
        diff = fc_rel_a - fc_rel_b
        if abs(diff) < 0.010:
            text = f"{label_a}와 {label_b}의 착지 후 릴리스 후보까지 걸리는 시간은 거의 비슷합니다."
            short = "거의 비슷함"
        elif diff < 0:
            text = f"{label_a}는 {label_b}보다 앞발 착지 후 릴리스 후보까지 {abs(diff):.3f}초 더 빠릅니다."
            short = f"{label_a}가 {abs(diff):.3f}s 빠름"
        else:
            text = f"{label_b}는 {label_a}보다 앞발 착지 후 릴리스 후보까지 {abs(diff):.3f}초 더 빠릅니다."
            short = f"{label_b}가 {abs(diff):.3f}s 빠름"
        sentences.append(text)
        cards.append({
            "title": "착지 → 릴리스 후보",
            "a": f"{fc_rel_a:.3f} s",
            "b": f"{fc_rel_b:.3f} s",
            "summary": short,
            "help": "앞발 착지부터 자세 기반 릴리스 후보 프레임까지의 시간입니다.",
        })

    return {
        "headline": sentences[0] if sentences else "비교 가능한 주요 Feature 값이 충분하지 않습니다.",
        "sentences": sentences,
        "cards": cards,
    }


def get_manifest_pitch_row(
    pitch_id: str,
) -> pd.Series | None:
    manifest = read_manifest()

    if (
        manifest.empty
        or
        "pitch_id" not in manifest.columns
    ):
        return None

    matches = manifest.loc[
        manifest["pitch_id"]
        .astype(str)
        .str.strip()
        == str(pitch_id).strip()
    ]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def load_analysis_events(pitch_id: str) -> dict:
    """Load the canonical analysis_events.json for one pitch."""
    path = PIPELINE_OUTPUT_DIR / pitch_id / "analysis_events.json"

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_arm_slot_label(value) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return None

    key = (
        text.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    mapping = {
        "overhand": "Overhand",
        "three_quarter": "Three-Quarter",
        "threequarter": "Three-Quarter",
        "sidearm": "Sidearm",
        "underhand": "Underhand",
        "submarine": "Underhand",
    }

    return mapping.get(key, text)


def get_pitcher_arm_slot_info(
    pitch_id: str,
    feature_path: Path | None = None,
) -> dict | None:
    """
    Canonical Arm Slot display source.

    UI intentionally exposes only the categorical slot and confidence.
    The internal 2D projected classification angle is not shown.
    """
    events = load_analysis_events(pitch_id)
    slot_obj = events.get("arm_slot")

    label = None
    confidence = None
    source = None

    if isinstance(slot_obj, dict):
        label = (
            slot_obj.get("category")
            or slot_obj.get("label")
            or slot_obj.get("arm_slot")
        )
        confidence = slot_obj.get("confidence")
        source = slot_obj.get("source")
    elif slot_obj is not None:
        label = slot_obj

    if label is None and feature_path is not None and feature_path.exists():
        row = load_feature_row(feature_path)
        label = first_value(row, ["arm_slot", "arm_slot_category"])
        confidence = first_value(row, ["arm_slot_confidence"])
        source = first_value(row, ["arm_slot_source"])

    label = normalize_arm_slot_label(label)
    if label is None:
        return None

    slot_key_map = {
        "Overhand": "overhand",
        "Three-Quarter": "three_quarter",
        "Sidearm": "sidearm",
        "Underhand": "submarine",
    }

    slot_key = slot_key_map.get(label, "three_quarter")
    visual_angle = ARM_SLOT_VISUAL_ANGLES.get(slot_key, 40.0)

    return {
        "label": label,
        "arm_slot": slot_key,
        "visual_angle": float(visual_angle),
        "confidence": (
            str(confidence).strip()
            if confidence is not None and not pd.isna(confidence)
            else None
        ),
        "source": (
            str(source).strip()
            if source is not None and not pd.isna(source)
            else None
        ),
    }


# ============================================================
# Statcast-style Arm Angle Visualization
# ============================================================

def get_pitcher_arm_angle_info(
    pitch_id: str,
    feature_path: Path,
) -> dict | None:
    """
    Arm Angle 우선순위
    1) 기준 MLB 투수: REFERENCE_ARM_ANGLES 값
    2) 실제 Arm Angle 컬럼이 명시적으로 존재하는 경우만 사용

    신규 업로드 투수의 2D projection angle은 Statcast Arm Angle로
    취급하지 않습니다. Arm Angle이 없으면 UI에서 사용자 지정
    Arm Slot 카드로 fallback합니다.

    주의: Release Elbow Angle은 Arm Angle과 다른 지표이므로
    대체값으로 사용하지 않습니다.
    """
    if pitch_id in REFERENCE_ARM_ANGLES:
        info = REFERENCE_ARM_ANGLES[pitch_id].copy()
        info["origin"] = "reference"
        return info

    row = load_feature_row(feature_path)
    value = first_value(
        row,
        [
            "arm_angle_deg",
            "pitcher_arm_angle_deg",
            "release_arm_angle_deg",
            "arm_slot_angle_deg",
            "arm_slot_deg",
        ],
    )

    try:
        if value is None or pd.isna(value):
            return None
        angle = float(value)
    except (TypeError, ValueError):
        return None

    source = first_value(
        row,
        [
            "arm_angle_source",
            "arm_slot_source",
        ],
    )

    return {
        "angle": angle,
        "source": str(source) if source is not None else "2D Estimated",
        "reference": "Feature CSV",
        "origin": "feature",
    }


def get_pitcher_arm_angle_value(
    pitch_id: str,
    feature_path: Path,
) -> float | None:
    info = get_pitcher_arm_angle_info(pitch_id, feature_path)
    return None if info is None else float(info["angle"])


def choose_arm_angle_template(angle_deg: float) -> Path:
    # Statcast와 같은 부호 체계: 0°=수평, 음수=언더핸드 방향
    return (
        ARM_ANGLE_TEMPLATE_UNDERHAND
        if angle_deg < 0
        else ARM_ANGLE_TEMPLATE_NORMAL
    )


def arm_angle_slot_label(angle_deg: float) -> str:
    if angle_deg < 0:
        return "언더핸드 / 서브마린"
    if angle_deg < 18:
        return "낮은 사이드암 계열"
    if angle_deg < 32:
        return "로우 쓰리쿼터 계열"
    if angle_deg < 48:
        return "쓰리쿼터 계열"
    if angle_deg < 60:
        return "오버핸드 계열"
    return "하이 오버핸드 계열"


# 각 템플릿에서 검은 선의 시작점만 수동 고정합니다.
# 선의 끝점은 angle 값으로 자동 계산합니다.
ARM_ANGLE_LINE_CONFIG = {
    "normal": {
        "origin_px": (593, 477),
        "line_length_px": 275,
        "baseline_len_px": 125,
        "label_ratio": (0.27, 0.46),
    },
    "underhand": {
        "origin_px": (679, 564),
        "line_length_px": 275,
        "baseline_len_px": 125,
        "label_ratio": (0.30, 0.49),
    },
}


def render_release_arm_angle_card(
    pitch_id: str,
    feature_path: Path,
) -> Image.Image | None:
    info = get_pitcher_arm_angle_info(pitch_id, feature_path)
    if info is None:
        return None

    angle_deg = float(info["angle"])
    template_path = choose_arm_angle_template(angle_deg)

    if not template_path.exists():
        raise FileNotFoundError(
            f"Arm Angle 템플릿 이미지를 찾지 못했습니다: {template_path}"
        )

    image = Image.open(template_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    config = ARM_ANGLE_LINE_CONFIG[
        "underhand" if angle_deg < 0 else "normal"
    ]

    ox, oy = config["origin_px"]
    length = int(config["line_length_px"])
    theta = math.radians(angle_deg)

    ex = int(round(ox + length * math.cos(theta)))
    ey = int(round(oy - length * math.sin(theta)))

    line_width = max(13, int(w * 0.017))
    endpoint_r = max(11, int(w * 0.013))
    baseline_width = max(4, int(w * 0.0042))

    black = (20, 20, 20, 255)
    gray = (135, 160, 165, 210)
    white = (245, 245, 245, 255)

    subtitle_font = _safe_font(24, bold=False)
    small_font = _safe_font(max(16, int(w * 0.022)), bold=False)
    value_font = _safe_font(max(34, int(w * 0.055)), bold=True)

    draw.text((330, 90), "Pitcher Arm Angle", font=subtitle_font, fill=(111, 126, 130))
    draw.line((470, 130, 620, 130), fill=(122, 198, 185), width=3)

    baseline_len = int(config["baseline_len_px"])
    draw.line(
        [(ox, oy), (ox + baseline_len, oy)],
        fill=gray,
        width=baseline_width,
    )

    draw.line(
        [(ox, oy), (ex, ey)],
        fill=black,
        width=line_width,
    )

    draw.ellipse(
        (ox - endpoint_r, oy - endpoint_r, ox + endpoint_r, oy + endpoint_r),
        fill=black,
    )
    draw.ellipse(
        (ex - endpoint_r, ey - endpoint_r, ex + endpoint_r, ey + endpoint_r),
        fill=black,
    )
    inner_r = max(2, endpoint_r // 2)
    draw.ellipse(
        (ex - inner_r, ey - inner_r, ex + inner_r, ey + inner_r),
        fill=white,
    )

    label_x = int(w * config["label_ratio"][0])
    label_y = int(h * config["label_ratio"][1])

    draw.text(
        (label_x, label_y),
        "ARM\nANGLE",
        font=small_font,
        fill=(85, 135, 145, 255),
        spacing=2,
        align="center",
    )
    draw.text(
        (label_x, label_y + int(h * 0.055)),
        f"{angle_deg:.0f}°",
        font=value_font,
        fill=black,
    )

    return image.convert("RGB")



def render_arm_slot_card(
    pitch_id: str,
) -> Image.Image | None:
    """
    Arm Slot 범주형 시각화 카드.

    - 실제 측정 각도는 표시하지 않는다.
    - 검은 두 선을 상완/전완처럼 사용해 대표적인 팔 방향만 보여준다.
    - Overhand / Three-Quarter / Sidearm은 일반 템플릿,
      Underhand는 언더핸드 템플릿을 사용한다.
    """
    info = get_pitcher_arm_slot_info(
        pitch_id
    )

    if info is None:
        return None

    visual_angle = float(
        info["visual_angle"]
    )
    slot_label = str(
        info["label"]
    )

    template_path = (
        ARM_ANGLE_TEMPLATE_UNDERHAND
        if slot_label == "Underhand"
        else ARM_ANGLE_TEMPLATE_NORMAL
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Arm Slot 템플릿 이미지를 찾지 못했습니다: {template_path}"
        )

    image = Image.open(
        template_path
    ).convert("RGBA")

    # Arm Slot 카드의 템플릿 배경은 원본 이미지의 회색 vignette 대신
    # 라이트/다크 모드 모두에서 카드 자체가 깨끗하게 보이도록 흰색으로 정리한다.
    # 채도가 낮고 밝은 픽셀만 배경으로 간주하므로 초록 실루엣은 보존한다.
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    rgb_max = arr.max(axis=2).astype(np.int16)
    rgb_min = arr.min(axis=2).astype(np.int16)
    neutral = (rgb_max - rgb_min) < 18
    bright = rgb_min > 205
    bg_mask = neutral & bright
    arr[bg_mask] = 255
    image = Image.fromarray(arr, mode="RGB").convert("RGBA")

    draw = ImageDraw.Draw(
        image
    )

    w, h = image.size

    config = ARM_ANGLE_LINE_CONFIG[
        "underhand"
        if slot_label == "Underhand"
        else "normal"
    ]

    shoulder_x, shoulder_y = config[
        "origin_px"
    ]

    total_length = int(
        config["line_length_px"]
        * 0.88
    )

    theta = math.radians(
        visual_angle
    )

    # 대표 Arm Slot 방향 벡터
    dx = math.cos(theta)
    dy = -math.sin(theta)

    # 팔꿈치를 완전한 직선보다 조금 아래쪽으로 꺾어서
    # 상완 + 전완처럼 보이게 한다.
    perp_x = -dy
    perp_y = dx

    elbow_length = total_length * 0.48
    bend = total_length * 0.09

    elbow_x = int(round(
        shoulder_x
        + elbow_length * dx
        + bend * perp_x
    ))
    elbow_y = int(round(
        shoulder_y
        + elbow_length * dy
        + bend * perp_y
    ))

    wrist_x = int(round(
        shoulder_x
        + total_length * dx
    ))
    wrist_y = int(round(
        shoulder_y
        + total_length * dy
    ))

    line_width = max(
        13,
        int(w * 0.017)
    )
    joint_r = max(
        10,
        int(w * 0.012)
    )

    black = (
        20,
        20,
        20,
        255,
    )
    joint_outline = (
        242,
        242,
        242,
        255,
    )

    subtitle_font = _safe_font(
        24,
        bold=False
    )
    value_font = _safe_font(
        max(
            30,
            int(w * 0.047)
        ),
        bold=True
    )

    draw.text(
        (330, 90),
        "Pitcher Arm Slot",
        font=subtitle_font,
        fill=(111, 126, 130),
    )

    draw.line(
        (470, 130, 620, 130),
        fill=(122, 198, 185),
        width=3,
    )

    # 상완 / 전완
    draw.line(
        [
            (shoulder_x, shoulder_y),
            (elbow_x, elbow_y),
        ],
        fill=black,
        width=line_width,
    )
    draw.line(
        [
            (elbow_x, elbow_y),
            (wrist_x, wrist_y),
        ],
        fill=black,
        width=line_width,
    )

    # Shoulder / Elbow / Wrist 점
    for x, y in [
        (shoulder_x, shoulder_y),
        (elbow_x, elbow_y),
        (wrist_x, wrist_y),
    ]:
        draw.ellipse(
            (
                x - joint_r,
                y - joint_r,
                x + joint_r,
                y + joint_r,
            ),
            fill=black,
            outline=joint_outline,
            width=2,
        )

    # 각도 숫자 대신 범주 이름만 표시
    label_x = int(w * 0.59)
    label_y = int(h * 0.73)

    draw.text(
        (
            label_x,
            label_y,
        ),
        slot_label,
        font=value_font,
        fill=black,
    )

    return image.convert(
        "RGB"
    )


@st.cache_data(show_spinner=False)
def generate_arm_slot_card_image(
    pitch_id: str,
) -> bytes | None:
    image = render_arm_slot_card(
        pitch_id
    )

    if image is None:
        return None

    import io

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def generate_release_arm_angle_card_image(
    pitch_id: str,
    feature_path_str: str,
) -> bytes | None:
    feature_path = Path(feature_path_str)
    image = render_release_arm_angle_card(pitch_id, feature_path)
    if image is None:
        return None

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def show_release_arm_angle_visual_card(
    pitch_id: str,
    feature_path: Path,
) -> None:
    st.subheader("투구 팔 각도")
    st.caption(
        "릴리스 순간의 Arm Angle을 단순한 방향선으로 표시합니다. "
        "0°는 수평 기준이며, 음수 값은 언더핸드 템플릿을 사용합니다."
    )

    info = get_pitcher_arm_angle_info(pitch_id, feature_path)
    if info is None:
        st.info(
            "이 투수의 Arm Angle 값이 없습니다. "
            "Arm Angle은 Release Elbow Angle과 다른 지표이므로 "
            "팔꿈치 각도로 대체하지 않습니다."
        )
        return

    image_bytes = generate_release_arm_angle_card_image(
        pitch_id=pitch_id,
        feature_path_str=str(feature_path),
    )

    if image_bytes is None:
        st.info("Arm Angle 시각화를 생성하지 못했습니다.")
        return

    # 한 장이 너무 커지지 않도록 중앙의 좁은 열에 배치
    left_spacer, center_col, right_spacer = st.columns([2.35, 2.3, 2.35])
    with center_col:
        st.image(image_bytes, width=CARD_DISPLAY_WIDTH)

    angle_deg = float(info["angle"])
    source = str(info.get("source", "—"))
    reference = str(info.get("reference", ""))

    st.caption(
        f"**Arm Angle {angle_deg:.1f}°** · "
        f"{arm_angle_slot_label(angle_deg)} · "
        f"Source: {source}"
        + (f" · {reference}" if reference else "")
    )


def show_easy_compare_summary(
    pitch_a: str,
    pitch_b: str,
    label_a: str,
    label_b: str,
) -> None:
    summary = build_easy_compare_summary(
        pitch_a, pitch_b, label_a, label_b
    )

    st.subheader("한눈에 보는 비교")

    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(120,120,120,0.30);
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 14px;
            background: rgba(255,255,255,0.025);
        ">
            <div style="font-size:1.08rem;font-weight:700;margin-bottom:8px;">
                {summary["headline"]}
            </div>
            <div style="color:#aeb6c2;line-height:1.65;">
                아래 내용은 두 투수의 분석 수치를 쉽게 풀어쓴 것입니다.
                특정 수치가 더 크거나 작다고 해서 곧바로 더 좋은 투구폼을 의미하지는 않습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = summary["cards"]

    for start in range(0, len(cards), 4):
        chunk = cards[start:start + 4]
        cols = st.columns(len(chunk))

        for col, card in zip(cols, chunk):
            with col:
                st.markdown(f"**{card['title']}**")
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid rgba(120,120,120,0.25);
                        border-radius:10px;
                        padding:12px 14px;
                        min-height:145px;
                    ">
                        <div style="font-size:0.88rem;color:#9aa4b2;">{label_a}</div>
                        <div style="font-size:1.15rem;font-weight:700;margin-bottom:6px;">{card['a']}</div>
                        <div style="font-size:0.88rem;color:#9aa4b2;">{label_b}</div>
                        <div style="font-size:1.15rem;font-weight:700;margin-bottom:8px;">{card['b']}</div>
                        <div style="font-size:0.90rem;color:#d5d9df;">{card['summary']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(card["help"])

    if summary["sentences"]:
        with st.expander("쉬운 설명 자세히 보기"):
            for sentence in summary["sentences"]:
                st.markdown(f"- {sentence}")

            st.markdown(
                """
                **용어 설명**
                - **FC (Front Foot Contact)**: 내딛는 앞발이 지면에 닿는 순간
                - **Release 후보**: 자세 변화로 추정한 릴리스 부근 프레임
                """
            )

# ============================================================
# Dynamic Pitcher Comparison
# ============================================================

def find_normalized_csv(pitch_id: str) -> Path | None:
    """투수별 정규화 Motion CSV를 탐색합니다."""
    pitch_pose_dir = PIPELINE_OUTPUT_DIR / pitch_id / "pose_csv"

    direct_candidates = [
        pitch_pose_dir / f"{pitch_id}_motion_normalized.csv",
        DATA_DIR / "pose_csv" / f"{pitch_id}_motion_normalized.csv",
        DATA_DIR / "pose_csv" / f"{pitch_id}_normalized.csv",
        pitch_pose_dir / f"{pitch_id}_normalized.csv",
        pitch_pose_dir / "sample_motion_normalized.csv",
    ]

    for path in direct_candidates:
        if path.exists():
            return path

    if pitch_pose_dir.exists():
        patterns = [
            "*motion_normalized*.csv",
            "*normalized*motion*.csv",
            "*normalized*.csv",
        ]
        for pattern in patterns:
            matches = sorted(pitch_pose_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


@st.cache_data(show_spinner=False)
def _cached_arm_comparison_gif(
    motion_a: str, feature_a: str, label_a: str,
    motion_b: str, feature_b: str, label_b: str,
    window: int,
    mode: str,
    file_versions: tuple[int, int, int, int],
) -> bytes:
    # file_versions invalidates Streamlit's cache when a pitch is reprocessed.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.core.comparison_visualization import make_arm_comparison_gif, make_leg_comparison_gif

    make_gif = make_arm_comparison_gif if mode == "arm" else make_leg_comparison_gif
    return make_gif(
        Path(motion_a), Path(feature_a), label_a,
        Path(motion_b), Path(feature_b), label_b,
        window=window,
    )


def show_arm_motion_compare(
    pitch_a: str, pitch_b: str, label_a: str, label_b: str,
    feature_a: Path, feature_b: Path,
) -> None:
    motion_a, motion_b = find_normalized_csv(pitch_a), find_normalized_csv(pitch_b)
    if motion_a is None or motion_b is None:
        st.info("팔 동작 GIF를 만들 정규화 Pose CSV가 없습니다.")
        return
    paths = (motion_a, feature_a, motion_b, feature_b)
    try:
        window = st.slider("릴리스 후보 전후 프레임", min_value=3, max_value=20,
                           value=10, step=1, key="arm_gif_window")
        gif = _cached_arm_comparison_gif(
            str(motion_a), str(feature_a), label_a,
            str(motion_b), str(feature_b), label_b,
            window, "arm",
            tuple(path.stat().st_mtime_ns for path in paths),
        )
    except (OSError, KeyError, ValueError) as exc:
        st.warning(f"팔 동작 GIF 생성 실패: {exc}")
        return
    st.image(gif, caption=f"각 투구의 자동 릴리스 후보 ±{window}프레임 팔 움직임",
             use_container_width=True)
    st.caption("가운데는 자동 검출 후보이며 실제 공이 떠난 순간과 다를 수 있습니다. 좌완을 반전하지 않고 영상의 좌우 방향을 유지했습니다. *는 팔꿈치 각도 검증에서 제외된 프레임입니다.")


def show_leg_lift_compare(
    pitch_a: str, pitch_b: str, label_a: str, label_b: str,
    feature_a: Path, feature_b: Path,
) -> None:
    motion_a, motion_b = find_normalized_csv(pitch_a), find_normalized_csv(pitch_b)
    if motion_a is None or motion_b is None:
        st.info("다리 동작 GIF를 만들 정규화 Pose CSV가 없습니다.")
        return
    paths = (motion_a, feature_a, motion_b, feature_b)
    try:
        window = st.slider("무릎 들기 기준 전후 프레임", min_value=5, max_value=30,
                           value=20, step=1, key="leg_gif_window")
        gif = _cached_arm_comparison_gif(
            str(motion_a), str(feature_a), label_a,
            str(motion_b), str(feature_b), label_b,
            window, "leg",
            tuple(path.stat().st_mtime_ns for path in paths),
        )
    except (OSError, KeyError, ValueError) as exc:
        st.warning(f"다리 동작 GIF 생성 실패: {exc}")
        return
    st.image(gif, caption=f"각 투구의 Knee Lift 기준 ±{window}프레임 앞다리 움직임",
             use_container_width=True)
    st.caption("엉덩이를 원점으로 맞춰 앞다리의 엉덩이–무릎–발목을 겹쳤습니다. Knee Lift는 현재 수동 기준 프레임이며, 서로 다른 촬영 각도는 보정되지 않습니다.")


def first_existing_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for column in aliases:
        if column in df.columns:
            return column
    return None


def normalized_axis_column(df: pd.DataFrame) -> str | None:
    return first_existing_column(
        df,
        [
            "pitch_progress_time_percent",
            "pitch_progress_percent",
            "normalized_pitch_percent",
            "motion_progress_percent",
            "normalized_time_percent",
            "pitch_progress",
        ],
    )


def elbow_angle_column(df: pd.DataFrame) -> str | None:
    return first_existing_column(
        df,
        [
            "throwing_elbow_angle_filtered",
            "throwing_elbow_angle_deg",
            "elbow_angle_deg",
            "throwing_elbow_angle",
            "elbow_angle",
        ],
    )


def wrist_speed_column(df: pd.DataFrame) -> str | None:
    return first_existing_column(
        df,
        [
            "wrist_relative_speed_body_s_validated",
            "wrist_speed_body_s_validated",
            "validated_wrist_speed_body_s",
            "wrist_speed_body_s",
            "wrist_relative_speed_body_s",
            "wrist_speed_body_per_s",
            "wrist_speed_normalized",
            "wrist_speed",
        ],
    )


def validity_column(df: pd.DataFrame, kind: str) -> str | None:
    aliases = (
        ["elbow_valid", "is_elbow_valid", "angle_valid", "is_angle_valid"]
        if kind == "elbow"
        else ["wrist_speed_valid", "is_wrist_speed_valid", "wrist_valid", "is_wrist_valid"]
    )
    return first_existing_column(df, aliases)


def prepare_curve(pitch_id: str, value_kind: str) -> pd.DataFrame | None:
    csv_path = find_normalized_csv(pitch_id)
    if csv_path is None:
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    x_col = normalized_axis_column(df)
    if value_kind == "elbow":
        y_col = elbow_angle_column(df)
        valid_col = validity_column(df, "elbow")
    else:
        y_col = wrist_speed_column(df)
        valid_col = validity_column(df, "wrist")

    if x_col is None or y_col is None:
        return None

    work = pd.DataFrame({
        "progress": pd.to_numeric(df[x_col], errors="coerce"),
        "value": pd.to_numeric(df[y_col], errors="coerce"),
    })

    if valid_col is not None:
        valid_series = df[valid_col]
        if valid_series.dtype == bool:
            valid_mask = valid_series.fillna(False)
        else:
            valid_mask = (
                valid_series.astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes", "valid"])
            )
        work.loc[~valid_mask.to_numpy(), "value"] = np.nan

    work = work.dropna(subset=["progress", "value"])
    work = work[(work["progress"] >= 0) & (work["progress"] <= 120)].copy()
    if work.empty:
        return None

    return (
        work.groupby("progress", as_index=False)["value"]
        .mean()
        .sort_values("progress")
    )


def show_dynamic_curve_compare(
    pitch_a: str,
    pitch_b: str,
    label_a: str,
    label_b: str,
    value_kind: str,
) -> None:
    curve_a = prepare_curve(pitch_a, value_kind)
    curve_b = prepare_curve(pitch_b, value_kind)

    if curve_a is None and curve_b is None:
        st.info("선택한 두 투수의 정규화 곡선 데이터를 찾지 못했습니다.")
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    if curve_a is not None:
        ax.plot(curve_a["progress"], curve_a["value"], linewidth=2.2, label=label_a)
    if curve_b is not None:
        ax.plot(curve_b["progress"], curve_b["value"], linewidth=2.2, label=label_b)

    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.45)
    ax.axvline(100, linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Pitch Progress (%)  ·  Knee Lift = 0%, Release = 100%")

    if value_kind == "elbow":
        ax.set_title("Normalized Elbow Angle Comparison")
        ax.set_ylabel("Elbow Angle (deg)")
    else:
        ax.set_title("Normalized Wrist Speed Comparison")
        ax.set_ylabel("Wrist Speed (body/s)")

    ax.grid(alpha=0.20)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def compare_metric_specs() -> dict[str, list[str]]:
    return {
        "Knee Lift → FC (s)": [
            "knee_lift_to_fc_time_s",
            "knee_lift_to_fc_time",
        ],
        "FC → Release 후보 (s)": [
            "fc_to_release_time_s",
            "fc_to_release_time",
            "fc_release_time_sec",
            "fc_to_release_sec",
        ],
    }


def format_compare_value(label: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if "(s)" in label:
        return f"{number:.3f} s"
    if "(°)" in label:
        return f"{number:.1f}°"
    if "(%)" in label:
        return f"{number:.1f}%"
    if "body/s" in label:
        return f"{number:.3f} body/s"
    return f"{number:.3f}"

def build_feature_compare_table(
    feature_path_a: Path,
    feature_path_b: Path,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    row_a = load_feature_row(feature_path_a)
    row_b = load_feature_row(feature_path_b)

    rows = []
    for label, aliases in compare_metric_specs().items():
        value_a = first_value(row_a, aliases)
        value_b = first_value(row_b, aliases)

        try:
            diff = (
                float(value_a) - float(value_b)
                if value_a is not None and value_b is not None
                else None
            )
        except (TypeError, ValueError):
            diff = None

        rows.append({
            "비교 항목": label,
            label_a: format_compare_value(label, value_a),
            label_b: format_compare_value(label, value_b),
            "차이 (A-B)": format_compare_value(label, diff),
        })

    return pd.DataFrame(rows)


def numeric_feature(feature_path: Path, aliases: list[str]) -> float | None:
    row = load_feature_row(feature_path)
    value = first_value(row, aliases)
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def show_compare_summary_cards(
    feature_path_a: Path,
    feature_path_b: Path,
    label_a: str,
    label_b: str,
) -> None:
    cards = [
        (
            "Knee Lift → FC",
            ["knee_lift_to_fc_time_s", "knee_lift_to_fc_time"],
            "s", 3,
        ),
        (
            "FC → Release 후보",
            ["fc_to_release_time_s", "fc_to_release_time"],
            "s",
            3,
        ),
    ]

    cols = st.columns(3)
    for col, (title, aliases, suffix, digits) in zip(cols, cards):
        a = numeric_feature(feature_path_a, aliases)
        b = numeric_feature(feature_path_b, aliases)
        with col:
            if a is None or b is None:
                st.metric(title, "—")
            else:
                st.metric(
                    title,
                    f"{a:.{digits}f}{suffix} / {b:.{digits}f}{suffix}",
                    delta=f"A-B {a - b:+.{digits}f}{suffix}",
                    help=f"앞 값: {label_a} · 뒤 값: {label_b}",
                )
    with cols[2]:
        knee_a = lead_knee_angle_change(str(load_feature_row(feature_path_a).get("pitch_id", "")), feature_path_a)
        knee_b = lead_knee_angle_change(str(load_feature_row(feature_path_b).get("pitch_id", "")), feature_path_b)
        if knee_a is None or knee_b is None:
            st.metric("FC → Release 앞무릎 변화", "—")
        else:
            st.metric("FC → Release 앞무릎 변화",
                      f"{knee_a:+.1f}° / {knee_b:+.1f}°",
                      help="앞 값: 투수 A · 뒤 값: 투수 B. +는 영상상 무릎이 더 펴짐을 뜻합니다."
                           "2D 투영값이므로 촬영 각도가 다르면 직접 우열을 비교할 수 없습니다.")


def _feature_number(
    row: pd.Series | None,
    aliases: list[str],
) -> float | None:
    if row is None:
        return None

    value = first_value(
        row,
        aliases,
    )

    if (
        value is None
        or pd.isna(value)
    ):
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def load_motion_graph_data(
    pitch_id: str,
) -> tuple[pd.DataFrame | None, str | None, str | None, str | None]:
    csv_path = find_normalized_csv(
        pitch_id
    )

    if csv_path is None:
        return None, None, None, None

    try:
        df = pd.read_csv(
            csv_path
        )
    except Exception:
        return None, None, None, None

    progress_col = normalized_axis_column(
        df
    )

    elbow_col = elbow_angle_column(
        df
    )

    wrist_col = wrist_speed_column(
        df
    )

    if progress_col is None:
        return None, None, None, None

    return (
        df,
        progress_col,
        elbow_col,
        wrist_col,
    )


def _motion_graph_event_values(
    feature_path: Path | None,
) -> dict[str, float | None]:
    row = None

    if (
        feature_path is not None
        and feature_path.exists()
    ):
        row = load_feature_row(
            feature_path
        )

    return {
        "knee_frame": _feature_number(
            row,
            [
                "knee_lift_frame",
                "maximum_knee_lift_frame",
                "lead_leg_lift_frame",
            ],
        ),
        "fc_frame": _feature_number(
            row,
            [
                "foot_contact_frame",
                "front_foot_contact_frame",
                "fc_frame",
            ],
        ),
        "release_frame": _feature_number(
            row,
            [
                "release_candidate_frame",
                "release_frame",
                "pose_release_candidate_frame",
            ],
        ),
        "fc_percent": _feature_number(
            row,
            [
                "foot_contact_percent",
                "fc_percent",
                "front_foot_contact_percent",
                "fc_position_percent",
                "fc_position_pct",
            ],
        ),
        "release_elbow": _feature_number(
            row,
            [
                "elbow_angle_release_deg",
                "release_elbow_angle",
                "elbow_angle_release",
                "throwing_elbow_angle_release",
            ],
        ),
        "release_wrist": _feature_number(
            row,
            [
                "release_wrist_speed_body_s",
                "release_wrist_speed",
                "wrist_speed_release",
            ],
        ),
        "peak_wrist": _feature_number(
            row,
            [
                "peak_wrist_speed_knee_to_release_body_s",
                "peak_wrist_speed_body_s",
                "peak_wrist_speed",
            ],
        ),
    }


def _add_pitch_event_lines(
    ax,
    fc_percent: float | None,
) -> None:
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.3,
        alpha=0.65,
        label="Knee Lift (0%)",
    )

    if fc_percent is not None:
        ax.axvline(
            fc_percent,
            linestyle="--",
            linewidth=1.3,
            alpha=0.65,
            label=f"Front Foot Contact ({fc_percent:.1f}%)",
        )

    ax.axvline(
        100,
        linestyle="--",
        linewidth=1.3,
        alpha=0.65,
        label="Release Candidate (100%)",
    )


def _show_motion_graph_note(
    title: str,
    body: str,
    metrics: list[tuple[str, str]],
) -> None:
    """
    Streamlit Markdown이 들여쓰기된 HTML을 코드 블록으로 해석하지 않도록
    HTML 문자열을 한 줄 형태로 조립한다.
    """
    metric_items = ""

    if metrics:
        for label, value in metrics:
            metric_items += (
                '<div style="min-width:150px;margin-right:1.15rem;'
                'margin-top:0.55rem;">'
                '<div style="color:#8f98a6;font-size:0.78rem;'
                'margin-bottom:0.15rem;">'
                f'{label}'
                '</div>'
                '<div style="color:#f5f7fa;font-size:1.00rem;'
                'font-weight:700;">'
                f'{value}'
                '</div>'
                '</div>'
            )

        metric_items = (
            '<div style="display:flex;flex-wrap:wrap;">'
            + metric_items
            + '</div>'
        )

    html = (
        '<div style="border:1px solid #29303b;'
        'border-radius:10px;padding:0.85rem 1rem;'
        'margin-top:-0.15rem;margin-bottom:1.35rem;'
        'background:#11161f;">'
        '<div style="font-size:0.90rem;font-weight:700;'
        'color:#dbe4ee;margin-bottom:0.35rem;">'
        f'{title}'
        '</div>'
        '<div style="color:#a8b0bc;font-size:0.86rem;'
        'line-height:1.55;">'
        f'{body}'
        '</div>'
        f'{metric_items}'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def show_web_motion_graphs(
    pitch_id: str,
    feature_path: Path | None = None,
) -> bool:
    """
    웹 화면용 동작 그래프.
    연구/아카이브용 전체 그래프와 달리
    Knee Lift 전후 맥락만 포함한 -20%~120% 구간을 보여준다.
    """
    (
        df,
        progress_col,
        elbow_col,
        wrist_col,
    ) = load_motion_graph_data(
        pitch_id
    )

    if (
        df is None
        or progress_col is None
    ):
        return False

    events = _motion_graph_event_values(
        feature_path
    )

    progress = pd.to_numeric(
        df[progress_col],
        errors="coerce",
    )

    web_mask = (
        progress.notna()
        & (progress >= -20)
        & (progress <= 120)
    )

    web_df = df.loc[
        web_mask
    ].copy()

    if web_df.empty:
        return False

    web_progress = pd.to_numeric(
        web_df[progress_col],
        errors="coerce",
    )

    fc_percent = events[
        "fc_percent"
    ]

    st.subheader(
        "동작 그래프"
    )

    st.caption(
        "웹에서는 핵심 투구 구간을 쉽게 읽을 수 있도록 "
        "Maximum Knee Lift = 0%, Release Candidate = 100%를 기준으로 "
        "-20%~120% 구간만 확대해 표시합니다."
    )

    # --------------------------------------------------------
    # 1) Elbow angle
    # --------------------------------------------------------
    if elbow_col is not None:
        elbow = pd.to_numeric(
            web_df[elbow_col],
            errors="coerce",
        )

        elbow_valid_col = validity_column(
            web_df,
            "elbow",
        )

        if elbow_valid_col is not None:
            valid = (
                web_df[elbow_valid_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                        "valid",
                    ]
                )
            )

            if web_df[
                elbow_valid_col
            ].dtype == bool:
                valid = web_df[
                    elbow_valid_col
                ].fillna(False)

            elbow = elbow.where(
                valid
            )

        fig, ax = plt.subplots(
            figsize=(10.8, 4.6)
        )

        ax.plot(
            web_progress,
            elbow,
            linewidth=2.0,
            label="Throwing Elbow Angle",
        )

        _add_pitch_event_lines(
            ax,
            fc_percent,
        )

        ax.set_xlim(
            -20,
            120,
        )

        ax.set_title(
            "Throwing Elbow Angle"
        )

        ax.set_xlabel(
            "Normalized Pitch Progress (%)"
        )

        ax.set_ylabel(
            "2D Projected Elbow Angle (degree)"
        )

        ax.grid(
            alpha=0.22
        )

        ax.legend(
            loc="best",
            fontsize=8.5,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(
            fig
        )

        elbow_metrics = []

        if events[
            "release_elbow"
        ] is not None:
            elbow_metrics.append(
                (
                    "Release 팔꿈치 각도",
                    f"{events['release_elbow']:.1f}°",
                )
            )

        if fc_percent is not None:
            elbow_metrics.append(
                (
                    "Front Foot Contact",
                    f"{fc_percent:.1f}%",
                )
            )

        _show_motion_graph_note(
            "이 그래프는 무엇을 보여주나요?",
            "투구 진행에 따라 투구팔의 팔꿈치가 굽혀지고 펴지는 변화를 보여줍니다. "
            "각도가 클수록 영상 평면에서 팔꿈치가 상대적으로 더 펴진 상태입니다. "
            "관절각은 단일 영상에서 계산한 2D 투영값입니다.",
            elbow_metrics,
        )

    return True


@st.cache_data(show_spinner=False)
def _cached_knee_lift_gifs(motion_path: str, feature_path: str,
                           versions: tuple[int, int]) -> tuple[bytes, bytes]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.core.knee_lift_animation import make_knee_lift_gifs
    return make_knee_lift_gifs(Path(motion_path), Path(feature_path))


def show_knee_lift_animation(pitch_id: str, feature_path: Path | None) -> None:
    motion = find_normalized_csv(pitch_id)
    if motion is None or feature_path is None or not feature_path.is_file():
        return
    st.subheader("앞다리 들기와 무릎 높이")
    st.caption("첫 프레임부터 릴리스 후보까지 앞다리의 2D 움직임과 무릎의 엉덩이 대비 높이를 보여줍니다. 높이는 몸통 길이로 나눈 값입니다.")
    try:
        leg_gif, graph_gif = _cached_knee_lift_gifs(
            str(motion), str(feature_path),
            (motion.stat().st_mtime_ns, feature_path.stat().st_mtime_ns),
        )
    except (OSError, ValueError, KeyError, IndexError) as exc:
        st.info(f"무릎 높이 애니메이션을 표시할 수 없습니다: {exc}")
        return
    left, right = st.columns(2)
    with left:
        st.image(leg_gif, use_container_width=True)
    with right:
        st.image(graph_gif, use_container_width=True)


def _video_at_speed(path: Path, pitch_id: str, speed: float) -> tuple[Path, str | None]:
    if speed == 1.0:
        return path, None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, OSError):
            return path, "배속 영상 생성에 ffmpeg가 필요합니다. 원본 속도로 재생합니다."
    target_dir = PIPELINE_OUTPUT_DIR / pitch_id / "playback"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{path.stem}_{path.stat().st_mtime_ns}_{speed:g}x.mp4"
    if not target.is_file():
        tmp = target.with_name(target.stem + ".part.mp4")
        cap = cv2.VideoCapture(str(path))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        cap.release()
        output_fps = source_fps * speed if source_fps > 0 else 30.0 * speed
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
             "-vf", f"setpts=(PTS-STARTPTS)/{speed:g}", "-r", f"{output_fps:.3f}",
             "-an", "-c:v", "libx264",
             "-preset", "ultrafast", "-crf", "27", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(tmp)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            return path, "배속 영상을 만들지 못해 원본 속도로 재생합니다."
        tmp.replace(target)
    return target, None


def show_analysis_video(pitch_id: str, video: Path, *, key: str) -> None:
    speed = st.select_slider("재생 속도", options=(0.5, 1.0, 1.5, 2.0),
                             value=1.0, format_func=lambda v: f"{v:g}×", key=key)
    with st.spinner("재생 영상을 준비 중입니다..."):
        playable, warning = _video_at_speed(video, pitch_id, speed)
    if warning:
        st.caption(warning)
    st.video(str(playable))


def find_analysis_video(pitch_id: str) -> Path | None:
    pitch_root = PIPELINE_OUTPUT_DIR / pitch_id
    video_candidates = [
        pitch_root / "videos" / f"{pitch_id}_analysis_overlay.mp4",
        pitch_root / "videos" / f"{pitch_id}_sample_analysis_overlay.mp4",
        pitch_root / "videos" / f"{pitch_id}_pitch_segment.mp4",
        pitch_root / "videos" / f"{pitch_id}_sample_pose.mp4",
    ]

    return next((candidate for candidate in video_candidates if candidate.is_file()), None)


def show_archived_visuals(
    pitch_id: str,
    feature_path: Path | None = None,
) -> None:
    video = find_analysis_video(pitch_id)

    if video is not None:
        st.subheader("분석 영상")
        st.caption(
            "검출된 주요 이벤트와 투구 동작을 영상에서 확인합니다."
        )

        # 페이지 전체 폭을 쓰지 않고 중앙에 적당한 크기로 표시
        left_spacer, video_col, right_spacer = st.columns([1.15, 3.7, 1.15])

        with video_col:
            show_analysis_video(pitch_id, video, key=f"analysis_video_{pitch_id}")

    show_knee_lift_animation(pitch_id, feature_path)



def render_header() -> None:
    # 상단 전체 영역에 MLB 로고를 큰 워터마크 배경처럼 배치
    st.markdown(
        """
        <style>
        .pitch-hero {
            position: relative;
            min-height: 225px;
            overflow: hidden;
            display: flex;
            align-items: center;
            padding: 20px 28px 18px 0;
            margin-top: -6px;
            margin-bottom: -18px;
            isolation: isolate;
        }

        .pitch-hero-logo {
            position: absolute;
            z-index: 0;
            width: 620px;
            max-width: 72vw;
            right: -45px;
            top: 50%;
            transform: translateY(-50%);
            opacity: 0.10;
            filter: saturate(0.90);
            pointer-events: none;
            user-select: none;
        }

        .pitch-hero-content {
            position: relative;
            z-index: 1;
            max-width: 690px;
            padding-left: 0;
        }

        .pitch-hero-title {
            margin: 0 0 12px 0;
            font-size: 3.0rem;
            line-height: 1.12;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--text-color, #17202a);
        }

        .pitch-hero-subtitle {
            color: var(--text-color, #17202a);
            opacity: 0.72;
            font-size: 1.0rem;
            line-height: 1.65;
            max-width: 650px;
        }

        @media (max-width: 900px) {
            .pitch-hero {
                min-height: 200px;
            }
            .pitch-hero-logo {
                width: 430px;
                right: -100px;
                opacity: 0.08;
            }
            .pitch-hero-title {
                font-size: 2.35rem;
            }
            .pitch-hero-content {
                max-width: 78%;
            }
        }
        </style>

        <div class="pitch-hero">
            <img
                class="pitch-hero-logo"
                src="https://www.mlbstatic.com/team-logos/league-on-dark/1.svg"
                alt="MLB background logo"
            />
            <div class="pitch-hero-content">
                <div class="pitch-hero-title">투구 동작 분석 시스템</div>
                <div class="pitch-hero-subtitle">
                    Pitching Motion Analysis Web App · 2D Pose 기반 이벤트 검출, 동작 정규화, Feature 비교
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Header
# ============================================================

render_header()


# ============================================================
# Tabs
# ============================================================

reference_tab, analyze_tab, compare_tab = st.tabs([
    "기준 투수",
    "새 투수 분석",
    "투수 비교",
])


# ============================================================
# Reference Pitchers
# ============================================================

with reference_tab:
    st.header("기준 투수 보기")
    st.caption("미리 분석해둔 기준 투수의 Feature, 영상, 그래프를 확인합니다.")

    feature_files = discover_feature_files()

    if not feature_files:
        st.info("현재 표시할 Feature CSV가 없습니다.")
    else:
        labels, label_to_id = selection_index_map(feature_files)
        selected_label = st.selectbox("기준 투수 선택", labels)
        selected = label_to_id[selected_label]

        feature_path = feature_files[selected]
        show_feature_dashboard(feature_path)
        show_event_frames(selected, feature_path)
        show_visual_cards_row(selected, feature_path)
        show_shoulder_opening_sequence(selected, feature_path)
        show_coaching_insights(selected, feature_path)
        show_archived_visuals(selected, feature_path)



def get_video_metadata(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0, 0.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    cap.release()

    if not fps or fps <= 0:
        fps = 30.0

    return total_frames, fps


def _shift_review_frame(
    state_key: str,
    delta: int,
    max_frame: int,
) -> None:
    current = int(
        st.session_state.get(
            state_key,
            0
        )
    )

    st.session_state[
        state_key
    ] = max(
        0,
        min(
            max_frame,
            current + delta
        )
    )


def _set_manual_event_frame(
    manual_key: str,
    review_key: str,
) -> None:
    st.session_state[
        manual_key
    ] = int(
        st.session_state.get(
            review_key,
            0
        )
    )


def _reset_manual_events(
    pitch_id: str,
    feature_path: Path | None,
) -> None:
    defaults = {
        "knee": -1,
        "fc": -1,
        "release": -1,
    }

    if (
        feature_path is not None
        and
        feature_path.exists()
    ):
        row = load_feature_row(
            feature_path
        )

        mapping = {
            "knee": [
                "knee_lift_frame",
                "maximum_knee_lift_frame",
                "lead_leg_lift_frame",
            ],
            "fc": [
                "foot_contact_frame",
                "front_foot_contact_frame",
                "fc_frame",
            ],
            "release": [
                "release_candidate_frame",
                "release_frame",
                "pose_release_candidate_frame",
            ],
        }

        for name, aliases in mapping.items():
            value = first_value(
                row,
                aliases
            )

            if (
                value is not None
                and
                not pd.isna(
                    value
                )
            ):
                defaults[name] = int(
                    float(
                        value
                    )
                )

    st.session_state[
        f"manual_knee_{pitch_id}"
    ] = defaults["knee"]

    st.session_state[
        f"manual_fc_{pitch_id}"
    ] = defaults["fc"]

    st.session_state[
        f"manual_release_{pitch_id}"
    ] = defaults["release"]


def show_manual_event_editor(
    pitch_id: str,
    video_path: Path,
    feature_path: Path | None = None,
) -> tuple[int, int, int]:
    total_frames, fps = get_video_metadata(
        video_path
    )

    if total_frames <= 0:
        st.warning(
            "영상 프레임 정보를 읽지 못했습니다."
        )
        return -1, -1, -1

    max_frame = total_frames - 1

    review_key = (
        f"review_frame_{pitch_id}"
    )

    knee_key = (
        f"manual_knee_{pitch_id}"
    )

    fc_key = (
        f"manual_fc_{pitch_id}"
    )

    release_key = (
        f"manual_release_{pitch_id}"
    )

    # 기존 분석 결과가 있으면 최초 기본값으로 사용
    if (
        knee_key not in st.session_state
        or
        fc_key not in st.session_state
        or
        release_key not in st.session_state
    ):
        _reset_manual_events(
            pitch_id,
            feature_path
        )

    if review_key not in st.session_state:
        initial = st.session_state.get(
            fc_key,
            0
        )

        if initial is None or initial < 0:
            initial = 0

        st.session_state[
            review_key
        ] = max(
            0,
            min(
                max_frame,
                int(
                    initial
                )
            )
        )

    st.markdown("### 이벤트 프레임 수동 보정")
    st.caption(
        "자동 검출이 부정확하면 영상을 프레임 단위로 확인한 뒤 "
        "현재 프레임을 Knee Lift / FC / Release로 직접 지정할 수 있습니다."
    )

    nav_cols = st.columns(
        [1, 1, 2.2, 1, 1]
    )

    with nav_cols[0]:
        st.button(
            "−10",
            key=f"review_m10_{pitch_id}",
            use_container_width=True,
            on_click=_shift_review_frame,
            args=(
                review_key,
                -10,
                max_frame,
            ),
        )

    with nav_cols[1]:
        st.button(
            "−1",
            key=f"review_m1_{pitch_id}",
            use_container_width=True,
            on_click=_shift_review_frame,
            args=(
                review_key,
                -1,
                max_frame,
            ),
        )

    with nav_cols[3]:
        st.button(
            "+1",
            key=f"review_p1_{pitch_id}",
            use_container_width=True,
            on_click=_shift_review_frame,
            args=(
                review_key,
                1,
                max_frame,
            ),
        )

    with nav_cols[4]:
        st.button(
            "+10",
            key=f"review_p10_{pitch_id}",
            use_container_width=True,
            on_click=_shift_review_frame,
            args=(
                review_key,
                10,
                max_frame,
            ),
        )

    current_frame = st.slider(
        "Frame",
        min_value=0,
        max_value=max_frame,
        key=review_key,
    )

    image_rgb, actual_frame, _ = read_video_frame(
        str(
            video_path
        ),
        int(
            current_frame
        ),
    )

    current_time = (
        float(
            actual_frame
        )
        / fps
        if fps > 0
        else 0.0
    )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:1.05rem;
            font-weight:700;
            margin:0.25rem 0 0.7rem 0;
        ">
            Frame {actual_frame} / {max_frame}
            &nbsp; · &nbsp;
            {current_time:.3f} s
            &nbsp; · &nbsp;
            {fps:.2f} FPS
        </div>
        """,
        unsafe_allow_html=True,
    )

    if image_rgb is not None:
        preview_left, preview_center, preview_right = st.columns(
            [1.15, 3.7, 1.15]
        )

        with preview_center:
            st.image(
                image_rgb,
                use_container_width=True,
            )

    assign_cols = st.columns(
        3
    )

    with assign_cols[0]:
        st.button(
            "현재 프레임 → Knee Lift",
            key=f"set_knee_{pitch_id}",
            use_container_width=True,
            on_click=_set_manual_event_frame,
            args=(
                knee_key,
                review_key,
            ),
        )

    with assign_cols[1]:
        st.button(
            "현재 프레임 → FC",
            key=f"set_fc_{pitch_id}",
            use_container_width=True,
            on_click=_set_manual_event_frame,
            args=(
                fc_key,
                review_key,
            ),
        )

    with assign_cols[2]:
        st.button(
            "현재 프레임 → Release",
            key=f"set_release_{pitch_id}",
            use_container_width=True,
            on_click=_set_manual_event_frame,
            args=(
                release_key,
                review_key,
            ),
        )

    selected_cols = st.columns(
        3
    )

    selected_cols[0].metric(
        "Knee Lift",
        (
            f"{int(st.session_state[knee_key])}F"
            if int(st.session_state[knee_key]) >= 0
            else "자동"
        ),
    )

    selected_cols[1].metric(
        "Front Foot Contact",
        (
            f"{int(st.session_state[fc_key])}F"
            if int(st.session_state[fc_key]) >= 0
            else "자동"
        ),
    )

    selected_cols[2].metric(
        "Release Candidate",
        (
            f"{int(st.session_state[release_key])}F"
            if int(st.session_state[release_key]) >= 0
            else "자동"
        ),
    )

    if feature_path is not None and feature_path.exists():
        st.button(
            "자동 검출값으로 되돌리기",
            key=f"reset_manual_events_{pitch_id}",
            on_click=_reset_manual_events,
            args=(
                pitch_id,
                feature_path,
            ),
        )

    return (
        int(
            st.session_state[
                knee_key
            ]
        ),
        int(
            st.session_state[
                fc_key
            ]
        ),
        int(
            st.session_state[
                release_key
            ]
        ),
    )


# ============================================================
# Analyze New Pitcher
# ============================================================

with analyze_tab:
    st.header("새 투수 분석")
    st.caption(
        "새 영상을 업로드해 자동 분석을 실행하고, "
        "필요하면 주요 이벤트 프레임을 직접 보정합니다."
    )

    left, right = st.columns(
        [1, 1]
    )

    with left:
        pitcher_name = st.text_input(
            "투수 이름",
            value="새 투수"
        )

        pitch_id_input = st.text_input(
            "Pitch ID (영문/숫자 권장)",
            value="uploaded_01"
        )

        throwing_side = st.radio(
            "투구 손",
            [
                "right",
                "left"
            ],
            horizontal=True,
            format_func=
                lambda x:
                "우완 (Right)"
                if x == "right"
                else "좌완 (Left)",
        )

        st.caption(
            "Arm Slot은 Sports2D Stage 3에서 자동 분류됩니다. "
            "사용자가 직접 선택하지 않습니다."
        )
        arm_slot = ""

    with right:
        uploaded = st.file_uploader(
            "투구 영상 업로드",
            type=[
                "mp4",
                "mov",
                "avi",
                "mkv"
            ],
            help=
                "지원 형식: MP4, MOV, AVI, MKV",
        )

    pitch_id = safe_pitch_id(
        pitch_id_input
    )

    if (
        pitch_id
        != pitch_id_input
        .strip()
        .lower()
    ):
        st.caption(
            f"저장에 사용할 Pitch ID: "
            f"`{pitch_id}`"
        )

    video_path = None

    # 업로드 즉시 raw_videos에 저장.
    # Streamlit rerun 뒤에도 프레임 검토가 유지된다.
    if uploaded is not None:
        RAW_VIDEO_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        suffix = (
            Path(
                uploaded.name
            )
            .suffix
            .lower()
            or ".mp4"
        )

        video_filename = (
            f"{pitch_id}{suffix}"
        )

        video_path = (
            RAW_VIDEO_DIR
            / video_filename
        )

        uploaded_bytes = (
            uploaded.getvalue()
        )

        if not video_path.exists() or video_path.read_bytes() != uploaded_bytes:
            video_path.write_bytes(
                uploaded_bytes
            )
            st.session_state[f"upload_changed_{pitch_id}"] = True
            st.session_state[f"roi_changed_{pitch_id}"] = True

    else:
        # 이미 한 번 분석한 pitch라면 기존 raw video 재사용
        manifest = read_manifest()

        if (
            not manifest.empty
            and
            "pitch_id"
            in manifest.columns
        ):
            match = manifest.loc[
                manifest[
                    "pitch_id"
                ]
                .astype(str)
                == pitch_id
            ]

            if (
                len(match) > 0
                and
                "video_file"
                in match.columns
            ):
                existing_name = str(
                    match.iloc[0][
                        "video_file"
                    ]
                )

                candidate = (
                    RAW_VIDEO_DIR
                    / existing_name
                )

                if candidate.exists():
                    video_path = candidate

    existing_feature_path = (
        find_feature_csv_for_pitch(
            pitch_id
        )
    )

    # --------------------------------------------------------
    # Initial Analysis
    # --------------------------------------------------------

    stage2_ready = canonical_stage2_ready(pitch_id)
    needs_prepare = bool(st.session_state.get(f"upload_changed_{pitch_id}", False)) or not stage2_ready
    roi_json = SPORTS2D_DIR / "input" / f"{pitch_id}_roi.json"
    roi_crop = SPORTS2D_DIR / "cropped" / f"{pitch_id}_roi.mp4"
    roi_ready = roi_json.is_file() and roi_crop.is_file() and not st.session_state.get(f"roi_changed_{pitch_id}", False)

    if video_path is not None and needs_prepare:
        st.info(
            "먼저 투수 영역을 드래그해 지정하세요. 별도 OpenCV 창에서 첫 프레임의 투수 몸을 선택하고 Enter를 누르면 크롭 영상이 생성됩니다."
        )
        if st.button("투수 영역 드래그로 선택", use_container_width=True):
            with st.spinner("투수 영역 선택 창을 확인하세요..."):
                roi_success, roi_log = select_pitcher_roi(pitch_id, video_path)
            st.session_state[f"roi_log_{pitch_id}"] = roi_log
            if roi_success:
                st.session_state[f"roi_changed_{pitch_id}"] = False
                roi_ready = True
                st.success("투수 영역 크롭이 완료됐습니다. 이제 분석을 실행하세요.")
            else:
                st.error("투수 영역 선택에 실패했습니다. 아래 로그를 확인하세요.")
                st.code(roi_log)

    analyze_clicked = st.button(
        "Canonical 분석 실행",
        type="primary",
        use_container_width=True,
        disabled=(
            video_path is None
            or (needs_prepare and not roi_ready)
        ),
    )

    if (
        analyze_clicked
        and
        video_path is not None
    ):
        # 최초 분석에서는 기존 수동값을 비운다.
        manifest_row = {
            "enabled":
                "1",

            "pitch_id":
                pitch_id,

            "pitcher_name":
                pitcher_name.strip()
                or pitch_id,

            "throwing_side":
                throwing_side,

            "arm_slot":
                arm_slot,

            "video_file":
                video_path.name,

            "manual_knee_lift_frame":
                "",

            "manual_fc_frame":
                "",

            "manual_release_frame":
                "",
        }

        upsert_manifest_row(
            manifest_row
        )

        progress = st.progress(
            0,
            text="분석 준비 중..."
        )

        progress.progress(
            10,
            text=
                "영상 저장 및 manifest 등록 완료"
        )

        changed = bool(st.session_state.get(f"upload_changed_{pitch_id}", False))
        stage2_final = (SPORTS2D_DIR / "stage2" / pitch_id / "pose_csv"
                        / f"{pitch_id}_sports2d_wrist_motion_validated.csv")
        stage3_summary = (SPORTS2D_DIR / "stage3" / pitch_id
                          / "stage3_events_final_summary.json")
        # A previous attempt may have completed expensive Sports2D stages but
        # stopped while building events (e.g. a missing automatic Knee Lift).
        # Reuse them only if both are newer than the currently selected ROI.
        resume_events = (needs_prepare and roi_crop.is_file()
                         and stage2_final.is_file() and stage3_summary.is_file()
                         and stage2_final.stat().st_mtime_ns >= roi_crop.stat().st_mtime_ns
                         and stage3_summary.stat().st_mtime_ns >= roi_crop.stat().st_mtime_ns)
        prepare_log = ""
        prepared = True
        if needs_prepare and not resume_events:
            with st.spinner("Sports2D 관절 추출 및 Stage 2 입력 준비 중..."):
                prepared, prepare_log = prepare_uploaded_video(
                    pitch_id, video_path, force=changed,
                )
        if prepared:
            with st.spinner("Canonical Pitch Analysis Pipeline을 실행 중입니다..."):
                success, pipeline_log = run_analysis(
                    pitch_id, from_step=3 if resume_events else 1,
                    to_step=6, force=changed or resume_events,
                )
        else:
            success, pipeline_log = False, "입력 준비에 실패하여 분석을 시작하지 않았습니다."
        log = (prepare_log + "\n" + pipeline_log).strip()
        if success:
            st.session_state[f"upload_changed_{pitch_id}"] = False

        progress.progress(
            100,
            text="분석 완료" if success else "분석 오류: 아래 로그를 확인하세요"
        )

        st.session_state[
            f"analysis_log_{pitch_id}"
        ] = log

        st.session_state[
            f"analysis_success_{pitch_id}"
        ] = success

        # 새 자동 결과를 manual editor 기본값으로 다시 읽게 함
        for key in [
            f"manual_knee_{pitch_id}",
            f"manual_fc_{pitch_id}",
            f"manual_release_{pitch_id}",
            f"review_frame_{pitch_id}",
        ]:
            st.session_state.pop(
                key,
                None
            )

        if success:
            st.success(
                f"`{pitch_id}` 분석이 완료되었습니다."
            )
        else:
            st.error(
                "Pipeline 실행 중 오류가 발생했습니다."
            )

        existing_feature_path = (
            find_feature_csv_for_pitch(
                pitch_id
            )
        )

    # --------------------------------------------------------
    # Persisted Analysis Result
    # --------------------------------------------------------

    if (video_path is not None and video_path.is_file() and roi_ready
            and existing_feature_path is not None and existing_feature_path.is_file()):
        with st.expander("스켈레톤 위치 복구"):
            st.caption("크롭 영상의 기존 Sports2D TRC를 재사용해 투수와 관절의 위치를 다시 맞춥니다.")
            if st.button("좌표 재정렬 후 분석 갱신", key=f"repair_pose_{pitch_id}"):
                with st.spinner("좌표를 확인하고 분석 결과를 다시 만드는 중..."):
                    repaired, repair_log = prepare_uploaded_video(
                        pitch_id, video_path, force=True, reuse_trc=True,
                    )
                    success, pipeline_log = (
                        run_analysis(pitch_id, from_step=1, to_step=6, force=True)
                        if repaired else (False, "좌표 재정렬에 실패했습니다.")
                    )
                if success:
                    st.session_state[f"upload_changed_{pitch_id}"] = False
                    st.success("스켈레톤 위치와 분석 결과를 갱신했습니다.")
                    st.rerun()
                else:
                    st.error("복구에 실패했습니다. 아래 로그를 확인하세요.")
                    st.code((repair_log + "\n" + pipeline_log).strip())

    if (
        existing_feature_path is not None
        and
        existing_feature_path.exists()
    ):
        st.divider()
        st.subheader("분석 결과")

        show_feature_dashboard(
            existing_feature_path
        )

        show_event_frames(
            pitch_id,
            existing_feature_path
        )

        show_visual_cards_row(
            pitch_id,
            existing_feature_path
        )

        show_shoulder_opening_sequence(
            pitch_id,
            existing_feature_path
        )

        show_coaching_insights(
            pitch_id,
            existing_feature_path
        )

        show_archived_visuals(
            pitch_id,
            existing_feature_path
        )

        # ----------------------------------------------------
        # Manual Review
        # ----------------------------------------------------
        if video_path is not None and video_path.exists():
            st.divider()
            st.info(
                "수동 이벤트 보정 UI는 canonical analysis_events.json과의 "
                "override 연결을 정리한 뒤 다시 활성화합니다. "
                "현재 결과 화면은 자동 검출값을 기준으로 표시합니다."
            )

    log = st.session_state.get(
        f"analysis_log_{pitch_id}",
        ""
    )

    success_state = st.session_state.get(
        f"analysis_success_{pitch_id}",
        True
    )

    if log:
        with st.expander(
            "Pipeline 로그 보기",
            expanded=
                not success_state,
        ):
            st.code(
                log,
                language="text"
            )


# ============================================================
# Compare Shoulder Opening
# ============================================================

def show_compare_shoulder_opening(
    pitch_a: str,
    pitch_b: str,
    label_a: str,
    label_b: str,
    feature_path_a: Path,
    feature_path_b: Path,
) -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{label_a}**")
        show_interactive_3d_viewer(pitch_a, feature_path_a, key=f"compare_3d_a_{pitch_a}")
    with col_b:
        st.markdown(f"**{label_b}**")
        show_interactive_3d_viewer(pitch_b, feature_path_b, key=f"compare_3d_b_{pitch_b}")


# ============================================================
# Compare
# ============================================================

with compare_tab:
    st.header("투수 비교")
    st.caption("선택한 두 투수의 핵심 Feature와 정규화 동작 곡선을 비교합니다.")

    feature_files = discover_feature_files()

    if len(feature_files) < 2:
        st.info("비교하려면 최소 2개의 Feature 결과가 필요합니다.")
    else:
        labels, label_to_id = selection_index_map(feature_files)

        c1, c2 = st.columns(2)
        with c1:
            label_a = st.selectbox(
                "투수 A",
                labels,
                index=0,
                key="compare_a",
            )
        with c2:
            default_b = 1 if len(labels) > 1 else 0
            label_b = st.selectbox(
                "투수 B",
                labels,
                index=default_b,
                key="compare_b",
            )

        pitch_a = label_to_id[label_a]
        pitch_b = label_to_id[label_b]

        if pitch_a == pitch_b:
            st.warning("서로 다른 두 투수를 선택하세요.")
        else:
            st.subheader("분석 영상 비교")
            video_a, video_b = st.columns(2)
            with video_a:
                st.markdown(f"**{label_a}**")
                analysis_a = find_analysis_video(pitch_a)
                if analysis_a:
                    show_analysis_video(pitch_a, analysis_a, key=f"analysis_compare_a_{pitch_a}")
                else:
                    st.info("분석 영상이 없습니다.")
            with video_b:
                st.markdown(f"**{label_b}**")
                analysis_b = find_analysis_video(pitch_b)
                if analysis_b:
                    show_analysis_video(pitch_b, analysis_b, key=f"analysis_compare_b_{pitch_b}")
                else:
                    st.info("분석 영상이 없습니다.")

            st.subheader("투구 팔 움직임")
            show_arm_motion_compare(
                pitch_a, pitch_b, label_a, label_b,
                feature_files[pitch_a], feature_files[pitch_b],
            )

            st.subheader("앞다리 들기 움직임")
            show_leg_lift_compare(
                pitch_a, pitch_b, label_a, label_b,
                feature_files[pitch_a], feature_files[pitch_b],
            )

            st.subheader("핵심 차이")
            show_compare_summary_cards(
                feature_files[pitch_a],
                feature_files[pitch_b],
                label_a,
                label_b,
            )

            show_easy_compare_summary(
                pitch_a,
                pitch_b,
                label_a,
                label_b,
            )

            st.divider()
            show_compare_shoulder_opening(
                pitch_a,
                pitch_b,
                label_a,
                label_b,
                feature_files[pitch_a],
                feature_files[pitch_b],
            )

            st.divider()
            st.subheader("주요 Feature 비교")

            compare_df = build_feature_compare_table(
                feature_files[pitch_a],
                feature_files[pitch_b],
                label_a,
                label_b,
            )

            st.dataframe(
                compare_df,
                use_container_width=True,
                hide_index=True,
            )
