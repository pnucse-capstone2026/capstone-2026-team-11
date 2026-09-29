import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# YOLO Pose COCO 17 Keypoints
# ============================================================

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# ============================================================
# Skeleton 연결 관계
# ============================================================

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),

    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),

    ("left_hip", "right_hip"),

    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


# ============================================================
# 좌표 읽기
# ============================================================

def get_clean_point(row, joint_name):
    x_col = f"{joint_name}_x_clean"
    y_col = f"{joint_name}_y_clean"

    if (
        x_col not in row.index
        or y_col not in row.index
    ):
        return None

    x = row[x_col]
    y = row[y_col]

    if pd.isna(x) or pd.isna(y):
        return None

    return (
        int(round(float(x))),
        int(round(float(y)))
    )


def align_overlay_pose_to_roi(df, roi_json):
    """Resolve crop/global coordinates at rendering time for uploaded videos.

    The selected ROI is in original-video pixels. Only the drawing coordinates
    are shifted; event times and calculated features stay as they were.
    """
    if roi_json is None:
        return df
    with open(roi_json, encoding="utf-8") as stream:
        roi = json.load(stream)
    selected, expanded = roi["selected_roi"], roi["expanded_roi"]
    dx, dy = float(expanded["offset_x"]), float(expanded["offset_y"])
    target = np.array([
        float(selected["x"]) + float(selected["width"]) / 2,
        float(selected["y"]) + float(selected["height"]) / 2,
    ])
    scale = np.array([
        max(float(selected["width"]), 1),
        max(float(selected["height"]), 1),
    ])
    centers = []
    for _, row in df.head(15).iterrows():
        points = []
        for joint in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            x = pd.to_numeric(row.get(f"{joint}_x_clean"), errors="coerce")
            y = pd.to_numeric(row.get(f"{joint}_y_clean"), errors="coerce")
            if np.isfinite(x) and np.isfinite(y):
                points.append((x, y))
        if len(points) >= 2:
            centers.append(np.mean(points, axis=0))
    if not centers:
        raise ValueError("투수 ROI와 비교할 몸통 관절 좌표가 없습니다.")
    center = np.median(centers, axis=0)
    offsets = ((0.0, 0.0), (-dx, -dy), (dx, dy))
    errors = [np.linalg.norm((center + offset - target) / scale) for offset in offsets]
    index = int(np.argmin(errors))
    if errors[index] > 1.5:
        raise ValueError("스켈레톤이 선택한 투수 영역과 맞지 않습니다. ROI와 인물 추적을 확인하세요.")
    shift_x, shift_y = offsets[index]
    print(f"[OVERLAY ROI] 좌표 보정 x={shift_x:g}, y={shift_y:g} (오차={errors[index]:.2f})")
    if index == 0:
        return df
    df = df.copy()
    for column in df.columns:
        if column.endswith("_x_clean"):
            df[column] = pd.to_numeric(df[column], errors="coerce") + shift_x
        elif column.endswith("_y_clean"):
            df[column] = pd.to_numeric(df[column], errors="coerce") + shift_y
    return df


# ============================================================
# Skeleton 그리기
# ============================================================

def draw_skeleton(
    frame,
    row,
    throwing_side="left"
):
    points = {}

    for joint in KEYPOINT_NAMES:
        point = get_clean_point(
            row,
            joint
        )

        points[joint] = point

    # --------------------------------------
    # 연결선
    # --------------------------------------

    for joint_a, joint_b in SKELETON_CONNECTIONS:

        point_a = points.get(
            joint_a
        )

        point_b = points.get(
            joint_b
        )

        if (
            point_a is None
            or point_b is None
        ):
            continue

        cv2.line(
            frame,
            point_a,
            point_b,
            (0, 255, 0),
            2
        )

    # --------------------------------------
    # 관절 점
    # --------------------------------------

    for joint, point in points.items():

        if point is None:
            continue

        # 투구팔은 조금 크게 표시
        if joint.startswith(
            throwing_side
        ):
            radius = 6
        else:
            radius = 4

        cv2.circle(
            frame,
            point,
            radius,
            (0, 255, 255),
            -1
        )

    return frame


# ============================================================
# 반투명 정보 패널
# ============================================================

def draw_info_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.55
):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# 텍스트 유틸
# ============================================================

def put_text(
    frame,
    text,
    x,
    y,
    scale=0.7,
    thickness=2
):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 이벤트 배너
# ============================================================

def draw_event_banner(
    frame,
    event_name,
    confidence=""
):
    if (
        event_name is None
        or event_name == ""
        or pd.isna(event_name)
    ):
        return

    height, width = frame.shape[:2]

    banner_height = 75

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, banner_height),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    text = event_name

    if (
        confidence is not None
        and confidence != ""
        and not pd.isna(confidence)
    ):
        text += f" [{confidence}]"

    cv2.putText(
        frame,
        text,
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )



# ============================================================
# Final Event Helpers
# ============================================================

EVENT_DISPLAY_HALF_WINDOW = 4


def get_first_numeric_metadata(df, candidates):
    """
    전체 행에 반복 저장된 metadata 컬럼에서 첫 유효 숫자를 반환.
    """
    for column in candidates:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) > 0:
            return int(round(float(values.iloc[0])))

    return None


def find_first_event_frame(
    df,
    column_candidates,
    keyword
):
    """
    이벤트 문자열 컬럼에서 keyword가 들어간 첫 frame을 찾는 fallback.
    """
    for column in column_candidates:
        if column not in df.columns:
            continue

        series = (
            df[column]
            .fillna("")
            .astype(str)
        )

        mask = series.str.contains(
            keyword,
            case=False,
            regex=False
        )

        rows = df.loc[mask]

        if len(rows) > 0:
            return int(rows.iloc[0]["frame"])

    return None


def get_final_event_frames(df):
    """
    STEP 14 Motion Normalization metadata를 우선 사용하고,
    없으면 STEP 12/이벤트 문자열을 fallback으로 사용.
    """
    knee_lift_frame = get_first_numeric_metadata(
        df,
        [
            "normalization_knee_lift_frame",
            "final_knee_lift_frame_v10",
        ]
    )

    if knee_lift_frame is None:
        knee_lift_frame = find_first_event_frame(
            df,
            [
                "pitch_event_v10",
                "pitch_event",
            ],
            "Maximum Knee Lift"
        )

    foot_contact_frame = get_first_numeric_metadata(
        df,
        [
            "normalization_fc_frame",
            "final_fc_frame_v10",
        ]
    )

    if foot_contact_frame is None:
        foot_contact_frame = find_first_event_frame(
            df,
            [
                "pitch_event_v10",
                "pitch_event",
            ],
            "Front Foot Contact"
        )

    release_frame = get_first_numeric_metadata(
        df,
        [
            "normalization_release_frame",
        ]
    )

    if release_frame is None:
        release_frame = find_first_event_frame(
            df,
            [
                "release_event_simple",
                "pitch_event_v10",
                "pitch_event",
            ],
            "Release Candidate"
        )

    return {
        "MAX KNEE LIFT": knee_lift_frame,
        "FRONT FOOT CONTACT": foot_contact_frame,
        "RELEASE CANDIDATE": release_frame,
    }


def get_release_confidence(df):
    """
    Release confidence metadata를 찾는다.
    """
    for column in [
        "normalization_release_confidence",
        "release_confidence_simple",
        "release_confidence",
    ]:
        if column not in df.columns:
            continue

        values = (
            df[column]
            .dropna()
            .astype(str)
        )

        values = values[
            values.str.strip() != ""
        ]

        if len(values) > 0:
            return str(values.iloc[0]).strip()

    return ""


def get_active_final_event(
    frame_number,
    event_frames,
    half_window=EVENT_DISPLAY_HALF_WINDOW
):
    """
    이벤트 기준 ±half_window frame 동안 이벤트명을 표시.
    여러 이벤트 window가 겹치면 가장 가까운 이벤트를 선택.
    """
    candidates = []

    for event_name, event_frame in event_frames.items():
        if event_frame is None:
            continue

        distance = abs(
            int(frame_number)
            - int(event_frame)
        )

        if distance <= half_window:
            candidates.append(
                (
                    distance,
                    event_name,
                    int(event_frame)
                )
            )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item: item[0]
    )

    _, event_name, event_frame = candidates[0]

    return event_name, event_frame


def draw_final_event_banner(
    frame,
    event_name,
    event_frame,
    current_frame,
    confidence=""
):
    """
    발표/웹 시연용 큰 이벤트 배너.
    이벤트 전후 몇 프레임 동안 유지한다.
    """
    if event_name is None:
        return frame

    height, width = frame.shape[:2]

    banner_h = max(
        90,
        int(height * 0.11)
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, banner_h),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame
    )

    title_scale = max(
        0.85,
        min(
            1.35,
            width / 1150.0
        )
    )

    title_thickness = max(
        2,
        int(round(title_scale * 2.5))
    )

    title = event_name

    if (
        event_name == "RELEASE CANDIDATE"
        and confidence
    ):
        title += f"  [{confidence.upper()}]"

    cv2.putText(
        frame,
        title,
        (30, int(banner_h * 0.57)),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        title_thickness,
        cv2.LINE_AA
    )

    offset = int(current_frame) - int(event_frame)

    if offset == 0:
        timing_text = f"EVENT FRAME  {event_frame}"
    elif offset < 0:
        timing_text = (
            f"EVENT FRAME  {event_frame}  |  "
            f"{abs(offset)} frame before"
        )
    else:
        timing_text = (
            f"EVENT FRAME  {event_frame}  |  "
            f"{offset} frame after"
        )

    cv2.putText(
        frame,
        timing_text,
        (32, int(banner_h * 0.86)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, title_scale * 0.48),
        (200, 210, 220),
        max(1, title_thickness - 1),
        cv2.LINE_AA
    )

    return frame


# ============================================================
# Browser-compatible MP4 helper
# ============================================================

def convert_video_for_web(
    input_path,
    output_path
):
    """
    OpenCV의 mp4v 결과를 브라우저 호환성이 높은
    H.264 + yuv420p MP4로 변환한다.

    ffmpeg가 없으면 False를 반환한다.
    """
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        print(
            "[경고] ffmpeg를 찾지 못했습니다. "
            "브라우저용 H.264 변환을 건너뜁니다."
        )
        return False

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("[경고] ffmpeg H.264 변환 실패")
        if result.stderr:
            print(result.stderr[-2000:])
        return False

    return (
        os.path.exists(output_path)
        and os.path.getsize(output_path) > 0
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def create_analysis_overlay(
    video_path,
    csv_path,
    output_video_path,
    throwing_side="left",
    roi_json=None,
):
    # --------------------------------------------------------
    # 1. 파일 확인
    # --------------------------------------------------------

    if not os.path.exists(
        video_path
    ):
        print(
            f"[오류] 원본 영상이 없습니다: "
            f"{video_path}"
        )
        return

    if not os.path.exists(
        csv_path
    ):
        print(
            f"[오류] CSV 파일이 없습니다: "
            f"{csv_path}"
        )
        return

    df = pd.read_csv(
        csv_path
    )
    df = align_overlay_pose_to_roi(df, roi_json)

    final_event_frames = get_final_event_frames(
        df
    )

    release_confidence_final = get_release_confidence(
        df
    )

    print()
    print(
        "===== [STEP 10] Analysis Overlay ====="
    )
    print(
        f"영상        : {video_path}"
    )
    print(
        f"CSV         : {csv_path}"
    )
    print(
        f"분석 프레임 : {len(df)}"
    )
    print(
        f"투구팔      : {throwing_side}"
    )
    print(
        "======================================"
    )

    print()
    print("----- Final Event Overlay -----")

    for event_name, event_frame in final_event_frames.items():
        print(
            f"{event_name:22s}: "
            f"{event_frame if event_frame is not None else 'N/A'}"
        )

    print(
        f"Event label window     : "
        f"±{EVENT_DISPLAY_HALF_WINDOW} frames"
    )

    # --------------------------------------------------------
    # 2. 영상 열기
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        print(
            "[오류] 영상을 열 수 없습니다."
        )
        return

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_video_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"영상 FPS     : {fps:.2f}"
    )

    print(
        f"영상 프레임  : {total_video_frames}"
    )

    # --------------------------------------------------------
    # 3. 출력 영상
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_video_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # OpenCV mp4v는 Chrome/Edge에서 0:00으로 보일 수 있으므로
    # 우선 임시 MP4를 만든 뒤 마지막에 H.264로 변환한다.
    output_video_path = str(
        output_video_path
    )

    output_path_obj = Path(
        output_video_path
    )

    temp_video_path = (
        output_path_obj.parent
        / (
            output_path_obj.stem
            + "_mp4v_temp.mp4"
        )
    )

    if temp_video_path.exists():
        temp_video_path.unlink()

    writer = cv2.VideoWriter(
        str(temp_video_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print(
            "[오류] 출력 영상을 만들 수 없습니다."
        )
        cap.release()
        return

    # --------------------------------------------------------
    # 4. CSV frame → row mapping
    # --------------------------------------------------------

    frame_map = {}

    for idx, row in df.iterrows():

        frame_number = int(
            row["frame"]
        )

        frame_map[
            frame_number
        ] = idx

    if len(frame_map) == 0:
        print(
            "[오류] CSV frame 데이터가 없습니다."
        )
        cap.release()
        writer.release()
        return

    min_analysis_frame = min(
        frame_map.keys()
    )

    max_analysis_frame = max(
        frame_map.keys()
    )

    print()
    print(
        f"분석 범위    : "
        f"{min_analysis_frame}"
        f" ~ "
        f"{max_analysis_frame}"
    )

    # --------------------------------------------------------
    # 5. 영상 프레임 처리
    # --------------------------------------------------------

    video_frame_idx = 0

    analyzed_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # --------------------------------------
        # 분석 구간인지 확인
        # --------------------------------------

        if video_frame_idx in frame_map:

            row_idx = (
                frame_map[
                    video_frame_idx
                ]
            )

            row = df.iloc[
                row_idx
            ]

            # ==================================
            # Skeleton
            # ==================================

            frame = draw_skeleton(
                frame,
                row,
                throwing_side=throwing_side
            )

            # ==================================
            # 좌측 정보 패널
            # ==================================

            panel_width = min(
                520,
                width - 20
            )

            draw_info_panel(
                frame,
                10,
                90,
                panel_width,
                355,
                alpha=0.58
            )

            current_frame = int(
                row["frame"]
            )

            timestamp = float(
                row["timestamp"]
            )

            phase = (
                row.get(
                    "pitch_phase",
                    ""
                )
            )

            event_name = (
                row.get(
                    "pitch_event",
                    ""
                )
            )

            event_conf = (
                row.get(
                    "event_confidence",
                    ""
                )
            )

            # ----------------------------------
            # 팔꿈치 각도
            # ----------------------------------

            elbow_angle = row.get(
                "throwing_elbow_angle_filtered",
                np.nan
            )

            elbow_valid = row.get(
                "throwing_elbow_angle_valid",
                False
            )

            # ----------------------------------
            # 손목 속도
            # ----------------------------------

            wrist_speed = row.get(
                "wrist_relative_speed_body_s_validated",
                np.nan
            )

            wrist_speed_valid = row.get(
                "wrist_speed_valid",
                False
            )

            # ==================================
            # 텍스트
            # ==================================

            y = 125
            gap = 34

            put_text(
                frame,
                f"Frame : {current_frame}",
                30,
                y
            )

            y += gap

            put_text(
                frame,
                f"Time  : {timestamp:.3f} s",
                30,
                y
            )

            y += gap

            put_text(
                frame,
                f"Phase : {phase}",
                30,
                y
            )

            y += gap

            # 팔꿈치
            if (
                bool(elbow_valid)
                and np.isfinite(
                    elbow_angle
                )
            ):

                elbow_text = (
                    f"Elbow : "
                    f"{elbow_angle:.1f} deg"
                )

            else:

                elbow_text = (
                    "Elbow : invalid"
                )

            put_text(
                frame,
                elbow_text,
                30,
                y
            )

            y += gap

            # 손목 속도
            if (
                bool(wrist_speed_valid)
                and np.isfinite(
                    wrist_speed
                )
            ):

                wrist_text = (
                    f"Wrist Speed : "
                    f"{wrist_speed:.2f} body/s"
                )

            else:

                wrist_text = (
                    "Wrist Speed : invalid"
                )

            put_text(
                frame,
                wrist_text,
                30,
                y
            )

            y += gap

            stride_leg = row.get(
                "stride_leg",
                ""
            )

            put_text(
                frame,
                f"Stride Leg : {stride_leg}",
                30,
                y
            )

            # ==================================
            # Final Event Banner
            #
            # Knee Lift / FC / Release Candidate를
            # 이벤트 전후 몇 프레임 동안 크게 표시.
            # ==================================

            active_event_name, active_event_frame = (
                get_active_final_event(
                    current_frame,
                    final_event_frames
                )
            )

            if active_event_name is not None:
                banner_confidence = (
                    release_confidence_final
                    if active_event_name
                    == "RELEASE CANDIDATE"
                    else ""
                )

                draw_final_event_banner(
                    frame,
                    active_event_name,
                    active_event_frame,
                    current_frame,
                    banner_confidence
                )

            analyzed_count += 1

        else:
            # ----------------------------------
            # 분석 외 구간
            # ----------------------------------

            draw_info_panel(
                frame,
                10,
                20,
                330,
                80,
                alpha=0.5
            )

            put_text(
                frame,
                "Outside analysis segment",
                25,
                60,
                scale=0.65
            )

        # ======================================
        # 오른쪽 위 작은 프레임 표시
        # ======================================

        cv2.putText(
            frame,
            f"Video Frame {video_frame_idx}",
            (
                max(
                    width - 310,
                    10
                ),
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(
            frame
        )

        video_frame_idx += 1

        if (
            video_frame_idx % 30 == 0
        ):
            print(
                f"[진행] "
                f"{video_frame_idx}/"
                f"{total_video_frames}"
            )

    # --------------------------------------------------------
    # 6. 종료
    # --------------------------------------------------------

    cap.release()
    writer.release()

    # --------------------------------------------------------
    # 6-A. Browser-compatible H.264 MP4
    # --------------------------------------------------------

    converted = convert_video_for_web(
        temp_video_path,
        output_path_obj,
    )

    if converted:
        try:
            temp_video_path.unlink()
        except OSError:
            pass

        print(
            "[WEB VIDEO] H.264 / yuv420p 변환 완료"
        )

    else:
        if output_path_obj.exists():
            try:
                output_path_obj.unlink()
            except OSError:
                pass

        os.replace(
            temp_video_path,
            output_path_obj,
        )

        print(
            "[WEB VIDEO] mp4v 원본을 유지합니다. "
            "브라우저에서 재생되지 않으면 ffmpeg 설치가 필요합니다."
        )

    print()
    print(
        "===== Overlay Result ====="
    )

    print(
        f"전체 영상 프레임 : "
        f"{video_frame_idx}"
    )

    print(
        f"분석 표시 프레임 : "
        f"{analyzed_count}"
    )

    print(
        f"출력 영상        : "
        f"{output_video_path}"
    )

    print()
    print(
        "[완료] Analysis Overlay 영상 생성 완료"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--events")
    parser.add_argument("--output", required=True)
    parser.add_argument("--throwing-side", default="left")
    parser.add_argument("--roi-json")
    args = parser.parse_args()
    create_analysis_overlay(
        video_path=args.video,
        csv_path=args.input,
        output_video_path=args.output,
        throwing_side=args.throwing_side,
        roi_json=args.roi_json,
    )
