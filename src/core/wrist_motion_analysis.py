import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 좌표 가져오기
# ============================================================

def get_xy(df, joint_name):
    """
    clean 좌표를 NumPy 배열로 반환

    shape:
        (frame_count, 2)
    """

    x_col = f"{joint_name}_x_clean"
    y_col = f"{joint_name}_y_clean"

    if (
        x_col not in df.columns
        or y_col not in df.columns
    ):
        raise ValueError(
            f"{joint_name}의 clean 좌표가 없습니다."
        )

    return df[
        [x_col, y_col]
    ].to_numpy(dtype=float)


# ============================================================
# 두 점의 중점
# ============================================================

def midpoint(points_a, points_b):
    return (
        points_a + points_b
    ) / 2.0


# ============================================================
# 프레임 간 이동거리
# ============================================================

def calculate_displacement(points):
    """
    각 프레임 사이의 이동거리(px)를 계산한다.

    첫 프레임은 이전 프레임이 없으므로 NaN.
    """

    delta = np.diff(
        points,
        axis=0
    )

    distance = np.sqrt(
        delta[:, 0] ** 2
        + delta[:, 1] ** 2
    )

    return np.insert(
        distance,
        0,
        np.nan
    )


# ============================================================
# 속도 계산
# ============================================================

def calculate_velocity(points, timestamps):
    """
    px/s 단위 속도 계산

    timestamps를 직접 사용하므로
    FPS를 별도로 하드코딩할 필요가 없다.
    """

    delta_points = np.diff(
        points,
        axis=0
    )

    delta_time = np.diff(
        timestamps
    )

    distance = np.sqrt(
        delta_points[:, 0] ** 2
        + delta_points[:, 1] ** 2
    )

    velocity = np.full(
        len(points),
        np.nan
    )

    valid_dt = delta_time > 0

    velocity_part = np.full(
        len(delta_time),
        np.nan
    )

    velocity_part[
        valid_dt
    ] = (
        distance[valid_dt]
        / delta_time[valid_dt]
    )

    velocity[1:] = (
        velocity_part
    )

    return velocity


# ============================================================
# 몸 크기 계산
# ============================================================

def calculate_body_scale(
    shoulder_mid,
    hip_mid,
    left_ankle,
    right_ankle
):
    """
    프레임별 신체 크기 기준을 계산한다.

    기본:
        shoulder midpoint
            ↓
        ankle midpoint

    전신 길이를 사용한다.

    값이 이상하거나 사용할 수 없으면
    shoulder -> hip 길이를 이용한다.
    """

    ankle_mid = midpoint(
        left_ankle,
        right_ankle
    )

    full_body_vector = (
        ankle_mid
        - shoulder_mid
    )

    full_body_scale = np.sqrt(
        full_body_vector[:, 0] ** 2
        + full_body_vector[:, 1] ** 2
    )

    torso_vector = (
        hip_mid
        - shoulder_mid
    )

    torso_scale = np.sqrt(
        torso_vector[:, 0] ** 2
        + torso_vector[:, 1] ** 2
    )

    # 이상하게 너무 작은 전신값은
    # torso 기반 값으로 교체
    body_scale = (
        full_body_scale.copy()
    )

    invalid = (
        ~np.isfinite(body_scale)
        |
        (body_scale < 20)
    )

    body_scale[
        invalid
    ] = (
        torso_scale[invalid]
        * 2.5
    )

    return body_scale


# ============================================================
# 아주 약한 속도 smoothing
# ============================================================

def smooth_speed(
    speed,
    window=3
):
    """
    속도 그래프의 한두 프레임 흔들림만 줄인다.

    좌표 자체를 또 smoothing하는 것이 아니라
    계산된 속도 신호에만 median filter를 적용한다.
    """

    series = pd.Series(
        speed
    )

    return (
        series
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )


# ============================================================
# 메인 분석
# ============================================================

def analyze_wrist_motion(
    csv_path,
    output_csv_path,
    graph_dir,
    throwing_side="left",
    smoothing_window=3
):
    # --------------------------------------------------------
    # 1. 파일 확인
    # --------------------------------------------------------

    if not os.path.exists(
        csv_path
    ):
        print(
            f"[오류] CSV 파일을 찾을 수 없습니다: "
            f"{csv_path}"
        )
        return

    df = pd.read_csv(
        csv_path
    )

    print()
    print(
        "===== [STEP 8] Wrist Motion Analysis ====="
    )

    print(
        f"입력 CSV     : {csv_path}"
    )

    print(
        f"총 프레임 수 : {len(df)}"
    )

    print(
        f"투구팔       : {throwing_side}"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # 2. timestamp 확인
    # --------------------------------------------------------

    if "timestamp" not in df.columns:
        print(
            "[오류] timestamp 컬럼이 없습니다."
        )
        return

    timestamps = (
        df["timestamp"]
        .to_numpy(dtype=float)
    )

    if len(timestamps) < 2:
        print(
            "[오류] 프레임 수가 부족합니다."
        )
        return

    delta_time = np.diff(
        timestamps
    )

    valid_dt = delta_time[
        delta_time > 0
    ]

    if len(valid_dt) == 0:
        print(
            "[오류] timestamp 값이 올바르지 않습니다."
        )
        return

    median_dt = np.median(
        valid_dt
    )

    estimated_fps = (
        1.0 / median_dt
    )

    print(
        f"추정 FPS     : "
        f"{estimated_fps:.2f}"
    )

    # --------------------------------------------------------
    # 3. 관절 좌표
    # --------------------------------------------------------

    wrist = get_xy(
        df,
        f"{throwing_side}_wrist"
    )

    left_shoulder = get_xy(
        df,
        "left_shoulder"
    )

    right_shoulder = get_xy(
        df,
        "right_shoulder"
    )

    left_hip = get_xy(
        df,
        "left_hip"
    )

    right_hip = get_xy(
        df,
        "right_hip"
    )

    left_ankle = get_xy(
        df,
        "left_ankle"
    )

    right_ankle = get_xy(
        df,
        "right_ankle"
    )

    # --------------------------------------------------------
    # 4. 몸통 중심
    # --------------------------------------------------------

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_mid = midpoint(
        left_hip,
        right_hip
    )

    # 어깨 중점 + 골반 중점의 중간
    torso_center = midpoint(
        shoulder_mid,
        hip_mid
    )

    # --------------------------------------------------------
    # 5. 손목 절대 이동
    # --------------------------------------------------------

    wrist_dx = np.insert(
        np.diff(
            wrist[:, 0]
        ),
        0,
        np.nan
    )

    wrist_dy = np.insert(
        np.diff(
            wrist[:, 1]
        ),
        0,
        np.nan
    )

    wrist_displacement = (
        calculate_displacement(
            wrist
        )
    )

    wrist_speed = (
        calculate_velocity(
            wrist,
            timestamps
        )
    )

    # --------------------------------------------------------
    # 6. 몸통 이동 속도
    # --------------------------------------------------------

    torso_speed = (
        calculate_velocity(
            torso_center,
            timestamps
        )
    )

    # --------------------------------------------------------
    # 7. 몸통 기준 손목 상대좌표
    # --------------------------------------------------------

    wrist_relative = (
        wrist
        - torso_center
    )

    wrist_relative_displacement = (
        calculate_displacement(
            wrist_relative
        )
    )

    wrist_relative_speed = (
        calculate_velocity(
            wrist_relative,
            timestamps
        )
    )

    # --------------------------------------------------------
    # 8. 신체 크기 계산
    # --------------------------------------------------------

    body_scale = calculate_body_scale(
        shoulder_mid,
        hip_mid,
        left_ankle,
        right_ankle
    )

    # 영상 전체의 median body scale
    reference_body_scale = (
        np.nanmedian(
            body_scale
        )
    )

    if (
        not np.isfinite(
            reference_body_scale
        )
        or reference_body_scale <= 0
    ):
        print(
            "[오류] 신체 크기 기준 계산 실패"
        )
        return

    # --------------------------------------------------------
    # 9. 정규화 속도
    # --------------------------------------------------------

    normalized_wrist_speed = (
        wrist_speed
        / reference_body_scale
    )

    normalized_relative_speed = (
        wrist_relative_speed
        / reference_body_scale
    )

    # --------------------------------------------------------
    # 10. 속도 약한 smoothing
    # --------------------------------------------------------

    wrist_speed_smooth = (
        smooth_speed(
            wrist_speed,
            window=smoothing_window
        )
    )

    relative_speed_smooth = (
        smooth_speed(
            wrist_relative_speed,
            window=smoothing_window
        )
    )

    normalized_speed_smooth = (
        smooth_speed(
            normalized_relative_speed,
            window=smoothing_window
        )
    )

    # --------------------------------------------------------
    # 11. 결과 컬럼 생성
    # --------------------------------------------------------

    motion_df = pd.DataFrame(
        {
            "wrist_dx":
                wrist_dx,

            "wrist_dy":
                wrist_dy,

            "wrist_displacement_px":
                wrist_displacement,

            "wrist_speed_px_s":
                wrist_speed,

            "wrist_speed_px_s_smooth":
                wrist_speed_smooth,

            "torso_center_x":
                torso_center[:, 0],

            "torso_center_y":
                torso_center[:, 1],

            "torso_speed_px_s":
                torso_speed,

            "wrist_relative_x":
                wrist_relative[:, 0],

            "wrist_relative_y":
                wrist_relative[:, 1],

            "wrist_relative_displacement_px":
                wrist_relative_displacement,

            "wrist_relative_speed_px_s":
                wrist_relative_speed,

            "wrist_relative_speed_px_s_smooth":
                relative_speed_smooth,

            "body_scale_px":
                body_scale,

            "wrist_speed_body_s":
                normalized_wrist_speed,

            "wrist_relative_speed_body_s":
                normalized_relative_speed,

            "wrist_relative_speed_body_s_smooth":
                normalized_speed_smooth,
        },
        index=df.index
    )

    result_df = pd.concat(
        [
            df.copy(),
            motion_df
        ],
        axis=1
    )

    # --------------------------------------------------------
    # 12. 최대 속도 프레임
    # --------------------------------------------------------

    speed_series = (
        result_df[
            "wrist_relative_speed_body_s_smooth"
        ]
    )

    valid_speed = (
        speed_series.dropna()
    )

    if len(valid_speed) == 0:
        print(
            "[오류] 유효 속도 데이터가 없습니다."
        )
        return

    max_speed_idx = (
        valid_speed.idxmax()
    )

    max_speed_frame = int(
        result_df.loc[
            max_speed_idx,
            "frame"
        ]
    )

    max_speed_timestamp = float(
        result_df.loc[
            max_speed_idx,
            "timestamp"
        ]
    )

    max_speed_px = float(
        result_df.loc[
            max_speed_idx,
            "wrist_relative_speed_px_s_smooth"
        ]
    )

    max_speed_normalized = float(
        result_df.loc[
            max_speed_idx,
            "wrist_relative_speed_body_s_smooth"
        ]
    )

    # 후보 표시
    result_df[
        "wrist_peak_speed_candidate"
    ] = False

    result_df.loc[
        max_speed_idx,
        "wrist_peak_speed_candidate"
    ] = True

    # --------------------------------------------------------
    # 13. 출력 폴더
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_csv_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 14. CSV 저장
    # --------------------------------------------------------

    result_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    frames = (
        result_df["frame"]
    )

    # ========================================================
    # GRAPH 1
    # 손목 화면 좌표
    # ========================================================

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        frames,
        wrist[:, 0],
        label="Wrist X"
    )

    plt.plot(
        frames,
        wrist[:, 1],
        label="Wrist Y"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Pixel Coordinate"
    )

    plt.title(
        f"{throwing_side.capitalize()} Wrist Position"
    )

    plt.legend()
    plt.grid()

    position_graph_path = (
        os.path.join(
            graph_dir,
            "wrist_position.png"
        )
    )

    plt.savefig(
        position_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # GRAPH 2
    # 절대 속도 vs 몸통 상대속도
    # ========================================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        wrist_speed_smooth,
        label="Absolute Wrist Speed"
    )

    plt.plot(
        frames,
        relative_speed_smooth,
        label="Torso-relative Wrist Speed"
    )

    plt.axvline(
        x=max_speed_frame,
        linestyle="--",
        label=(
            f"Peak candidate "
            f"(Frame {max_speed_frame})"
        )
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Speed (pixel / second)"
    )

    plt.title(
        "Wrist Speed Analysis"
    )

    plt.legend()
    plt.grid()

    speed_graph_path = (
        os.path.join(
            graph_dir,
            "wrist_speed.png"
        )
    )

    plt.savefig(
        speed_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # GRAPH 3
    # 정규화 손목 속도
    # ========================================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        normalized_speed_smooth,
        label=(
            "Normalized Torso-relative "
            "Wrist Speed"
        )
    )

    plt.axvline(
        x=max_speed_frame,
        linestyle="--",
        label=(
            f"Peak candidate "
            f"(Frame {max_speed_frame})"
        )
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Body Length / second"
    )

    plt.title(
        "Normalized Wrist Speed"
    )

    plt.legend()
    plt.grid()

    normalized_graph_path = (
        os.path.join(
            graph_dir,
            "wrist_speed_normalized.png"
        )
    )

    plt.savefig(
        normalized_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # GRAPH 4
    # 손목 궤적
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        wrist[:, 0],
        wrist[:, 1],
        marker=".",
        markersize=3
    )

    # 영상 좌표는 y가 아래 방향으로 증가하므로
    # 실제 영상처럼 보기 위해 뒤집는다.
    plt.gca().invert_yaxis()

    plt.xlabel(
        "X (pixel)"
    )

    plt.ylabel(
        "Y (pixel)"
    )

    plt.title(
        f"{throwing_side.capitalize()} Wrist Trajectory"
    )

    plt.grid()

    trajectory_graph_path = (
        os.path.join(
            graph_dir,
            "wrist_trajectory.png"
        )
    )

    plt.savefig(
        trajectory_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 결과 통계
    # ========================================================

    print()
    print(
        "===== Wrist Motion Statistics ====="
    )

    print(
        f"기준 신체 크기 : "
        f"{reference_body_scale:.2f} px"
    )

    print()
    print(
        "[최대 손목 상대속도 후보]"
    )

    print(
        f"프레임       : "
        f"{max_speed_frame}"
    )

    print(
        f"시간         : "
        f"{max_speed_timestamp:.3f} sec"
    )

    print(
        f"상대 속도    : "
        f"{max_speed_px:.2f} px/s"
    )

    print(
        f"정규화 속도  : "
        f"{max_speed_normalized:.3f} body/s"
    )

    print()
    print(
        "※ 최대 손목 속도 프레임은 "
        "아직 Ball Release가 아닙니다."
    )

    print(
        "   이후 Pitching Phase 및 Release "
        "Detection의 후보값으로 사용합니다."
    )

    print()
    print(
        f"결과 CSV : "
        f"{output_csv_path}"
    )

    print(
        f"위치 그래프 : "
        f"{position_graph_path}"
    )

    print(
        f"속도 그래프 : "
        f"{speed_graph_path}"
    )

    print(
        f"정규화 그래프 : "
        f"{normalized_graph_path}"
    )

    print(
        f"궤적 그래프 : "
        f"{trajectory_graph_path}"
    )

    print()
    print(
        "[완료] Wrist Motion Analysis 완료"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from pipeline_config import (
        get_throwing_side
    )

    INPUT_CSV = (
        "data/pose_csv/"
        "sample_joint_angles_validated_v2.csv"
    )

    OUTPUT_CSV = (
        "data/pose_csv/"
        "sample_wrist_motion.csv"
    )

    GRAPH_DIR = (
        "results/graphs/"
        "sample_wrist_motion"
    )

    THROWING_SIDE = (
        get_throwing_side()
    )

    analyze_wrist_motion(
        csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        graph_dir=GRAPH_DIR,
        throwing_side=THROWING_SIDE,
        smoothing_window=3
    )