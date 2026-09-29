import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    "right_ankle"
]


# ============================================================
# 관절별 기본 설정
# ============================================================

CONF_THRESHOLDS = {
    "nose": 0.40,
    "left_eye": 0.35,
    "right_eye": 0.35,
    "left_ear": 0.35,
    "right_ear": 0.35,

    "left_shoulder": 0.50,
    "right_shoulder": 0.50,

    "left_elbow": 0.35,
    "right_elbow": 0.35,

    "left_wrist": 0.30,
    "right_wrist": 0.30,

    "left_hip": 0.45,
    "right_hip": 0.45,

    "left_knee": 0.35,
    "right_knee": 0.35,

    "left_ankle": 0.30,
    "right_ankle": 0.30,
}


# 관절별 허용 이동량 배수
# median 이동량 대비 몇 배 이상이면 이상치 후보로 볼지
MOVE_MULTIPLIERS = {
    "nose": 6.0,
    "left_eye": 6.0,
    "right_eye": 6.0,
    "left_ear": 6.0,
    "right_ear": 6.0,

    "left_shoulder": 5.0,
    "right_shoulder": 5.0,

    "left_elbow": 6.0,
    "right_elbow": 6.0,

    # 손목은 실제로 빠르게 움직이므로 더 여유롭게
    "left_wrist": 8.0,
    "right_wrist": 8.0,

    "left_hip": 5.0,
    "right_hip": 5.0,

    "left_knee": 6.0,
    "right_knee": 6.0,

    "left_ankle": 7.0,
    "right_ankle": 7.0,
}


def calculate_step_distance(x, y):
    """
    프레임 간 좌표 이동량 계산
    """

    dx = x.diff()
    dy = y.diff()

    distance = np.sqrt(
        dx ** 2 + dy ** 2
    )

    return distance


def detect_jump_outliers(
    x,
    y,
    multiplier=6.0
):
    """
    프레임 간 이동량이 median 대비
    지나치게 큰 경우 이상치 후보로 판정
    """

    distance = calculate_step_distance(
        x,
        y
    )

    valid_distance = distance[
        distance.notna()
    ]

    if len(valid_distance) == 0:
        return pd.Series(
            False,
            index=x.index
        )

    median_move = valid_distance.median()

    # 거의 움직임이 없을 때 0 division 방지
    if median_move <= 1e-6:
        return pd.Series(
            False,
            index=x.index
        )

    threshold = (
        median_move
        * multiplier
    )

    outlier = (
        distance > threshold
    )

    outlier = outlier.fillna(
        False
    )

    return outlier


def remove_isolated_outliers(
    series,
    invalid_mask,
    max_invalid_run=4
):
    """
    너무 긴 구간은 함부로 삭제하지 않고,
    짧은 이상 구간만 NaN 처리한다.
    """

    result = series.copy()

    mask = invalid_mask.to_numpy()

    n = len(mask)
    i = 0

    while i < n:

        if not mask[i]:
            i += 1
            continue

        start = i

        while (
            i < n
            and mask[i]
        ):
            i += 1

        end = i

        run_length = (
            end - start
        )

        if run_length <= max_invalid_run:
            result.iloc[
                start:end
            ] = np.nan

    return result


def interpolate_short_gaps(
    series,
    max_gap=4
):
    """
    짧은 NaN 구간만 선형 보간
    """

    return series.interpolate(
        method="linear",
        limit=max_gap,
        limit_direction="both"
    )


def light_smoothing(
    series,
    window=3
):
    """
    아주 약한 rolling smoothing.
    투구 손목 속도를 죽이지 않도록 window=3.
    """

    return series.rolling(
        window=window,
        center=True,
        min_periods=1
    ).median()


def clean_joint(
    df,
    joint_name,
    max_invalid_run=4,
    interpolate_gap=4,
    smoothing_window=3
):
    x_col = f"{joint_name}_x"
    y_col = f"{joint_name}_y"
    conf_col = f"{joint_name}_conf"

    x = df[x_col].copy()
    y = df[y_col].copy()
    conf = df[conf_col].copy()

    conf_threshold = (
        CONF_THRESHOLDS.get(
            joint_name,
            0.35
        )
    )

    move_multiplier = (
        MOVE_MULTIPLIERS.get(
            joint_name,
            6.0
        )
    )

    # --------------------------------
    # 1. 저신뢰 좌표
    # --------------------------------
    low_conf_mask = (
        conf < conf_threshold
    )

    # --------------------------------
    # 2. 급격한 점프
    # --------------------------------
    jump_mask = detect_jump_outliers(
        x,
        y,
        multiplier=move_multiplier
    )

    # --------------------------------
    # 3. 최종 이상치 후보
    # --------------------------------
    invalid_mask = (
        low_conf_mask
        | jump_mask
    )

    # --------------------------------
    # 4. 짧은 이상 구간만 제거
    # --------------------------------
    x_clean = remove_isolated_outliers(
        x,
        invalid_mask,
        max_invalid_run=max_invalid_run
    )

    y_clean = remove_isolated_outliers(
        y,
        invalid_mask,
        max_invalid_run=max_invalid_run
    )

    # --------------------------------
    # 5. 짧은 NaN 보간
    # --------------------------------
    x_interp = interpolate_short_gaps(
        x_clean,
        max_gap=interpolate_gap
    )

    y_interp = interpolate_short_gaps(
        y_clean,
        max_gap=interpolate_gap
    )

    # --------------------------------
    # 6. 아주 약한 smoothing
    # --------------------------------
    x_smooth = light_smoothing(
        x_interp,
        window=smoothing_window
    )

    y_smooth = light_smoothing(
        y_interp,
        window=smoothing_window
    )

    return {
        "x_clean": x_smooth,
        "y_clean": y_smooth,
        "low_conf_mask": low_conf_mask,
        "jump_mask": jump_mask,
        "invalid_mask": invalid_mask,
    }


def clean_pose_csv(
    csv_path,
    output_csv_path,
    graph_dir,
    throwing_side="left"
):
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
        "===== [STEP 6] Pose Cleaning ====="
    )

    print(
        f"입력 CSV     : {csv_path}"
    )

    print(
        f"총 프레임 수 : {len(df)}"
    )

    print(
        "=================================="
    )

    clean_df = df.copy()

    os.makedirs(
        os.path.dirname(
            output_csv_path
        ),
        exist_ok=True
    )

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    summary = []

    for joint in KEYPOINT_NAMES:

        result = clean_joint(
            df,
            joint_name=joint,
            max_invalid_run=4,
            interpolate_gap=4,
            smoothing_window=3
        )

        clean_df[
            f"{joint}_x_clean"
        ] = result[
            "x_clean"
        ]

        clean_df[
            f"{joint}_y_clean"
        ] = result[
            "y_clean"
        ]

        clean_df[
            f"{joint}_low_conf"
        ] = result[
            "low_conf_mask"
        ]

        clean_df[
            f"{joint}_jump"
        ] = result[
            "jump_mask"
        ]

        clean_df[
            f"{joint}_invalid"
        ] = result[
            "invalid_mask"
        ]

        low_count = int(
            result[
                "low_conf_mask"
            ].sum()
        )

        jump_count = int(
            result[
                "jump_mask"
            ].sum()
        )

        invalid_count = int(
            result[
                "invalid_mask"
            ].sum()
        )

        summary.append({
            "joint": joint,
            "low_conf_frames": low_count,
            "jump_frames": jump_count,
            "invalid_frames": invalid_count
        })

    # --------------------------------
    # CSV 저장
    # --------------------------------
    clean_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df = pd.DataFrame(
        summary
    )

    summary_path = os.path.join(
        graph_dir,
        "cleaning_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 투구팔 그래프 생성
    # =====================================================

    joints_to_plot = [
        f"{throwing_side}_shoulder",
        f"{throwing_side}_elbow",
        f"{throwing_side}_wrist"
    ]

    frames = clean_df["frame"]

    for joint in joints_to_plot:

        # X 그래프
        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(
            frames,
            clean_df[f"{joint}_x"],
            label="Original X"
        )

        plt.plot(
            frames,
            clean_df[f"{joint}_x_clean"],
            label="Clean X"
        )

        plt.xlabel("Frame")
        plt.ylabel("Pixel X")

        plt.title(
            f"{joint} X Cleaning"
        )

        plt.legend()
        plt.grid()

        path_x = os.path.join(
            graph_dir,
            f"{joint}_x_cleaning.png"
        )

        plt.savefig(
            path_x,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # Y 그래프
        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(
            frames,
            clean_df[f"{joint}_y"],
            label="Original Y"
        )

        plt.plot(
            frames,
            clean_df[f"{joint}_y_clean"],
            label="Clean Y"
        )

        plt.xlabel("Frame")
        plt.ylabel("Pixel Y")

        plt.title(
            f"{joint} Y Cleaning"
        )

        plt.legend()
        plt.grid()

        path_y = os.path.join(
            graph_dir,
            f"{joint}_y_cleaning.png"
        )

        plt.savefig(
            path_y,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================
    # 투구팔 이상치 요약 그래프
    # =====================================================

    wrist_joint = (
        f"{throwing_side}_wrist"
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        frames,
        clean_df[
            f"{wrist_joint}_conf"
        ],
        label="Wrist confidence"
    )

    invalid_frames = clean_df[
        f"{wrist_joint}_invalid"
    ]

    invalid_frame_numbers = (
        clean_df.loc[
            invalid_frames,
            "frame"
        ]
    )

    invalid_conf_values = (
        clean_df.loc[
            invalid_frames,
            f"{wrist_joint}_conf"
        ]
    )

    plt.scatter(
        invalid_frame_numbers,
        invalid_conf_values,
        label="Invalid candidate",
        s=20
    )

    plt.axhline(
        y=CONF_THRESHOLDS[
            wrist_joint
        ],
        linestyle="--",
        label="Confidence threshold"
    )

    plt.xlabel("Frame")
    plt.ylabel("Confidence")

    plt.title(
        f"{wrist_joint} Cleaning Diagnosis"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()
    plt.grid()

    wrist_graph_path = (
        os.path.join(
            graph_dir,
            f"{wrist_joint}_cleaning_diagnosis.png"
        )
    )

    plt.savefig(
        wrist_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "===== Cleaning Summary ====="
    )

    for item in summary:

        if item["joint"] in joints_to_plot:

            print()
            print(
                item["joint"]
            )

            print(
                f" 저신뢰 프레임 : "
                f"{item['low_conf_frames']}"
            )

            print(
                f" 점프 후보     : "
                f"{item['jump_frames']}"
            )

            print(
                f" 전체 이상 후보: "
                f"{item['invalid_frames']}"
            )

    print()
    print(
        f"정제 CSV : {output_csv_path}"
    )

    print(
        f"요약 CSV : {summary_path}"
    )

    print(
        f"그래프 폴더: {graph_dir}"
    )

    print()
    print(
        "[완료] Pose Cleaning 완료"
    )


if __name__ == "__main__":

    from pipeline_config import (
        get_throwing_side
    )

    INPUT_CSV = (
        "results/segment/sample/"
        "pitch_segment_pose.csv"
    )

    OUTPUT_CSV = (
        "data/pose_csv/"
        "sample_pose_clean.csv"
    )

    GRAPH_DIR = (
        "results/graphs/"
        "sample_pose_cleaning"
    )

    THROWING_SIDE = (
        get_throwing_side()
    )

    clean_pose_csv(
        csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        graph_dir=GRAPH_DIR,
        throwing_side=THROWING_SIDE
    )