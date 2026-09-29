import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_angle(point_a, point_b, point_c):
    """
    세 점 A-B-C에서
    B를 꼭짓점으로 하는 각도를 계산한다.

    반환값:
        degree 단위 각도 (0 ~ 180)
    """

    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)

    # B를 기준으로 두 벡터 생성
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # 길이가 0이면 각도 계산 불가
    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine_angle = np.dot(ba, bc) / (
        norm_ba * norm_bc
    )

    # 부동소수점 오차 방지
    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle_radian = np.arccos(
        cosine_angle
    )

    angle_degree = np.degrees(
        angle_radian
    )

    return float(angle_degree)


def get_clean_point(row, joint_name):
    """
    정제된 관절 좌표 반환
    """

    x_col = f"{joint_name}_x_clean"
    y_col = f"{joint_name}_y_clean"

    x = row[x_col]
    y = row[y_col]

    if pd.isna(x) or pd.isna(y):
        return None

    return (float(x), float(y))


def calculate_joint_angles(
    csv_path,
    output_csv_path,
    graph_dir,
    throwing_side="left"
):
    # --------------------------------------
    # 1. CSV 확인
    # --------------------------------------
    if not os.path.exists(csv_path):
        print(
            f"[오류] CSV 파일을 찾을 수 없습니다: "
            f"{csv_path}"
        )
        return

    df = pd.read_csv(csv_path)

    print()
    print(
        "===== [STEP 7] Joint Angle Analysis ====="
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
        "========================================="
    )

    # --------------------------------------
    # 2. 필요한 컬럼 확인
    # --------------------------------------
    required_joints = [
        f"{throwing_side}_shoulder",
        f"{throwing_side}_elbow",
        f"{throwing_side}_wrist",

        "left_hip",
        "left_knee",
        "left_ankle",

        "right_hip",
        "right_knee",
        "right_ankle",
    ]

    missing_columns = []

    for joint in required_joints:
        for axis in ["x_clean", "y_clean"]:
            col = f"{joint}_{axis}"

            if col not in df.columns:
                missing_columns.append(col)

    if missing_columns:
        print()
        print("[오류] 필요한 컬럼이 없습니다.")

        for col in missing_columns:
            print(" -", col)

        return

    # --------------------------------------
    # 3. 각도 저장 리스트
    # --------------------------------------
    throwing_elbow_angles = []
    left_knee_angles = []
    right_knee_angles = []

    # --------------------------------------
    # 4. 프레임별 계산
    # --------------------------------------
    for i in range(len(df)):
        row = df.iloc[i]

        # ==============================
        # 투구팔 팔꿈치
        # Shoulder - Elbow - Wrist
        # ==============================

        shoulder = get_clean_point(
            row,
            f"{throwing_side}_shoulder"
        )

        elbow = get_clean_point(
            row,
            f"{throwing_side}_elbow"
        )

        wrist = get_clean_point(
            row,
            f"{throwing_side}_wrist"
        )

        if (
            shoulder is not None
            and elbow is not None
            and wrist is not None
        ):
            elbow_angle = calculate_angle(
                shoulder,
                elbow,
                wrist
            )
        else:
            elbow_angle = np.nan

        throwing_elbow_angles.append(
            elbow_angle
        )

        # ==============================
        # 왼쪽 무릎
        # Hip - Knee - Ankle
        # ==============================

        left_hip = get_clean_point(
            row,
            "left_hip"
        )

        left_knee = get_clean_point(
            row,
            "left_knee"
        )

        left_ankle = get_clean_point(
            row,
            "left_ankle"
        )

        if (
            left_hip is not None
            and left_knee is not None
            and left_ankle is not None
        ):
            left_knee_angle = calculate_angle(
                left_hip,
                left_knee,
                left_ankle
            )
        else:
            left_knee_angle = np.nan

        left_knee_angles.append(
            left_knee_angle
        )

        # ==============================
        # 오른쪽 무릎
        # Hip - Knee - Ankle
        # ==============================

        right_hip = get_clean_point(
            row,
            "right_hip"
        )

        right_knee = get_clean_point(
            row,
            "right_knee"
        )

        right_ankle = get_clean_point(
            row,
            "right_ankle"
        )

        if (
            right_hip is not None
            and right_knee is not None
            and right_ankle is not None
        ):
            right_knee_angle = calculate_angle(
                right_hip,
                right_knee,
                right_ankle
            )
        else:
            right_knee_angle = np.nan

        right_knee_angles.append(
            right_knee_angle
        )

    # --------------------------------------
    # 5. DataFrame에 추가
    # --------------------------------------
    result_df = df.copy()

    result_df[
        "throwing_elbow_angle"
    ] = throwing_elbow_angles

    result_df[
        "left_knee_angle"
    ] = left_knee_angles

    result_df[
        "right_knee_angle"
    ] = right_knee_angles

    # --------------------------------------
    # 6. CSV 저장
    # --------------------------------------
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

    result_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 7. 팔꿈치 각도 그래프
    # ======================================
    frames = result_df["frame"]

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        frames,
        result_df[
            "throwing_elbow_angle"
        ],
        label="Throwing Elbow Angle"
    )

    plt.xlabel("Frame")
    plt.ylabel("Angle (degree)")

    plt.title(
        f"{throwing_side.capitalize()} Throwing Elbow Angle"
    )

    plt.ylim(
        0,
        190
    )

    plt.legend()
    plt.grid()

    elbow_graph_path = os.path.join(
        graph_dir,
        "throwing_elbow_angle.png"
    )

    plt.savefig(
        elbow_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================
    # 8. 무릎 각도 그래프
    # ======================================

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        frames,
        result_df[
            "left_knee_angle"
        ],
        label="Left Knee Angle"
    )

    plt.plot(
        frames,
        result_df[
            "right_knee_angle"
        ],
        label="Right Knee Angle"
    )

    plt.xlabel("Frame")
    plt.ylabel("Angle (degree)")

    plt.title(
        "Left / Right Knee Angle"
    )

    plt.ylim(
        0,
        190
    )

    plt.legend()
    plt.grid()

    knee_graph_path = os.path.join(
        graph_dir,
        "knee_angles.png"
    )

    plt.savefig(
        knee_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================
    # 9. 통합 그래프
    # ======================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        result_df[
            "throwing_elbow_angle"
        ],
        label="Throwing Elbow"
    )

    plt.plot(
        frames,
        result_df[
            "left_knee_angle"
        ],
        label="Left Knee"
    )

    plt.plot(
        frames,
        result_df[
            "right_knee_angle"
        ],
        label="Right Knee"
    )

    plt.xlabel("Frame")
    plt.ylabel("Angle (degree)")

    plt.title(
        "Pitching Joint Angle Analysis"
    )

    plt.ylim(
        0,
        190
    )

    plt.legend()
    plt.grid()

    combined_graph_path = os.path.join(
        graph_dir,
        "joint_angles_combined.png"
    )

    plt.savefig(
        combined_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================
    # 10. 통계 출력
    # ======================================

    elbow_series = result_df[
        "throwing_elbow_angle"
    ]

    left_knee_series = result_df[
        "left_knee_angle"
    ]

    right_knee_series = result_df[
        "right_knee_angle"
    ]

    print()
    print(
        "===== Angle Statistics ====="
    )

    print()
    print("[Throwing Elbow]")

    print(
        f"최소 각도 : "
        f"{elbow_series.min():.2f}°"
    )

    print(
        f"최대 각도 : "
        f"{elbow_series.max():.2f}°"
    )

    print(
        f"평균 각도 : "
        f"{elbow_series.mean():.2f}°"
    )

    if elbow_series.notna().any():
        min_elbow_idx = (
            elbow_series.idxmin()
        )

        max_elbow_idx = (
            elbow_series.idxmax()
        )

        print(
            f"최소 각도 프레임 : "
            f"{int(result_df.loc[min_elbow_idx, 'frame'])}"
        )

        print(
            f"최대 각도 프레임 : "
            f"{int(result_df.loc[max_elbow_idx, 'frame'])}"
        )

    print()
    print("[Left Knee]")

    print(
        f"최소 각도 : "
        f"{left_knee_series.min():.2f}°"
    )

    print(
        f"최대 각도 : "
        f"{left_knee_series.max():.2f}°"
    )

    print()
    print("[Right Knee]")

    print(
        f"최소 각도 : "
        f"{right_knee_series.min():.2f}°"
    )

    print(
        f"최대 각도 : "
        f"{right_knee_series.max():.2f}°"
    )

    print()
    print(
        f"결과 CSV : "
        f"{output_csv_path}"
    )

    print(
        f"팔꿈치 그래프 : "
        f"{elbow_graph_path}"
    )

    print(
        f"무릎 그래프   : "
        f"{knee_graph_path}"
    )

    print(
        f"통합 그래프   : "
        f"{combined_graph_path}"
    )

    print()
    print(
        "[완료] Joint Angle Analysis 완료"
    )


if __name__ == "__main__":

    from pipeline_config import (
        get_throwing_side
    )

    INPUT_CSV = (
        "data/pose_csv/"
        "sample_pose_clean.csv"
    )

    OUTPUT_CSV = (
        "data/pose_csv/"
        "sample_joint_angles.csv"
    )

    GRAPH_DIR = (
        "results/graphs/"
        "sample_joint_angles"
    )

    THROWING_SIDE = (
        get_throwing_side()
    )

    calculate_joint_angles(
        csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        graph_dir=GRAPH_DIR,
        throwing_side=THROWING_SIDE
    )