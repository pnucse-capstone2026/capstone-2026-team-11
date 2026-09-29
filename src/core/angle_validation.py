import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def distance(p1, p2):
    return np.linalg.norm(
        np.array(p1, dtype=float)
        - np.array(p2, dtype=float)
    )


def get_point(row, joint):
    x = row[f"{joint}_x_clean"]
    y = row[f"{joint}_y_clean"]

    if pd.isna(x) or pd.isna(y):
        return None

    return (float(x), float(y))


def rolling_median_abs_diff(series, window=7):
    """
    프레임 간 변화량의 절대값을 계산한 뒤
    주변 중앙값을 구한다.
    """
    diff = series.diff().abs()

    local_median = diff.rolling(
        window=window,
        center=True,
        min_periods=1
    ).median()

    return diff, local_median


def validate_elbow_angles_v2(
    csv_path,
    output_csv_path,
    graph_dir,
    throwing_side="left",

    min_joint_conf=0.30,

    min_upper_arm_ratio=0.45,
    min_forearm_ratio=0.45,
    min_total_arm_ratio=0.50,

    # 각도 급변 기준
    angle_jump_multiplier=4.0,
    min_angle_jump_threshold=25.0,

    # 팔 길이 비율 급변 기준
    length_ratio_jump_multiplier=4.0,
    min_length_ratio_jump_threshold=0.20,
):
    if not os.path.exists(csv_path):
        print(
            f"[오류] CSV 파일을 찾을 수 없습니다: "
            f"{csv_path}"
        )
        return

    df = pd.read_csv(csv_path)

    print()
    print(
        "===== [STEP 7.5 v2] Angle Validation ====="
    )
    print(f"입력 CSV     : {csv_path}")
    print(f"총 프레임 수 : {len(df)}")
    print(f"투구팔       : {throwing_side}")
    print(
        "=========================================="
    )

    shoulder = f"{throwing_side}_shoulder"
    elbow = f"{throwing_side}_elbow"
    wrist = f"{throwing_side}_wrist"

    required_columns = [
        "frame",
        "throwing_elbow_angle",

        f"{shoulder}_x_clean",
        f"{shoulder}_y_clean",
        f"{shoulder}_conf",

        f"{elbow}_x_clean",
        f"{elbow}_y_clean",
        f"{elbow}_conf",

        f"{wrist}_x_clean",
        f"{wrist}_y_clean",
        f"{wrist}_conf",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print("[오류] 필요한 컬럼이 없습니다.")

        for col in missing:
            print(" -", col)

        return

    # =====================================================
    # 1. 팔 길이 계산
    # =====================================================

    upper_lengths = []
    forearm_lengths = []
    total_lengths = []

    for _, row in df.iterrows():

        shoulder_point = get_point(
            row,
            shoulder
        )

        elbow_point = get_point(
            row,
            elbow
        )

        wrist_point = get_point(
            row,
            wrist
        )

        if (
            shoulder_point is None
            or elbow_point is None
            or wrist_point is None
        ):
            upper_lengths.append(np.nan)
            forearm_lengths.append(np.nan)
            total_lengths.append(np.nan)
            continue

        upper = distance(
            shoulder_point,
            elbow_point
        )

        forearm = distance(
            elbow_point,
            wrist_point
        )

        total = upper + forearm

        upper_lengths.append(
            upper
        )

        forearm_lengths.append(
            forearm
        )

        total_lengths.append(
            total
        )

    upper_series = pd.Series(
        upper_lengths,
        index=df.index
    )

    forearm_series = pd.Series(
        forearm_lengths,
        index=df.index
    )

    total_series = pd.Series(
        total_lengths,
        index=df.index
    )

    median_upper = upper_series.median()
    median_forearm = forearm_series.median()
    median_total = total_series.median()

    print()
    print(
        "===== 기준 팔 길이 ====="
    )
    print(
        f"Upper arm median : "
        f"{median_upper:.2f} px"
    )
    print(
        f"Forearm median   : "
        f"{median_forearm:.2f} px"
    )
    print(
        f"Total arm median : "
        f"{median_total:.2f} px"
    )

    # =====================================================
    # 2. 팔 길이 비율
    # =====================================================

    upper_ratio = (
        upper_series
        / median_upper
    )

    forearm_ratio = (
        forearm_series
        / median_forearm
    )

    total_ratio = (
        total_series
        / median_total
    )

    # 상완 / 전완 상대 비율
    arm_segment_ratio = (
        forearm_series
        / upper_series.replace(0, np.nan)
    )

    # =====================================================
    # 3. Confidence 조건
    # =====================================================

    confidence_valid = (
        (
            df[f"{shoulder}_conf"]
            >= min_joint_conf
        )
        &
        (
            df[f"{elbow}_conf"]
            >= min_joint_conf
        )
        &
        (
            df[f"{wrist}_conf"]
            >= min_joint_conf
        )
    )

    # =====================================================
    # 4. 기본 팔 길이 조건
    # =====================================================

    upper_valid = (
        upper_ratio
        >= min_upper_arm_ratio
    )

    forearm_valid = (
        forearm_ratio
        >= min_forearm_ratio
    )

    total_valid = (
        total_ratio
        >= min_total_arm_ratio
    )

    # =====================================================
    # 5. 각도 급변 검사
    # =====================================================

    elbow_angle = df[
        "throwing_elbow_angle"
    ]

    angle_diff, angle_local_median = (
        rolling_median_abs_diff(
            elbow_angle,
            window=7
        )
    )

    angle_jump_threshold = np.maximum(
        angle_local_median
        * angle_jump_multiplier,
        min_angle_jump_threshold
    )

    angle_jump = (
        angle_diff
        > angle_jump_threshold
    ).fillna(False)

    # =====================================================
    # 6. 팔 길이 비율 급변 검사
    # =====================================================

    ratio_diff, ratio_local_median = (
        rolling_median_abs_diff(
            arm_segment_ratio,
            window=7
        )
    )

    ratio_jump_threshold = np.maximum(
        ratio_local_median
        * length_ratio_jump_multiplier,
        min_length_ratio_jump_threshold
    )

    length_ratio_jump = (
        ratio_diff
        > ratio_jump_threshold
    ).fillna(False)

    # =====================================================
    # 7. 급변 주변 프레임도 같이 의심
    # =====================================================

    # 한 프레임만 찍고 끝나는 경우가 많아서
    # 앞/뒤 한 프레임까지 확장
    expanded_angle_jump = (
        angle_jump
        |
        angle_jump.shift(
            1,
            fill_value=False
        )
        |
        angle_jump.shift(
            -1,
            fill_value=False
        )
    )

    expanded_length_jump = (
        length_ratio_jump
        |
        length_ratio_jump.shift(
            1,
            fill_value=False
        )
        |
        length_ratio_jump.shift(
            -1,
            fill_value=False
        )
    )

    # =====================================================
    # 8. 최종 유효성
    # =====================================================

    elbow_angle_valid = (
        confidence_valid
        &
        upper_valid
        &
        forearm_valid
        &
        total_valid
        &
        ~expanded_angle_jump
        &
        ~expanded_length_jump
    )

    filtered_angle = (
        elbow_angle.where(
            elbow_angle_valid,
            np.nan
        )
    )

    # =====================================================
    # 9. Invalid 이유 생성
    # =====================================================

    reasons = []

    for i in range(
        len(df)
    ):
        row_reasons = []

        if not confidence_valid.iloc[i]:
            row_reasons.append(
                "low_confidence"
            )

        if not upper_valid.iloc[i]:
            row_reasons.append(
                "short_upper_arm"
            )

        if not forearm_valid.iloc[i]:
            row_reasons.append(
                "short_forearm"
            )

        if not total_valid.iloc[i]:
            row_reasons.append(
                "short_total_arm"
            )

        if expanded_angle_jump.iloc[i]:
            row_reasons.append(
                "angle_jump"
            )

        if expanded_length_jump.iloc[i]:
            row_reasons.append(
                "arm_ratio_jump"
            )

        if row_reasons:
            reasons.append(
                "|".join(
                    row_reasons
                )
            )
        else:
            reasons.append(
                "valid"
            )

    # =====================================================
    # 10. 새 컬럼을 한꺼번에 생성
    # =====================================================

    validation_df = pd.DataFrame(
        {
            "throwing_upper_arm_length":
                upper_series,

            "throwing_forearm_length":
                forearm_series,

            "throwing_total_arm_length":
                total_series,

            "upper_arm_length_ratio":
                upper_ratio,

            "forearm_length_ratio":
                forearm_ratio,

            "total_arm_length_ratio":
                total_ratio,

            "forearm_upperarm_ratio":
                arm_segment_ratio,

            "elbow_angle_conf_valid":
                confidence_valid,

            "upper_arm_length_valid":
                upper_valid,

            "forearm_length_valid":
                forearm_valid,

            "total_arm_length_valid":
                total_valid,

            "elbow_angle_diff":
                angle_diff,

            "elbow_angle_jump":
                expanded_angle_jump,

            "arm_ratio_diff":
                ratio_diff,

            "arm_ratio_jump":
                expanded_length_jump,

            "throwing_elbow_angle_valid":
                elbow_angle_valid,

            "throwing_elbow_angle_filtered":
                filtered_angle,

            "elbow_angle_invalid_reason":
                reasons,
        },
        index=df.index
    )

    result_df = pd.concat(
        [
            df.copy(),
            validation_df
        ],
        axis=1
    )

    # =====================================================
    # 11. 저장
    # =====================================================

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

    # =====================================================
    # 12. 각도 검증 그래프
    # =====================================================

    frames = result_df[
        "frame"
    ]

    invalid_mask = (
        ~result_df[
            "throwing_elbow_angle_valid"
        ]
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        result_df[
            "throwing_elbow_angle"
        ],
        label="Original Elbow Angle",
        alpha=0.45
    )

    plt.plot(
        frames,
        result_df[
            "throwing_elbow_angle_filtered"
        ],
        label="Validated Elbow Angle",
        linewidth=2
    )

    plt.scatter(
        result_df.loc[
            invalid_mask,
            "frame"
        ],
        result_df.loc[
            invalid_mask,
            "throwing_elbow_angle"
        ],
        label="Invalid Angle",
        s=28
    )

    plt.xlabel("Frame")
    plt.ylabel("Angle (degree)")

    plt.title(
        "Throwing Elbow Angle Validation v2"
    )

    plt.ylim(
        0,
        190
    )

    plt.legend()
    plt.grid()

    angle_graph_path = os.path.join(
        graph_dir,
        "elbow_angle_validation_v2.png"
    )

    plt.savefig(
        angle_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 13. 팔 길이 비율 그래프
    # =====================================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        upper_ratio,
        label="Upper Arm Ratio"
    )

    plt.plot(
        frames,
        forearm_ratio,
        label="Forearm Ratio"
    )

    plt.plot(
        frames,
        total_ratio,
        label="Total Arm Ratio"
    )

    plt.plot(
        frames,
        arm_segment_ratio,
        label="Forearm / Upper Arm Ratio"
    )

    plt.axhline(
        y=min_upper_arm_ratio,
        linestyle="--",
        label="Upper Arm Min"
    )

    plt.axhline(
        y=min_forearm_ratio,
        linestyle=":",
        label="Forearm Min"
    )

    plt.xlabel("Frame")
    plt.ylabel("Ratio")

    plt.title(
        "Projected Arm Length Validation v2"
    )

    plt.legend()
    plt.grid()

    length_graph_path = os.path.join(
        graph_dir,
        "arm_length_validation_v2.png"
    )

    plt.savefig(
        length_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 14. 변화량 진단 그래프
    # =====================================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        angle_diff,
        label="Elbow Angle Difference"
    )

    plt.plot(
        frames,
        angle_jump_threshold,
        label="Dynamic Angle Threshold"
    )

    plt.xlabel("Frame")
    plt.ylabel("Degree / frame")

    plt.title(
        "Elbow Angle Temporal Consistency"
    )

    plt.legend()
    plt.grid()

    temporal_graph_path = os.path.join(
        graph_dir,
        "elbow_angle_temporal_validation.png"
    )

    plt.savefig(
        temporal_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # 15. 통계
    # =====================================================

    valid_count = int(
        elbow_angle_valid.sum()
    )

    invalid_count = int(
        (
            ~elbow_angle_valid
        ).sum()
    )

    confidence_invalid = int(
        (
            ~confidence_valid
        ).sum()
    )

    upper_invalid = int(
        (
            ~upper_valid
        ).sum()
    )

    forearm_invalid = int(
        (
            ~forearm_valid
        ).sum()
    )

    total_invalid = int(
        (
            ~total_valid
        ).sum()
    )

    angle_jump_count = int(
        expanded_angle_jump.sum()
    )

    ratio_jump_count = int(
        expanded_length_jump.sum()
    )

    print()
    print(
        "===== Angle Validation v2 Summary ====="
    )

    print(
        f"유효 각도 프레임   : "
        f"{valid_count}"
    )

    print(
        f"무효 각도 프레임   : "
        f"{invalid_count}"
    )

    print()
    print(
        f"Confidence 문제    : "
        f"{confidence_invalid}"
    )

    print(
        f"Upper arm 축소     : "
        f"{upper_invalid}"
    )

    print(
        f"Forearm 축소       : "
        f"{forearm_invalid}"
    )

    print(
        f"Total arm 축소     : "
        f"{total_invalid}"
    )

    print(
        f"각도 급변 후보     : "
        f"{angle_jump_count}"
    )

    print(
        f"팔 비율 급변 후보  : "
        f"{ratio_jump_count}"
    )

    print()
    print(
        f"결과 CSV : "
        f"{output_csv_path}"
    )

    print(
        f"각도 그래프 : "
        f"{angle_graph_path}"
    )

    print(
        f"팔 길이 그래프 : "
        f"{length_graph_path}"
    )

    print(
        f"시간 연속성 그래프 : "
        f"{temporal_graph_path}"
    )

    print()
    print(
        "[완료] Angle Validation v2 완료"
    )


if __name__ == "__main__":

    from pipeline_config import (
        get_throwing_side
    )

    INPUT_CSV = (
        "data/pose_csv/"
        "sample_joint_angles.csv"
    )

    OUTPUT_CSV = (
        "data/pose_csv/"
        "sample_joint_angles_validated_v2.csv"
    )

    GRAPH_DIR = (
        "results/graphs/"
        "sample_angle_validation_v2"
    )

    THROWING_SIDE = (
        get_throwing_side()
    )

    validate_elbow_angles_v2(
        csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        graph_dir=GRAPH_DIR,
        throwing_side=THROWING_SIDE,

        min_joint_conf=0.30,

        min_upper_arm_ratio=0.45,
        min_forearm_ratio=0.45,
        min_total_arm_ratio=0.50,

        angle_jump_multiplier=4.0,
        min_angle_jump_threshold=25.0,

        length_ratio_jump_multiplier=4.0,
        min_length_ratio_jump_threshold=0.20,
    )