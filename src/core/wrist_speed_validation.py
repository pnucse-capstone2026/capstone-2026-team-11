import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rolling_dynamic_threshold(
    series,
    window=9,
    multiplier=4.0,
    minimum_threshold=0.8
):
    """
    주변 프레임의 변화량을 기준으로
    동적 이상치 threshold를 만든다.
    """

    local_median = (
        series
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .median()
    )

    threshold = np.maximum(
        local_median * multiplier,
        minimum_threshold
    )

    return threshold


def validate_wrist_speed(
    csv_path,
    output_csv_path,
    graph_dir,
    throwing_side="left",

    speed_window=9,
    speed_multiplier=4.0,
    min_speed_threshold_body_s=0.8,

    expand_invalid_frames=1
):
    # ========================================================
    # 1. 입력 파일 확인
    # ========================================================

    if not os.path.exists(csv_path):
        print(
            f"[오류] CSV 파일을 찾을 수 없습니다: "
            f"{csv_path}"
        )
        return

    df = pd.read_csv(csv_path)

    print()
    print(
        "===== [STEP 8.5] Wrist Speed Validation ====="
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
        "============================================="
    )

    wrist = f"{throwing_side}_wrist"

    required_columns = [
        "frame",
        "timestamp",

        f"{wrist}_invalid",

        "throwing_elbow_angle_valid",

        "wrist_relative_speed_px_s",
        "wrist_relative_speed_px_s_smooth",

        "wrist_relative_speed_body_s",
        "wrist_relative_speed_body_s_smooth",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print()
        print(
            "[오류] 필요한 컬럼이 없습니다."
        )

        for col in missing:
            print(
                " -",
                col
            )

        return

    # ========================================================
    # 2. 기존 wrist invalid 정보
    # ========================================================

    wrist_invalid = (
        df[
            f"{wrist}_invalid"
        ]
        .astype(bool)
    )

    wrist_valid = (
        ~wrist_invalid
    )

    # ========================================================
    # 3. 이전 프레임도 valid인지 확인
    # ========================================================

    previous_wrist_valid = (
        wrist_valid
        .shift(
            1,
            fill_value=False
        )
    )

    # 속도는 이전 → 현재 프레임 변화량이므로
    # 둘 다 valid여야 함
    coordinate_pair_valid = (
        wrist_valid
        &
        previous_wrist_valid
    )

    # ========================================================
    # 4. 팔꿈치 각도 valid 여부
    # ========================================================

    elbow_valid = (
        df[
            "throwing_elbow_angle_valid"
        ]
        .astype(bool)
    )

    previous_elbow_valid = (
        elbow_valid
        .shift(
            1,
            fill_value=False
        )
    )

    elbow_pair_valid = (
        elbow_valid
        &
        previous_elbow_valid
    )

    # ========================================================
    # 5. 기본 속도 valid
    # ========================================================

    basic_speed_valid = (
        coordinate_pair_valid
        &
        elbow_pair_valid
    )

    # ========================================================
    # 6. 속도 자체의 이상치 검사
    # ========================================================

    speed_series = (
        df[
            "wrist_relative_speed_body_s_smooth"
        ]
        .copy()
    )

    speed_threshold = (
        rolling_dynamic_threshold(
            speed_series,
            window=speed_window,
            multiplier=speed_multiplier,
            minimum_threshold=(
                min_speed_threshold_body_s
            )
        )
    )

    speed_outlier = (
        speed_series
        > speed_threshold
    )

    speed_outlier = (
        speed_outlier
        .fillna(False)
    )

    # ========================================================
    # 7. 속도 이상치 주변 프레임 확장
    # ========================================================

    expanded_speed_outlier = (
        speed_outlier.copy()
    )

    if expand_invalid_frames > 0:

        for shift_value in range(
            1,
            expand_invalid_frames + 1
        ):

            expanded_speed_outlier = (
                expanded_speed_outlier
                |
                speed_outlier.shift(
                    shift_value,
                    fill_value=False
                )
                |
                speed_outlier.shift(
                    -shift_value,
                    fill_value=False
                )
            )

    # ========================================================
    # 8. 최종 speed valid
    # ========================================================

    final_speed_valid = (
        basic_speed_valid
        &
        ~expanded_speed_outlier
    )

    # ========================================================
    # 9. validated speed
    # ========================================================

    validated_speed_body = (
        speed_series
        .where(
            final_speed_valid,
            np.nan
        )
    )

    validated_speed_px = (
        df[
            "wrist_relative_speed_px_s_smooth"
        ]
        .where(
            final_speed_valid,
            np.nan
        )
    )

    # ========================================================
    # 10. Invalid 이유
    # ========================================================

    reasons = []

    for i in range(
        len(df)
    ):

        row_reasons = []

        if not coordinate_pair_valid.iloc[i]:
            row_reasons.append(
                "invalid_wrist_coordinate"
            )

        if not elbow_pair_valid.iloc[i]:
            row_reasons.append(
                "invalid_elbow_angle"
            )

        if expanded_speed_outlier.iloc[i]:
            row_reasons.append(
                "speed_outlier"
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

    # ========================================================
    # 11. 최대 속도 후보 다시 찾기
    # ========================================================

    valid_speed_series = (
        validated_speed_body
        .dropna()
    )

    if len(
        valid_speed_series
    ) == 0:

        print()
        print(
            "[오류] 유효한 손목 속도 프레임이 없습니다."
        )

        return

    peak_idx = (
        valid_speed_series
        .idxmax()
    )

    peak_frame = int(
        df.loc[
            peak_idx,
            "frame"
        ]
    )

    peak_time = float(
        df.loc[
            peak_idx,
            "timestamp"
        ]
    )

    peak_speed_body = float(
        validated_speed_body.loc[
            peak_idx
        ]
    )

    peak_speed_px = float(
        validated_speed_px.loc[
            peak_idx
        ]
    )

    peak_candidate = pd.Series(
        False,
        index=df.index
    )

    peak_candidate.loc[
        peak_idx
    ] = True

    # ========================================================
    # 12. 새 컬럼 생성
    # ========================================================

    validation_df = pd.DataFrame(
        {
            "wrist_coordinate_pair_valid":
                coordinate_pair_valid,

            "wrist_elbow_pair_valid":
                elbow_pair_valid,

            "wrist_speed_dynamic_threshold":
                speed_threshold,

            "wrist_speed_outlier":
                expanded_speed_outlier,

            "wrist_speed_valid":
                final_speed_valid,

            "wrist_relative_speed_px_s_validated":
                validated_speed_px,

            "wrist_relative_speed_body_s_validated":
                validated_speed_body,

            "validated_wrist_peak_candidate":
                peak_candidate,

            "wrist_speed_invalid_reason":
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

    # ========================================================
    # 13. 출력 폴더
    # ========================================================

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

    # ========================================================
    # 14. CSV 저장
    # ========================================================

    result_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    frames = (
        result_df[
            "frame"
        ]
    )

    # ========================================================
    # GRAPH 1
    # 원본 속도 + validated 속도
    # ========================================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        result_df[
            "wrist_relative_speed_body_s_smooth"
        ],
        label=(
            "Original Normalized Wrist Speed"
        ),
        alpha=0.45
    )

    plt.plot(
        frames,
        result_df[
            "wrist_relative_speed_body_s_validated"
        ],
        label=(
            "Validated Wrist Speed"
        ),
        linewidth=2
    )

    invalid_mask = (
        ~result_df[
            "wrist_speed_valid"
        ]
    )

    plt.scatter(
        result_df.loc[
            invalid_mask,
            "frame"
        ],
        result_df.loc[
            invalid_mask,
            "wrist_relative_speed_body_s_smooth"
        ],
        label="Invalid Speed",
        s=25
    )

    plt.axvline(
        x=peak_frame,
        linestyle="--",
        label=(
            f"Validated Peak "
            f"(Frame {peak_frame})"
        )
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Body Length / second"
    )

    plt.title(
        "Validated Wrist Speed"
    )

    plt.legend()

    plt.grid()

    validated_graph_path = os.path.join(
        graph_dir,
        "wrist_speed_validated.png"
    )

    plt.savefig(
        validated_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # GRAPH 2
    # 속도 threshold 진단
    # ========================================================

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        frames,
        speed_series,
        label="Wrist Speed"
    )

    plt.plot(
        frames,
        speed_threshold,
        label="Dynamic Threshold"
    )

    plt.scatter(
        result_df.loc[
            expanded_speed_outlier,
            "frame"
        ],
        result_df.loc[
            expanded_speed_outlier,
            "wrist_relative_speed_body_s_smooth"
        ],
        label="Speed Outlier",
        s=28
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Body Length / second"
    )

    plt.title(
        "Wrist Speed Outlier Detection"
    )

    plt.legend()

    plt.grid()

    threshold_graph_path = (
        os.path.join(
            graph_dir,
            "wrist_speed_outlier_detection.png"
        )
    )

    plt.savefig(
        threshold_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # GRAPH 3
    # Valid / Invalid 상태
    # ========================================================

    plt.figure(
        figsize=(14, 4)
    )

    valid_numeric = (
        result_df[
            "wrist_speed_valid"
        ]
        .astype(int)
    )

    plt.step(
        frames,
        valid_numeric,
        where="mid",
        label="Speed Validity"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Valid"
    )

    plt.yticks(
        [0, 1],
        [
            "Invalid",
            "Valid"
        ]
    )

    plt.title(
        "Wrist Speed Validity"
    )

    plt.grid()

    plt.legend()

    validity_graph_path = (
        os.path.join(
            graph_dir,
            "wrist_speed_validity.png"
        )
    )

    plt.savefig(
        validity_graph_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 15. 통계
    # ========================================================

    valid_count = int(
        final_speed_valid.sum()
    )

    invalid_count = int(
        (
            ~final_speed_valid
        ).sum()
    )

    wrist_coordinate_invalid_count = int(
        (
            ~coordinate_pair_valid
        ).sum()
    )

    elbow_invalid_count = int(
        (
            ~elbow_pair_valid
        ).sum()
    )

    speed_outlier_count = int(
        expanded_speed_outlier.sum()
    )

    print()
    print(
        "===== Wrist Speed Validation Summary ====="
    )

    print(
        f"유효 속도 프레임      : "
        f"{valid_count}"
    )

    print(
        f"무효 속도 프레임      : "
        f"{invalid_count}"
    )

    print()
    print(
        f"Wrist 좌표 문제       : "
        f"{wrist_coordinate_invalid_count}"
    )

    print(
        f"Elbow angle 문제      : "
        f"{elbow_invalid_count}"
    )

    print(
        f"속도 급변 후보        : "
        f"{speed_outlier_count}"
    )

    print()
    print(
        "===== Validated Peak Candidate ====="
    )

    print(
        f"프레임       : "
        f"{peak_frame}"
    )

    print(
        f"시간         : "
        f"{peak_time:.3f} sec"
    )

    print(
        f"상대 속도    : "
        f"{peak_speed_px:.2f} px/s"
    )

    print(
        f"정규화 속도  : "
        f"{peak_speed_body:.3f} body/s"
    )

    print()
    print(
        "※ 이 값도 아직 Ball Release 확정값은 아닙니다."
    )

    print(
        "   이후 Pitching Phase Detection과 "
        "결합해서 Release 후보로 사용합니다."
    )

    print()
    print(
        f"결과 CSV : "
        f"{output_csv_path}"
    )

    print(
        f"Validated 그래프 : "
        f"{validated_graph_path}"
    )

    print(
        f"Outlier 그래프   : "
        f"{threshold_graph_path}"
    )

    print(
        f"Validity 그래프  : "
        f"{validity_graph_path}"
    )

    print()
    print(
        "[완료] Wrist Speed Validation 완료"
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
        "sample_wrist_motion.csv"
    )

    OUTPUT_CSV = (
        "data/pose_csv/"
        "sample_wrist_motion_validated.csv"
    )

    GRAPH_DIR = (
        "results/graphs/"
        "sample_wrist_speed_validation"
    )

    THROWING_SIDE = (
        get_throwing_side()
    )

    validate_wrist_speed(
        csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        graph_dir=GRAPH_DIR,
        throwing_side=THROWING_SIDE,

        speed_window=9,
        speed_multiplier=4.0,

        min_speed_threshold_body_s=0.8,

        expand_invalid_frames=1
    )