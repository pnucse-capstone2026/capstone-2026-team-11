import os
import pandas as pd
import matplotlib.pyplot as plt


def validate_pose_csv(
    csv_path,
    output_dir,
    throwing_side="left"
):
    # -----------------------------------
    # 1. CSV 존재 여부 확인
    # -----------------------------------
    if not os.path.exists(csv_path):
        print(f"[오류] CSV 파일을 찾을 수 없습니다: {csv_path}")
        return

    # -----------------------------------
    # 2. CSV 읽기
    # -----------------------------------
    df = pd.read_csv(csv_path)

    print("===== [STEP 4] Pose Validation =====")
    print(f"CSV 경로   : {csv_path}")
    print(f"총 프레임 수: {len(df)}")
    print(f"총 컬럼 수 : {len(df.columns)}")
    print("====================================")

    # -----------------------------------
    # 3. 투구팔 설정
    # -----------------------------------
    if throwing_side not in ["left", "right"]:
        print("[오류] throwing_side는 'left' 또는 'right'만 가능합니다.")
        return

    shoulder = f"{throwing_side}_shoulder"
    elbow = f"{throwing_side}_elbow"
    wrist = f"{throwing_side}_wrist"

    required_columns = [
        "frame",
        "timestamp",

        f"{shoulder}_x",
        f"{shoulder}_y",
        f"{shoulder}_conf",

        f"{elbow}_x",
        f"{elbow}_y",
        f"{elbow}_conf",

        f"{wrist}_x",
        f"{wrist}_y",
        f"{wrist}_conf",
    ]

    # 필요한 컬럼 확인
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("[오류] 다음 컬럼이 CSV에 없습니다.")
        for col in missing_columns:
            print(" -", col)
        return

    # -----------------------------------
    # 4. 출력 폴더 생성
    # -----------------------------------
    os.makedirs(output_dir, exist_ok=True)

    frames = df["frame"]

    # ==================================================
    # GRAPH 1
    # Shoulder X / Y
    # ==================================================
    plt.figure(figsize=(12, 6))

    plt.plot(
        frames,
        df[f"{shoulder}_x"],
        label=f"{shoulder}_x"
    )

    plt.plot(
        frames,
        df[f"{shoulder}_y"],
        label=f"{shoulder}_y"
    )

    plt.xlabel("Frame")
    plt.ylabel("Pixel Coordinate")
    plt.title(
        f"{throwing_side.capitalize()} Shoulder Position"
    )
    plt.legend()
    plt.grid()

    shoulder_path = os.path.join(
        output_dir,
        f"{throwing_side}_shoulder_xy.png"
    )

    plt.savefig(
        shoulder_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # GRAPH 2
    # Elbow X / Y
    # ==================================================
    plt.figure(figsize=(12, 6))

    plt.plot(
        frames,
        df[f"{elbow}_x"],
        label=f"{elbow}_x"
    )

    plt.plot(
        frames,
        df[f"{elbow}_y"],
        label=f"{elbow}_y"
    )

    plt.xlabel("Frame")
    plt.ylabel("Pixel Coordinate")
    plt.title(
        f"{throwing_side.capitalize()} Elbow Position"
    )
    plt.legend()
    plt.grid()

    elbow_path = os.path.join(
        output_dir,
        f"{throwing_side}_elbow_xy.png"
    )

    plt.savefig(
        elbow_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # GRAPH 3
    # Wrist X / Y
    # ==================================================
    plt.figure(figsize=(12, 6))

    plt.plot(
        frames,
        df[f"{wrist}_x"],
        label=f"{wrist}_x"
    )

    plt.plot(
        frames,
        df[f"{wrist}_y"],
        label=f"{wrist}_y"
    )

    plt.xlabel("Frame")
    plt.ylabel("Pixel Coordinate")
    plt.title(
        f"{throwing_side.capitalize()} Wrist Position"
    )
    plt.legend()
    plt.grid()

    wrist_path = os.path.join(
        output_dir,
        f"{throwing_side}_wrist_xy.png"
    )

    plt.savefig(
        wrist_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # GRAPH 4
    # Confidence
    # ==================================================
    plt.figure(figsize=(12, 6))

    plt.plot(
        frames,
        df[f"{shoulder}_conf"],
        label="Shoulder confidence"
    )

    plt.plot(
        frames,
        df[f"{elbow}_conf"],
        label="Elbow confidence"
    )

    plt.plot(
        frames,
        df[f"{wrist}_conf"],
        label="Wrist confidence"
    )

    # confidence 0.5 기준선
    plt.axhline(
        y=0.5,
        linestyle="--",
        label="Confidence 0.5"
    )

    plt.xlabel("Frame")
    plt.ylabel("Confidence")
    plt.title(
        f"{throwing_side.capitalize()} Arm Keypoint Confidence"
    )

    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid()

    confidence_path = os.path.join(
        output_dir,
        f"{throwing_side}_arm_confidence.png"
    )

    plt.savefig(
        confidence_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ==================================================
    # 5. 간단한 통계 출력
    # ==================================================
    print()
    print("===== Confidence Statistics =====")

    joints = [
        shoulder,
        elbow,
        wrist
    ]

    for joint in joints:
        conf_col = f"{joint}_conf"

        mean_conf = df[conf_col].mean()

        low_conf_count = (
            df[conf_col] < 0.5
        ).sum()

        print()
        print(joint)
        print(f" 평균 confidence : {mean_conf:.4f}")
        print(
            f" confidence < 0.5 : "
            f"{low_conf_count} frames"
        )

    print()
    print("===== 저장된 그래프 =====")
    print(shoulder_path)
    print(elbow_path)
    print(wrist_path)
    print(confidence_path)

    print()
    print("[완료] Pose 데이터 검증 그래프 생성 완료")


if __name__ == "__main__":

    from pipeline_config import (
        get_throwing_side
    )

    CSV_PATH = (
        "data/pose_csv/"
        "sample_pose.csv"
    )

    OUTPUT_DIR = (
        "results/graphs/"
        "sample_pose_validation"
    )

    THROWING_SIDE = (
        get_throwing_side()
    )

    validate_pose_csv(
        CSV_PATH,
        OUTPUT_DIR,
        throwing_side=THROWING_SIDE
    )