import cv2
import os
from ultralytics import YOLO


def run_pose_detection(
    video_path,
    output_video_path,
    model_name="yolo11n-pose.pt",
    confidence=0.5
):
    # -----------------------------
    # 1. 입력 영상 존재 여부 확인
    # -----------------------------
    if not os.path.exists(video_path):
        print(f"[오류] 영상 파일을 찾을 수 없습니다: {video_path}")
        return

    # -----------------------------
    # 2. YOLO Pose 모델 불러오기
    # -----------------------------
    print("[INFO] YOLO Pose 모델을 불러오는 중...")
    model = YOLO(model_name)

    # -----------------------------
    # 3. 입력 영상 열기
    # -----------------------------
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[오류] 영상 파일을 열 수 없습니다.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("===== [STEP 2] Pose Detection =====")
    print(f"영상 경로   : {video_path}")
    print(f"해상도     : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"총 프레임 수: {total_frames}")
    print("===================================")

    # -----------------------------
    # 4. 출력 폴더 생성
    # -----------------------------
    output_dir = os.path.dirname(output_video_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 5. 결과 영상 저장 객체 생성
    # -----------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("[오류] 출력 영상을 생성할 수 없습니다.")
        cap.release()
        return

    # -----------------------------
    # 6. 프레임별 Pose Detection
    # -----------------------------
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # YOLO Pose 추론
        results = model(
            frame,
            conf=confidence,
            verbose=False
        )

        # skeleton, bounding box 등이 그려진 이미지
        annotated_frame = results[0].plot()

        # 프레임 번호 표시
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # 결과 영상에 저장
        writer.write(annotated_frame)

        frame_idx += 1

        # 진행 상황 출력
        if frame_idx % 30 == 0:
            print(
                f"[진행] {frame_idx}/{total_frames} frames 처리 완료"
            )

    # -----------------------------
    # 7. 종료 처리
    # -----------------------------
    cap.release()
    writer.release()

    print()
    print("[완료] Pose Detection이 끝났습니다.")
    print(f"결과 영상: {output_video_path}")


if __name__ == "__main__":

    from pipeline_config import (
        get_video_path
    )

    VIDEO_PATH = (
        get_video_path()
    )

    OUTPUT_VIDEO_PATH = (
        "results/pose/sample_pose.mp4"
    )

    run_pose_detection(
        VIDEO_PATH,
        OUTPUT_VIDEO_PATH,
        model_name="yolo11n-pose.pt",
        confidence=0.5
    )