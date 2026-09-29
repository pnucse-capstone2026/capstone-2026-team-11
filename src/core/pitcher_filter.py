import cv2
import os
import numpy as np
from ultralytics import YOLO


def box_center(box):
    x1, y1, x2, y2 = box

    return np.array([
        (x1 + x2) / 2,
        (y1 + y2) / 2
    ])


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0

    return intersection / union


def run_pitcher_pose(
    video_path,
    output_video_path,
    model_name="yolo11n-pose.pt",
    confidence=0.5
):
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[오류] 영상을 열 수 없습니다.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # -----------------------------------
    # 첫 프레임 읽기
    # -----------------------------------
    ret, first_frame = cap.read()

    if not ret:
        print("[오류] 첫 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    print()
    print("첫 프레임에서 투수의 몸 전체를 드래그하세요.")
    print("선택 후 ENTER 또는 SPACE를 누르세요.")
    print()

    # 마우스로 투수 영역 직접 선택
    roi = cv2.selectROI(
        "Select Pitcher",
        first_frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Select Pitcher")

    roi_x, roi_y, roi_w, roi_h = roi

    initial_roi = np.array([
        roi_x,
        roi_y,
        roi_x + roi_w,
        roi_y + roi_h
    ], dtype=float)

    if roi_w == 0 or roi_h == 0:
        print("[오류] 투수 영역이 선택되지 않았습니다.")
        cap.release()
        return

    # 다시 처음부터 읽기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    os.makedirs(
        os.path.dirname(output_video_path),
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    previous_pitcher_box = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            conf=confidence,
            verbose=False
        )

        result = results[0]

        output_frame = frame.copy()

        pitcher_index = None

        if (
            result.boxes is not None
            and result.keypoints is not None
            and len(result.boxes) > 0
        ):

            boxes = result.boxes.xyxy.cpu().numpy()

            keypoints = (
                result.keypoints.xy.cpu().numpy()
            )

            # ---------------------------------
            # 첫 프레임:
            # 사용자가 지정한 ROI와 가장 많이
            # 겹치는 사람을 투수로 선택
            # ---------------------------------
            if previous_pitcher_box is None:

                best_iou = 0

                for i, box in enumerate(boxes):

                    iou = calculate_iou(
                        box,
                        initial_roi
                    )

                    if iou > best_iou:
                        best_iou = iou
                        pitcher_index = i

            # ---------------------------------
            # 이후 프레임:
            # 이전 투수 위치와 가장 가까운 사람
            # ---------------------------------
            else:

                previous_center = box_center(
                    previous_pitcher_box
                )

                best_distance = float("inf")

                for i, box in enumerate(boxes):

                    current_center = box_center(
                        box
                    )

                    distance = np.linalg.norm(
                        current_center
                        - previous_center
                    )

                    if distance < best_distance:
                        best_distance = distance
                        pitcher_index = i

            # ---------------------------------
            # 선택된 투수 표시
            # ---------------------------------
            if pitcher_index is not None:

                pitcher_box = boxes[pitcher_index]

                previous_pitcher_box = pitcher_box.copy()

                x1, y1, x2, y2 = pitcher_box

                cv2.rectangle(
                    output_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    output_frame,
                    "Pitcher",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                points = keypoints[
                    pitcher_index
                ]

                for x, y in points:

                    if x > 0 and y > 0:

                        cv2.circle(
                            output_frame,
                            (int(x), int(y)),
                            5,
                            (0, 255, 255),
                            -1
                        )

        cv2.putText(
            output_frame,
            f"Frame: {frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        writer.write(output_frame)

        frame_idx += 1

        if frame_idx % 30 == 0:
            print(
                f"[진행] {frame_idx} frames 처리"
            )

    cap.release()
    writer.release()

    print()
    print("[완료] 투수 추적 결과 저장 완료")
    print(output_video_path)


if __name__ == "__main__":

    from pipeline_config import (
        get_video_path
    )

    VIDEO_PATH = (
        get_video_path()
    )
    
    OUTPUT_VIDEO_PATH = (
        "results/pose/"
        "sample_pitcher_only.mp4"
    )

    run_pitcher_pose(
        VIDEO_PATH,
        OUTPUT_VIDEO_PATH
    )