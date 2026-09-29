import cv2
import os
import csv
import numpy as np
from ultralytics import YOLO


# YOLO Pose COCO 17 Keypoints
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

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def create_csv_header():
    header = [
        "frame",
        "timestamp"
    ]

    for name in KEYPOINT_NAMES:
        header.append(f"{name}_x")
        header.append(f"{name}_y")
        header.append(f"{name}_conf")

    return header


def run_pose_export(
    video_path,
    output_csv_path,
    model_name="yolo11n-pose.pt",
    confidence=0.5
):
    # -----------------------------------
    # 1. 영상 확인
    # -----------------------------------
    if not os.path.exists(video_path):
        print(f"[오류] 영상을 찾을 수 없습니다: {video_path}")
        return

    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[오류] 영상을 열 수 없습니다.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("===== [STEP 3] Pose CSV Export =====")
    print(f"영상       : {video_path}")
    print(f"해상도     : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"총 프레임 수: {total_frames}")
    print("====================================")

    # -----------------------------------
    # 2. 첫 프레임에서 투수 선택
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

    roi = cv2.selectROI(
        "Select Pitcher",
        first_frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Select Pitcher")

    roi_x, roi_y, roi_w, roi_h = roi

    if roi_w == 0 or roi_h == 0:
        print("[오류] 투수 영역이 선택되지 않았습니다.")
        cap.release()
        return

    initial_roi = np.array([
        roi_x,
        roi_y,
        roi_x + roi_w,
        roi_y + roi_h
    ], dtype=float)

    # 영상 처음으로 되돌리기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -----------------------------------
    # 3. CSV 폴더 생성
    # -----------------------------------
    output_dir = os.path.dirname(output_csv_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------
    # 4. CSV 파일 열기
    # -----------------------------------
    with open(
        output_csv_path,
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(create_csv_header())

        previous_pitcher_box = None
        frame_idx = 0
        detected_frames = 0
        missing_frames = 0

        # -----------------------------------
        # 5. 프레임별 Pose 추출
        # -----------------------------------
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            timestamp = (
                frame_idx / fps
                if fps > 0
                else 0.0
            )

            results = model(
                frame,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            pitcher_index = None

            row = [
                frame_idx,
                round(timestamp, 6)
            ]

            if (
                result.boxes is not None
                and result.keypoints is not None
                and len(result.boxes) > 0
            ):
                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                keypoints_xy = (
                    result.keypoints.xy
                    .cpu()
                    .numpy()
                )

                # keypoint confidence
                if result.keypoints.conf is not None:
                    keypoints_conf = (
                        result.keypoints.conf
                        .cpu()
                        .numpy()
                    )
                else:
                    keypoints_conf = None

                # -----------------------------------
                # 첫 검출:
                # 사용자가 지정한 ROI와
                # 가장 많이 겹치는 사람 선택
                # -----------------------------------
                if previous_pitcher_box is None:
                    best_iou = 0.0

                    for i, box in enumerate(boxes):
                        iou = calculate_iou(
                            box,
                            initial_roi
                        )

                        if iou > best_iou:
                            best_iou = iou
                            pitcher_index = i

                # -----------------------------------
                # 이후:
                # 이전 투수 위치와 가장 가까운 사람
                # -----------------------------------
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

            # -----------------------------------
            # 6. 투수 관절 좌표 저장
            # -----------------------------------
            if pitcher_index is not None:
                pitcher_box = boxes[pitcher_index]
                previous_pitcher_box = pitcher_box.copy()

                points = keypoints_xy[
                    pitcher_index
                ]

                if keypoints_conf is not None:
                    confs = keypoints_conf[
                        pitcher_index
                    ]
                else:
                    confs = np.zeros(
                        len(KEYPOINT_NAMES)
                    )

                for i in range(
                    len(KEYPOINT_NAMES)
                ):
                    x = float(points[i][0])
                    y = float(points[i][1])
                    conf = float(confs[i])

                    row.extend([
                        round(x, 4),
                        round(y, 4),
                        round(conf, 6)
                    ])

                detected_frames += 1

            else:
                # 투수를 못 찾은 프레임
                # 빈 값으로 채움
                for _ in KEYPOINT_NAMES:
                    row.extend([
                        "",
                        "",
                        ""
                    ])

                missing_frames += 1

            writer.writerow(row)

            frame_idx += 1

            if frame_idx % 30 == 0:
                print(
                    f"[진행] "
                    f"{frame_idx}/{total_frames} frames"
                )

    cap.release()

    print()
    print("[완료] Pose CSV 저장 완료")
    print(f"CSV 경로: {output_csv_path}")
    print(
        f"정상 검출 프레임: {detected_frames}"
    )
    print(
        f"미검출 프레임   : {missing_frames}"
    )


if __name__ == "__main__":

    from pipeline_config import (
        get_video_path
    )

    VIDEO_PATH = (
        get_video_path()
    )

    OUTPUT_CSV_PATH = (
        "data/pose_csv/sample_pose.csv"
    )

    run_pose_export(
        VIDEO_PATH,
        OUTPUT_CSV_PATH,
        model_name="yolo11n-pose.pt",
        confidence=0.5
    )