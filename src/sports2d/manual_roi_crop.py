#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import cv2


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def expand_roi(x, y, w, h, frame_w, frame_h, margin_x, margin_y):
    add_x = int(round(w * margin_x))
    add_y = int(round(h * margin_y))

    x1 = clamp(x - add_x, 0, frame_w - 1)
    y1 = clamp(y - add_y, 0, frame_h - 1)
    x2 = clamp(x + w + add_x, 1, frame_w)
    y2 = clamp(y + h + add_y, 1, frame_h)

    return x1, y1, x2, y2


def save_json(path, input_video, output_video, frame_w, frame_h, fps,
              total_frames, selected_roi, expanded_roi, margin_x, margin_y):
    sx, sy, sw, sh = selected_roi
    x1, y1, x2, y2 = expanded_roi

    data = {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "source": {
            "width": int(frame_w),
            "height": int(frame_h),
            "fps": float(fps),
            "total_frames": int(total_frames),
        },
        "selected_roi": {
            "x": int(sx), "y": int(sy),
            "width": int(sw), "height": int(sh),
        },
        "expanded_roi": {
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "offset_x": int(x1),
            "offset_y": int(y1),
        },
        "margin": {
            "margin_x_ratio": float(margin_x),
            "margin_y_ratio": float(margin_y),
        },
        "coordinate_restore": {
            "original_x": "cropped_x + offset_x",
            "original_y": "cropped_y + offset_y",
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Sports2D용 수동 투수 ROI 고정 크롭"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--margin_x", type=float, default=0.50)
    parser.add_argument("--margin_y", type=float, default=0.35)
    parser.add_argument("--codec", default="mp4v")
    args = parser.parse_args()

    if args.margin_x < 0 or args.margin_y < 0:
        raise ValueError("margin 값은 0 이상이어야 합니다.")
    if len(args.codec) != 4:
        raise ValueError("--codec은 4글자 fourcc여야 합니다.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    json_path = Path(args.json)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 영상 없음: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {input_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

    print()
    print("===== Sports2D Manual Pitcher ROI Crop =====")
    print(f"입력 영상 : {input_path}")
    print(f"해상도    : {frame_w} x {frame_h}")
    print(f"FPS       : {fps:.2f}")
    print(f"프레임 수 : {total_frames}")
    print()
    print("첫 프레임에서 투수 몸 전체를 드래그하세요.")
    print("선택 후 ENTER 또는 SPACE를 누르세요.")
    print()

    roi = cv2.selectROI(
        "Select Pitcher ROI",
        first_frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Select Pitcher ROI")

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("ROI가 선택되지 않았습니다.")

    x1, y1, x2, y2 = expand_roi(
        x, y, w, h,
        frame_w, frame_h,
        args.margin_x, args.margin_y,
    )

    crop_w = x2 - x1
    crop_h = y2 - y1

    preview = first_frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.putText(
        preview, "Selected ROI",
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )
    cv2.putText(
        preview, "Expanded Fixed Crop",
        (x1, max(50, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    print(f"선택 ROI      : x={x}, y={y}, w={w}, h={h}")
    print(f"확장 고정 ROI : x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"Crop 크기     : {crop_w} x {crop_h}")
    print(f"Offset        : x={x1}, y={y1}")
    print()
    print("초록색 박스가 최종 고정 Crop 영역입니다.")
    print("ENTER/SPACE: 진행, ESC: 취소")

    while True:
        cv2.imshow("ROI Preview", preview)
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):
            break
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            print("[취소] 작업을 종료했습니다.")
            return

    cv2.destroyAllWindows()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (crop_w, crop_h),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"출력 영상 생성 실패: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    print()
    print("[처리 시작] 고정 ROI 영상 생성")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        crop = frame[y1:y2, x1:x2]
        if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
            raise RuntimeError(f"Frame {frame_idx}: crop 크기 불일치")

        writer.write(crop)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"[진행] {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    save_json(
        json_path,
        input_path,
        output_path,
        frame_w,
        frame_h,
        fps,
        total_frames,
        (x, y, w, h),
        (x1, y1, x2, y2),
        args.margin_x,
        args.margin_y,
    )

    print()
    print("===== 완료 =====")
    print(f"ROI 영상 : {output_path}")
    print(f"ROI JSON : {json_path}")
    print(f"프레임 수: {frame_idx}")
    print()
    print("원본 좌표 복원:")
    print(f"  original_x = cropped_x + {x1}")
    print(f"  original_y = cropped_y + {y1}")


if __name__ == "__main__":
    main()
