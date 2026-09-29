import cv2
import os

def analyze_video_info(video_path, output_frame_dir, sample_interval=10):
    if not os.path.exists(video_path):
        print(f"[오류] 영상 파일을 찾을 수 없습니다: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[오류] 영상 파일 스트림을 열 수 없습니다.")
        return

    # 영상 속성 읽기
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    print("===== [STEP 1] 영상 기본 정보 확인 =====")
    print(f" 파일 경로   : {video_path}")
    print(f" 해상도     : {width} x {height}")
    print(f" FPS        : {fps:.2f}")
    print(f" 총 프레임 수: {total_frames} frames")
    print(f" 총 재생 시간: {duration:.2f} seconds")
    print("=========================================")

    # 검증용 프레임 이미지 추출 및 저장
    os.makedirs(output_frame_dir, exist_ok=True)
    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            out_path = os.path.join(output_frame_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"[완료] 총 {saved_count}개의 테스트 프레임이 '{output_frame_dir}'에 저장되었습니다.\n")

if __name__ == "__main__":
    # 사용자 환경에 맞는 영상 경로 지정
    from pipeline_config import (
        get_video_path
    )

    VIDEO_PATH = (
        get_video_path()
    )
    
    OUTPUT_DIR = "results/frames/sample_frames"
    
    # 10프레임 간격으로 검증용 샘플 이미지 저장
    analyze_video_info(VIDEO_PATH, OUTPUT_DIR, sample_interval=10)