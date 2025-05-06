import cv2
import time
import os


def process_videos(root_path, output_path, fps=10, chute_count=24, cam_count=8, video_format="avi"):
    """
    Processes video files from a specified root path, changing their frame rate
    and saving them to an output path.

    Args:
        root_path (str): The root directory containing the chute folders.
        output_path (str): The directory where the processed videos will be saved.
        fps (int, optional): The desired frame rate for the output videos. Defaults to 10.
        chute_count (int, optional): The number of chute folders to process. Defaults to 24.
        cam_count (int, optional): The number of cameras per chute. Defaults to 8.
        video_format (str, optional): The video file format. Defaults to "avi".
    """
    os.makedirs(output_path, exist_ok=True)

    for chute_idx in range(1, chute_count + 1):
        chute_folder = os.path.join(
            root_path, f"chute{str(chute_idx).zfill(2)}")
        output_chute_folder = os.path.join(
            output_path, f"chute{str(chute_idx).zfill(2)}")

        os.makedirs(output_chute_folder, exist_ok=True)

        for cam_idx in range(1, cam_count + 1):
            video_path = os.path.join(
                chute_folder, f"cam{cam_idx}.{video_format}")
            output_video_path = os.path.join(
                output_chute_folder, f"cam{cam_idx}.{video_format}")

            if not os.path.exists(video_path):
                print(f"파일 없음: {video_path}")
                continue

            print(f"변환 중: {video_path} -> {output_video_path}")
            video = cv2.VideoCapture(video_path)

            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = video.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")

            out = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height))

            prev_time = 0
            frame_interval = int(original_fps // fps)

            frame_count = 0
            while video.isOpened():
                ret, frame = video.read()

                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    out.write(frame)

                frame_count += 1

            video.release()
            out.release()

    print("영상 변환 및 저장 완료")


if __name__ == "__main__":
    ROOT_PATH = "..\\Dataset\\Data"
    OUTPUT_PATH = "..\\Dataset\\Processed"
    FPS = 10
    CHUTE_COUNT = 24
    CAM_COUNT = 8
    VIDEO_FORMAT = "avi"

    process_videos(ROOT_PATH, OUTPUT_PATH, FPS,
                   CHUTE_COUNT, CAM_COUNT, VIDEO_FORMAT)
