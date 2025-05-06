# %% [code]
import cv2
import os
import pandas as pd
import mediapipe as mp
import torch


def skeleton_csv(video_path, output_csv_path, output_csv_name='skeleton_video.csv'):
    """
    mediapipe를 사용해 비디오에서 18개 스켈레톤 랜드마크를 검출하여,
    각 프레임의 부위별 데이터(부위 이름, x, y, z, 정확도)를 CSV 파일로 저장하는 함수.

    Args:
        video_path (str): 입력 비디오 파일의 절대 경로.
        output_csv_path (str): 결과 CSV 파일을 저장할 디렉토리.
        output_csv_name (str): 생성할 CSV 파일의 이름 (기본: 'skeleton_video.csv').

    Returns:
        DataFrame: 생성된 CSV 데이터 (각 행: frame, landmark, x, y, z, accuracy).
                 오류 발생 시 None을 반환.
    """
    try:
        # Mediapipe Pose 초기화
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                            enable_segmentation=False, min_detection_confidence=0.5)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file at path:", video_path)
            return None

        # 추출할 부위 정보: (부위 이름, mediapipe 인덱스 또는 'neck' 처리)
        landmarks_info = [
            ("Nose", 0),
            ("Neck", "neck"),  # 왼쪽 어깨(11)와 오른쪽 어깨(12)의 평균
            ("Right Shoulder", 12),
            ("Right Elbow", 14),
            ("Right Wrist", 16),
            ("Left Shoulder", 11),
            ("Left Elbow", 13),
            ("Left Wrist", 15),
            ("Right Hip", 24),
            ("Right Knee", 26),
            ("Right Ankle", 28),
            ("Left Hip", 23),
            ("Left Knee", 25),
            ("Left Ankle", 27),
            ("Right Eye", 5),
            ("Left Eye", 2),
            ("Right Ear", 8),
            ("Left Ear", 7)
        ]

        # CSV 데이터 저장 리스트 (각 행: 프레임번호, 부위 이름, x, y, z, 정확도)
        csv_rows = []
        frame_idx = 0

        while True:
            success, image = cap.read()
            if not success:
                # print("모든 프레임 처리 완료, 총 프레임 수:", frame_idx)
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # 기준점: 좌우 엉덩이(landmark 23, 24)의 중간값
                ref_x = (
                    results.pose_landmarks.landmark[23].x + results.pose_landmarks.landmark[24].x) / 2
                ref_y = (
                    results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2
                ref_z = (
                    results.pose_landmarks.landmark[23].z + results.pose_landmarks.landmark[24].z) / 2

                for part, idx in landmarks_info:
                    if idx == "neck":
                        left_shoulder = results.pose_landmarks.landmark[11]
                        right_shoulder = results.pose_landmarks.landmark[12]
                        x_val = (left_shoulder.x + right_shoulder.x) / 2
                        y_val = (left_shoulder.y + right_shoulder.y) / 2
                        z_val = (left_shoulder.z + right_shoulder.z) / 2
                        visibility = (left_shoulder.visibility +
                                      right_shoulder.visibility) / 2
                    else:
                        lm = results.pose_landmarks.landmark[idx]
                        x_val = lm.x
                        y_val = lm.y
                        z_val = lm.z
                        visibility = lm.visibility
                    # 상대 좌표 계산 (기준점 빼기)
                    rel_x = x_val - ref_x
                    rel_y = y_val - ref_y
                    rel_z = z_val - ref_z
                    csv_rows.append({
                        "frame": frame_idx,
                        "landmark": part,
                        "x": rel_x,
                        "y": rel_y,
                        "z": rel_z,
                        "accuracy": visibility
                    })
            frame_idx += 1

        cap.release()
        pose.close()

        # 결과 CSV 디렉토리 없으면 생성
        if not os.path.exists(output_csv_path):
            os.makedirs(output_csv_path)
        csv_full_path = os.path.join(output_csv_path, output_csv_name)
        df = pd.DataFrame(csv_rows, columns=[
                          "frame", "landmark", "x", "y", "z", "accuracy"])
        df.to_csv(csv_full_path, index=False)
        # print("CSV 파일 저장 완료:", csv_full_path)
        return df

    except Exception as e:
        print("오류 발생:", e)
        return None
