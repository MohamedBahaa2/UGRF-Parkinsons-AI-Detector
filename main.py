# src/extract_features.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stride_lengths = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            stride = np.linalg.norm(np.array([left_heel.x, left_heel.y]) - np.array([right_heel.x, right_heel.y]))
            stride_lengths.append(stride)

    cap.release()
    avg_stride = np.mean(stride_lengths) if stride_lengths else 0
    return {"avg_stride_length": avg_stride, "total_frames": frame_count}

def process_all_videos(input_folder, output_csv):
    data = []
    for file in os.listdir(input_folder):
        if file.endswith(".mp4") or file.endswith(".avi"):
            video_path = os.path.join(input_folder, file)
            print(f"Processing: {file}")
            features = extract_keypoints_from_video(video_path)
            features["filename"] = file
            data.append(features)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved features to: {output_csv}")

if __name__ == "__main__":
    process_all_videos("data", "features/features.csv")

    print("[✓] Done")