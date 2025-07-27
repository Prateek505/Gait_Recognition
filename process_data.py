import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def extract_gait_features(landmarks):
    """Extracts a feature vector from pose landmarks."""
    # Example features: distances between key joints
    # Normalize by a reference distance, like shoulder width, to make it scale-invariant
    
    # Key landmarks
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Reference distance for normalization (shoulder width)
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    if shoulder_width < 1e-6: # Avoid division by zero
        return None

    # Feature calculations (normalized)
    features = [
        calculate_distance(left_hip, left_knee) / shoulder_width,
        calculate_distance(right_hip, right_knee) / shoulder_width,
        calculate_distance(left_knee, left_ankle) / shoulder_width,
        calculate_distance(right_knee, right_ankle) / shoulder_width,
        calculate_distance(left_shoulder, left_hip) / shoulder_width,
        calculate_distance(right_shoulder, right_hip) / shoulder_width,
        calculate_distance(left_hip, right_hip) / shoulder_width,
    ]
    return features

def process_videos_in_directory(root_dir):
    """Processes all videos in a directory to extract gait features."""
    all_features = []
    
    for person_name in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing videos for: {person_name}")
        for video_name in os.listdir(person_dir):
            video_path = os.path.join(person_dir, video_name)
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    features = extract_gait_features(results.pose_landmarks)
                    if features:
                        all_features.append([person_name] + features)
            
            cap.release()
            
    # Create a DataFrame and save to CSV
    columns = ['person', 'lh_lk', 'rh_rk', 'lk_la', 'rk_ra', 'ls_lh', 'rs_rh', 'lh_rh']
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv('gait_features.csv', index=False)
    print("Feature extraction complete. Saved to gait_features.csv")

if __name__ == "__main__":
    DATA_DIR = 'data'
    process_videos_in_directory(DATA_DIR)