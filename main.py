import cv2
import mediapipe as mp
import joblib
import numpy as np
import argparse

# --- Function Definitions from process_data.py ---
# For consistency, we reuse the same feature extraction logic.
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def extract_gait_features(landmarks):
    mp_pose = mp.solutions.pose
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    if shoulder_width < 1e-6:
        return None

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
# --- End of Function Definitions ---


def run_live_recognition(video_path=None):
    # Load the trained model and label encoder
    try:
        model = joblib.load('gait_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run train.py first.")
        return

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Setup video capture
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0) # Use webcam

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract features and predict
            features = extract_gait_features(results.pose_landmarks)
            if features:
                features_np = np.array(features).reshape(1, -1)
                prediction = model.predict(features_np)
                predicted_person = label_encoder.inverse_transform(prediction)[0]
                
                # Display the prediction
                cv2.putText(image_bgr, f"Person: {predicted_person}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Gait Recognition', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run live gait recognition.")
    parser.add_argument('--video_path', type=str, help="Path to a video file to process. If not provided, webcam will be used.")
    args = parser.parse_args()
    
    run_live_recognition(args.video_path)