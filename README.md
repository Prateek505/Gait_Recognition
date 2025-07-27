# Gait Recognition for Person Identification

This project identifies individuals based on their unique walking patterns (gait) using computer vision and machine learning. It uses MediaPipe for pose estimation and an SVM classifier to recognize users, simulating a 'self-following cart' scenario.

## Tech Stack

Python, OpenCV, MediaPipe, Scikit-learn, NumPy

## How to Run

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone [https://github.com/Prateek505/Gait_Recognition.git](https://github.com/Prateek505/Gait_Recognition.git)
    cd Gait_Recognition
    pip install -r requirements.txt
    ```

2.  **Add your training data:**
    Create folders inside the `data/` directory for each person and add short videos of them walking (e.g., `data/prateek/walk1.mp4`).

3.  **Run the scripts in order:**

    * **Process videos into features:**
        ```bash
        python process_data.py
        ```
    * **Train the model:**
        ```bash
        python train.py
        ```
    * **Run the live demo:**
        ```bash
        # For webcam
        python main.py

        # For a video file
        python main.py --video_path "path/to/video.mp4"
        ```
