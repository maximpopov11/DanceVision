import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
MODEL_PATH = "pose_landmarker_heavy.task"

# Hyperparameters
NUM_POSES = 2
MIN_POSE_DETECTION_CONFIDENCE = 0.8
MIN_POSE_PRESENCE_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.9

NUM_LABELS = 38
VIDEO_PATHS = ["v0_" + str(i) + ".mp4" for i in range(NUM_LABELS)]

DISPLAY_VIDEO = True


# Display video with landmarks drawn
def display_annotated_video(rgb_image, pose_landmarks):
    annotated_image = np.copy(rgb_image)

    for landmarks in pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image


# Parse frame landmarks and show on video if applicable
def parse_frame_landmarks(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image, _: int):
    frame_features = []

    pose_landmarks = []
    for landmarks in detection_result.pose_landmarks:
        pose_landmark = landmark_pb2.NormalizedLandmarkList()
        pose_landmark.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z
            ) for landmark in landmarks
        ])
        pose_landmarks.append(pose_landmark)

        frame_features.append(pose_landmark.landmark)

    video_features.append(frame_features)

    if DISPLAY_VIDEO:
        global video_window
        video_window = cv2.cvtColor(
            display_annotated_video(output_image.numpy_view(), pose_landmarks),
            cv2.COLOR_RGB2BGR)


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=NUM_POSES,
    min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
    min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    output_segmentation_masks=False,
    result_callback=parse_frame_landmarks
)

# List for each movement video clip containing list for each detected target containing list of body keypoint landmarks
features = []
video_features = []

video_window = None
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    for video in VIDEO_PATHS:
        video_features = []

        video_capture = cv2.VideoCapture(video)
        while video_capture.isOpened():
            video_ongoing, frame = video_capture.read()
            if not video_ongoing:
                break

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            if video_window is not None:
                cv2.imshow("MediaPipe Pose Landmark", video_window)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        features.append(video_features)

    video_capture.release()
    cv2.destroyAllWindows()

file = open('features.txt', 'w')
print(features, file=file)
file.close()
