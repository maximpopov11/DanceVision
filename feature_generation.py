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

DISPLAY_VIDEO = True

# TODO: go through all videos once everything else is good
VIDEO_PATH = "v0_0.mp4"


# Display video with landmarks drawn
def display_annotated_video(rgb_image, detection_result):
    # TODO: DRY parse_frame_landmarks
    annotated_image = np.copy(rgb_image)

    # List of 0 to num_poses lists with landmarks for each detected target
    pose_landmark_list = detection_result.pose_landmarks

    for landmarks in pose_landmark_list:
        pose_landmark = landmark_pb2.NormalizedLandmarkList()

        pose_landmark.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z
            ) for landmark in landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmark,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image


# Parse frame landmarks and show on video if applicable
def parse_frame_landmarks(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image, _: int):
    frame_features = []

    for landmarks in detection_result.pose_landmarks:
        pose_landmark = landmark_pb2.NormalizedLandmarkList()
        pose_landmark.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z
            ) for landmark in landmarks
        ])

        frame_features.append(pose_landmark.landmark)

    features.append(frame_features)

    if DISPLAY_VIDEO:
        global video_window
        video_window = cv2.cvtColor(
            display_annotated_video(output_image.numpy_view(), detection_result),
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

video_window = None
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    video = cv2.VideoCapture(VIDEO_PATH)

    while video.isOpened():
        video_ongoing, frame = video.read()
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

    video.release()
    cv2.destroyAllWindows()

file = open('features.txt', 'w')
print(features, file=file)
file.close()
