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

video_features = []
displaying_video = False
video_window = None


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

        for i in range(len(pose_landmark.landmark)):
            landmark = pose_landmark.landmark[i]
            frame_features += [landmark.x, landmark.y, landmark.z]
        landmarks.append(pose_landmark.landmark)

    # Either 66 real and 33 0's if 2 targets found, 66 0's and 33 real if 1 found, 99 0's if 0 found
    if len(frame_features) == 33 * 3 * 2:
        frame_features += [0] * 33 * 3
    elif len(frame_features) == 33 * 3:
        frame_features = [0] * 33 * 3 * 2 + frame_features
    elif len(frame_features) == 0:
        frame_features = [0] * 33 * 3 * 3
    else:
        raise ValueError(f"Number of frame features is not 66 or 33 or 0 but rather {len(frame_features)}")

    video_features.append(frame_features)

    if displaying_video:
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


def generate_features(videos, display_video=False, print_to_file=False):
    global video_features
    global displaying_video
    global video_window

    # Video list: clip list: 99 landmarks
    # (either 66 real and 33 0's if 2 targets found, 66 0's and 33 real if 1 found, 99 0's if 0 found)
    features = []

    displaying_video = display_video

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for video in videos:
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

    if print_to_file:
        file = open('features.txt', 'w')
        print(features, file=file)
        file.close()

    return features
