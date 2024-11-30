"""
Generate landmarked video from a video file
"""

import os
import argparse
import enum
from typing import Any, Callable
import cv2
import mediapipe
from mediapipe.framework.formats import landmark_pb2
import numpy


class DrawMode(enum.Enum):
    DEFAULT = 1
    NO_TESSELATION = 2
    NO_FACE_OVAL = 3
    ONLY_LANDMARKS = 4


def map_video_frames_to_file(video_path: str, f: Callable[[numpy.ndarray], numpy.ndarray], output_file: str) -> None:
    """
    Create a video capture object from the video file

    Args:
        video_path: str: the input video file
        f: Callable[[numpy.ndarray], [numpy.ndarray]]: the function to process each frame
        output_file: str: the output video file
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f'Could not open the video file {video_path}')
    
    output_dir = os.path.dirname(output_file)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(output_file, codec, fps, (frame_width, frame_height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = f(frame)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(processed_frame)

    video_capture.release()
    video_writer.release()


def create_detector() -> Any:
    """
    Create a face landmark detector

    Returns:
        mediapipe.tasks.vision.FaceLandmarker: the face landmark detector
    """
    base_options = mediapipe.tasks.BaseOptions(model_asset_path='face_landmarker.task')
    options = mediapipe.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
    return detector

default_connections = mediapipe.solutions.face_mesh.FACEMESH_CONTOURS
default_connection_drawing_spec = mediapipe.solutions.drawing_styles.get_default_face_mesh_contours_style()

nofaceoval_contours_connections = {connection for connection in default_connections}
nofaceoval_contours_connection_drawing_spec = mediapipe.solutions.drawing_styles.get_default_face_mesh_contours_style()
for connection in mediapipe.solutions.face_mesh.FACEMESH_FACE_OVAL:
    nofaceoval_contours_connections.remove(connection)
    nofaceoval_contours_connection_drawing_spec.pop(connection)
nofaceoval_contours_connections = frozenset(nofaceoval_contours_connections)


def draw_landmarks_on_image_internal(rgb_image: numpy.ndarray, detection_result: Any, draw_mode: DrawMode) -> numpy.ndarray:
    """
    Draw landmarks on the image

    Args:
        rgb_image: numpy.ndarray: the input image
        detection_result: Any: the detection result
        draw_mode: DrawMode: the draw mode

    Returns:
        numpy.ndarray: the annotated image
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = numpy.copy(rgb_image) if draw_mode != DrawMode.ONLY_LANDMARKS else numpy.zeros_like(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        if draw_mode == DrawMode.DEFAULT:
            mediapipe.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mediapipe.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mediapipe.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
        if draw_mode in (DrawMode.DEFAULT, DrawMode.NO_TESSELATION, DrawMode.NO_FACE_OVAL, DrawMode.ONLY_LANDMARKS):
            connections = None
            connection_drawing_spec = None
            if draw_mode == DrawMode.NO_FACE_OVAL or draw_mode == DrawMode.ONLY_LANDMARKS:
                connections = nofaceoval_contours_connections
                connection_drawing_spec = nofaceoval_contours_connection_drawing_spec
            else:
                connections = default_connections
                connection_drawing_spec = default_connection_drawing_spec
                
            mediapipe.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=connection_drawing_spec
            )
        if draw_mode in (DrawMode.DEFAULT, DrawMode.NO_TESSELATION, DrawMode.NO_FACE_OVAL, DrawMode.ONLY_LANDMARKS):
            mediapipe.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mediapipe.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mediapipe.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    return annotated_image


def draw_landmarks_on_image(detector: Any, frame: numpy.ndarray, draw_mode: DrawMode) -> numpy.ndarray:
    """
    Draw landmarks on the image

    Args:
        detector: mediapipe.tasks.vision.FaceLandmarker: the face landmark detector
        frame: numpy.ndarray: the input image
        draw_mode: DrawMode: the draw mode

    Returns:
        numpy.ndarray: the annotated image
    """
    image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image_internal(image.numpy_view(), detection_result, draw_mode)
    return annotated_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate landmarked video from a video file')
    parser.add_argument('-i', '--input', type=str, required=True, help='The input video file')
    parser.add_argument('-o', '--output', type=str, required=True, help='The output video file')
    parser.add_argument('-v', '--draw_mode', type=int, choices=[1, 2, 3, 4], default=1, help='The draw mode (1: default, 2: no tesselation, 3: no face oval, 4: only landmarks)')
    args = parser.parse_args()
    
    detector = create_detector()

    def process_frame(frame: numpy.ndarray) -> numpy.ndarray:
        return draw_landmarks_on_image(detector, frame, DrawMode(args.draw_mode))
                             
    map_video_frames_to_file(args.input, process_frame, args.output)
    print(f'Landmarked video is saved to {args.output}')
