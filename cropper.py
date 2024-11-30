"""
Crop the video based on the face bounding box
"""
import os
import subprocess
import argparse
from typing import Optional
import cv2
import mediapipe


def crop_video(video_path: str, face_bbox: mediapipe.tasks.components.containers.BoundingBox, output_file: str, resize: bool = False) -> None:
    """
    Crop the video based on the face bounding box

    Args:
        video_path: str: the input video file
        face_bbox: mediapipe.tasks.components.containers.BoundingBox: the face bounding box
        output_file: str: the output video file
        resize: bool: whether to resize the cropped video to 256x256
    """
    video_filename = os.path.basename(video_path)
    video_filename, _ = os.path.splitext(video_filename)

    output_dir = os.path.dirname(output_file)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vf = f'crop={face_bbox.width}:{face_bbox.height}:{face_bbox.origin_x}:{face_bbox.origin_y}'
    if resize:
        vf += f',scale=256:256'
    command = f'ffmpeg -y -i "{video_path}" -vf "{vf}" "{output_file}"'
    subprocess.run(command, shell=True, check=True)


def get_face_bounding_box(video_path: str) -> Optional[mediapipe.tasks.components.containers.BoundingBox]:
    """
    Get the face bounding box from the video

    Args:
        video_path: str: the input video file

    Returns:
        mediapipe.tasks.components.containers.BoundingBox: the face bounding box
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f'Could not open the video file {video_path}')
    
    base_options = mediapipe.tasks.BaseOptions(model_asset_path='./blaze_face_short_range.tflite')
    options = mediapipe.tasks.vision.FaceDetectorOptions(base_options=base_options)
    detector = mediapipe.tasks.vision.FaceDetector.create_from_options(options)

    min_top_left: tuple[int, int] = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_bottom_right: tuple[int, int] = (0, 0)

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)
        results = detector.detect(image)
        if len(results.detections) == 1:
            detection = results.detections[0]
            bbox = detection.bounding_box
            top_left = (bbox.origin_x, bbox.origin_y)
            bottom_right = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

            min_top_left = (min(min_top_left[0], top_left[0]), min(min_top_left[1], top_left[1]))
            max_bottom_right = (max(max_bottom_right[0], bottom_right[0]), max(max_bottom_right[1], bottom_right[1]))

    video_capture.release()

    if max_bottom_right == (0, 0):
        return None

    face_bbox = mediapipe.tasks.components.containers.BoundingBox(
        origin_x=min_top_left[0],
        origin_y=min_top_left[1],
        width=max_bottom_right[0] - min_top_left[0],
        height=max_bottom_right[1] - min_top_left[1]
    )
    return face_bbox


def loosen_bounding_box(bbox: mediapipe.tasks.components.containers.BoundingBox) -> mediapipe.tasks.components.containers.BoundingBox:
    """
    Loosen the bounding box

    Args:
        bbox: mediapipe.tasks.components.containers.BoundingBox: the bounding box

    Returns:
        mediapipe.tasks.components.containers.BoundingBox: the loosened bounding box
    """
    center_x = bbox.origin_x + bbox.width / 2
    center_y = bbox.origin_y + bbox.height / 2
    width = max(bbox.width, bbox.height)
    height = width
    origin_x = center_x - width / 2
    origin_y = center_y - height / 2

    return mediapipe.tasks.components.containers.BoundingBox(
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop the video based on the face bounding box')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output video file')
    args = parser.parse_args()

    face_bbox = get_face_bounding_box(args.input)
    crop_video(
        args.input,
        loosen_bounding_box(face_bbox),
        args.output,
        resize=True
    )
