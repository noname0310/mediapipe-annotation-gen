"""
Slice the video segments where the face is detected continuously for more than 3 seconds
"""
import os
import subprocess
import argparse
from typing import Optional
import cv2
import mediapipe


def slice_video_segments(video_path: str, range: tuple[float, float], output_dir: str) -> None:
    """
    Slice the video segments using the given range and save the output video

    Args:
        video_path: str: the input video file
        range: tuple[float, float]: the start and end range of the video
        output_dir: str: the output directory
    """
    start, end = range
    start = round(start, 2)
    end = round(end, 2)

    video_filename = os.path.basename(video_path)
    video_filename, _ = os.path.splitext(video_filename)
    output_file = os.path.join(output_dir, f'{video_filename}_{start}_{end}.mp4')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = f'ffmpeg -y -i "{video_path}" -ss {start} -to {end} -c:v libx264 -crf 18 -preset slow -c:a copy "{output_file}"'
    subprocess.run(command, shell=True, check=True)


def is_bounding_box_similar(
    box1: mediapipe.tasks.components.containers.BoundingBox,
    box2: mediapipe.tasks.components.containers.BoundingBox,
    video_width: int,
    video_height: int
) -> bool:
    """
    Check if the bounding boxes are similar

    Args:
        box1: mediapipe.tasks.components.containers.BoundingBox: the first bounding box
        box2: mediapipe.tasks.components.containers.BoundingBox: the second bounding box
        video_width: int: the width of the video
        video_height: int: the height of the video

    Returns:
        bool: True if the bounding boxes are similar, False otherwise
    """
    box1_origin_x = box1.origin_x / video_width
    box1_origin_y = box1.origin_y / video_height
    box1_width = box1.width / video_width
    box1_height = box1.height / video_height

    box2_origin_x = box2.origin_x / video_width
    box2_origin_y = box2.origin_y / video_height
    box2_width = box2.width / video_width
    box2_height = box2.height / video_height

    threshold = 0.1

    if abs(box1_origin_x - box2_origin_x) <= threshold and \
        abs(box1_origin_y - box2_origin_y) <= threshold and \
        abs(box1_width - box2_width) <= threshold and \
        abs(box1_height - box2_height) <= threshold:
        return True
    
    return False


def collect_ranges(video_path: str) -> list[tuple[float, float]]:
    """
    Collect the video segments where the face is detected continuously for more than 3 seconds

    Args:
        video_path: str: the input video file

    Returns:
        list[tuple[float, float]]: the list of ranges
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f'Could not open the video file {video_path}')

    base_options = mediapipe.tasks.BaseOptions(model_asset_path='./blaze_face_short_range.tflite')
    options = mediapipe.tasks.vision.FaceDetectorOptions(base_options=base_options)
    detector = mediapipe.tasks.vision.FaceDetector.create_from_options(options)

    ranges: list[tuple[float, float]] = []
    start: Optional[float] = None
    end: Optional[float] = None
    last_bounding_box = None
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)
        results = detector.detect(image)
        if results.detections and len(results.detections) == 1:
            if start is None:
                start = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                end = start
                last_bounding_box = results.detections[0].bounding_box
            else:
                if not is_bounding_box_similar(last_bounding_box, results.detections[0].bounding_box, frame.shape[1], frame.shape[0]):
                    ranges.append((start, end))
                    start = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    end = start
                    last_bounding_box = results.detections[0].bounding_box
                else:
                    end = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    last_bounding_box = results.detections[0].bounding_box
        else:
            if start is not None:
                ranges.append((start, end))
                start = None
                end = None
                last_bounding_box = None

    if start is not None:
        ranges.append((start, end))

    video_capture.release()

    ranges = [range for range in ranges if range[1] - range[0] >= 3]

    return ranges


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Slice the video segments where the face is detected')
    parser.add_argument('-i', '--input', type=str, required=True, help='The input video file')
    parser.add_argument('-o', '--output', type=str, default='sliced', help='The output directory')
    args = parser.parse_args()

    ranges = collect_ranges(args.input)
    print(f'Found {len(ranges)} video segments')
    for i, range in enumerate(ranges):
        print(f'Slicing video segment {i} from {range[0]} to {range[1]}')
        slice_video_segments(args.input, range, os.path.join(os.getcwd(), args.output))
