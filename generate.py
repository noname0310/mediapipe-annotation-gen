"""
Generate annotated video from youtube video id csv file
"""

import os
import subprocess
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate annotated video')
    parser.add_argument('-i', '--input', type=str, help='The input csv file')
    parser.add_argument('-s', '--start', type=int, choices=[0, 1, 2, 3], default=0, help='The start stage (0: download, 1: slice, 2: crop, 3: annotate)')
    args = parser.parse_args()


    def download():
        if args.input is None:
            raise ValueError('The input csv file is required')
        command = f'python download.py -i {args.input}'
        subprocess.run(command, shell=True, check=True)


    def slice():
        videos_dir = 'videos'
        videos = os.listdir(videos_dir)
        for video in videos:
            command = f'python slicer.py -i {os.path.join(videos_dir, video)}'
            subprocess.run(command, shell=True, check=True)


    def crop():
        videos_dir = 'sliced'
        videos = os.listdir(videos_dir)
        for video in videos:
            filename, _ = os.path.splitext(video)
            command = f'python cropper.py -i {os.path.join(videos_dir, video)} -o "cropped/{filename}.mp4"'
            subprocess.run(command, shell=True, check=True)


    def annotate():
        videos_dir = 'cropped'
        videos = os.listdir(videos_dir)
        for video in videos:
            filename, _ = os.path.splitext(video)
            command = f'python video_landmark.py -i {os.path.join(videos_dir, video)} -o "annotated/{filename}-v1.mp4" -v 1'
            subprocess.run(command, shell=True, check=True)
            command = f'python video_landmark.py -i {os.path.join(videos_dir, video)} -o "annotated/{filename}-v2.mp4" -v 2'
            subprocess.run(command, shell=True, check=True)
            command = f'python video_landmark.py -i {os.path.join(videos_dir, video)} -o "annotated/{filename}-v3.mp4" -v 3'
            subprocess.run(command, shell=True, check=True)
            command = f'python video_landmark.py -i {os.path.join(videos_dir, video)} -o "annotated/{filename}-v4.mp4" -v 4'
            subprocess.run(command, shell=True, check=True)


    tasks = [download, slice, crop, annotate]
    for i in range(args.start, len(tasks)):
        tasks[i]()
