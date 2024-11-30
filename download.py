"""
Download youtube videos from the given csv file
"""
import os
import subprocess
import csv
import argparse


def download_video(video_id: str, output_dir: str) -> None:
    """
    Download the video from the given youtube video id

    Args:
        video_id: str: the youtube video id
        output_dir: str: the output directory
    """
    output_file = os.path.join(output_dir, f'{video_id}')
    command = f'yt-dlp -o "{output_file}" https://www.youtube.com/watch?v={video_id}'
    subprocess.run(command, shell=True, check=True)


def read_csv_file(input_file: str) -> list[str]:
    """
    Read the input csv file and return the list of video_ids

    Args:
        input_file: str: the input csv file

    Returns:
        list[str]: the list of urls
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        video_ids = [row['video_id'] for row in reader]
    return video_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download youtube videos')
    parser.add_argument('-i', '--input', type=str, required=True, help='The input csv file')
    parser.add_argument('-o', '--output', type=str, default='videos', help='The output directory')
    args = parser.parse_args()

    urls = read_csv_file(args.input)
    for url in urls:
        print(f'Downloading video from {url}')
        download_video(url, os.path.join(os.getcwd(), args.output))
