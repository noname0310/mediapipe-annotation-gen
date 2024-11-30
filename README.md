# mediapipe-annotation-gen

A script to generate mediapipe face landmark annotations for youtube videos.

## External Dependencies

- yt-dlp (windows binary included)
- ffmpeg (windows binary included)

## Setup

Install the requirements

```bash
conda env create -f conda_requirements.txt
```

Activate the environment

```bash
conda activate mediapipe-env
```

## Scripts

- `download.py`: Downloads youtube videos from a list of video ids in a csv file.
    - `yt-dlp.exe` is used to download the videos.
    - `metadata.csv` is example csv file with video ids.

- `slicer.py`: Slices the downloaded videos into clips where the face is detected more than 3 seconds.
    - `blaze_face_short_range.tflite` is used to detect faces in the videos.
    - `ffmpeg.exe` is used to slice the videos.

- `cropper.py`: Crops the face from the clips.
    - `blaze_face_short_range.tflite` is used to detect faces bounding boxes in the clips.
    - `ffmpeg.exe` is used to crop the faces.

- `video_landmark.py`: Generates mediapipe face landmark annotations for the cropped faces.
    - `face_landmarker.task` is used to generate the annotations.

- **`generate.py`: script to run all the above scripts in sequence.**

## Usage

run the following command

```bash 
python generate.py -i metadata.csv
```

then video is downloaded, sliced, cropped and annotated in sequence.
