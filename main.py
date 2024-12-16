# This is a sample Python script.
import cv2
import easyocr
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time
import subprocess


def preprocess_video(input_video_path: str, output_video_path: str, crop_region: tuple, desired_fps: int):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError('Cannot open video file')
    frame_with = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip_rate = int(fps / desired_fps)

    process_frames = []
    ten_percent_frames = int(total_frames / 10)
    print(f'Ten percent frames: {ten_percent_frames}')
    for i in range(1, 10):
        process_frames.append((i * 10, int(ten_percent_frames * i)))

    print(f'Original video {input_video_path}: {frame_with}x{frame_height}, {fps} FPS, {total_frames} total frames')

    # x, y, w, h = (970, 0, 300, 200)
    x, y, w, h = crop_region
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # print(f'Processing frame {frame_count}')
        if frame_count % ten_percent_frames == 0:
            print(f'Processing frame {frame_count}')

        if frame_count % frame_skip_rate != 0:
            continue

        cropped_frame = frame[y:y + h, x:x + w]
        # Apply the contrast and brightness adjustments
        cropped_frame = cv2.convertScaleAbs(cropped_frame, alpha=1.5, beta=10)
        out.write(cropped_frame)
    cap.release()
    out.release()
    print(f'Processed Video saved to {output_video_path}')


def detect_kills(video_path: str, reader: easyocr.easyocr.Reader, output_csv_path: str, out_video_path: str):
    cap = cv2.VideoCapture(video_path)

    frame_with = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_with, frame_height))

    if not cap.isOpened():
        raise IOError('Cannot open video file')

    all_dfs = []
    current_frame = 0
    while True:
        current_frame += 1
        ret, frame = cap.read()
        if not ret:
            break

        result = reader.readtext(frame)
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Draw the bounding box
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Put the detected text on the frame
            text_position = (top_left[0], top_left[1] - 10)  # Adjust position slightly above the box
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        time_s = current_frame / fps
        img_df = pd.DataFrame(result, columns=['bbox', 'text', 'conf'])
        img_df['time_s'] = time_s
        img_df['current_frame'] = current_frame
        img_df.reset_index(drop=True)
        all_dfs.append(img_df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(output_csv_path)
    cap.release()
    out.release()

def convertWithFfMpeg(input_path: str, output_path: str, speed_factor: int = 30):
    """
        Speeds up a video by a given factor using FFmpeg.

        :param input_path: Path to the input video file
        :param output_path: Path to the output video file
        :param speed_factor: Factor by which to speed up the video
        """
    try:
        # Speed up video and audio
        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,  # Input file
            "-vf", f"setpts=PTS/{speed_factor}",  # Video filter to speed up
            "-af", f"atempo={min(speed_factor, 2)}",  # Audio filter, capped at 2x per step
            output_path  # Output file
        ]

        # Adjust audio speed if greater than 2x
        if speed_factor > 2:
            # Chain multiple atempo filters to achieve >2x audio speed
            atempo_filters = []
            temp_factor = speed_factor
            while temp_factor > 2:
                atempo_filters.append("atempo=2")
                temp_factor /= 2
            atempo_filters.append(f"atempo={temp_factor}")
            command[command.index(f"atempo={min(speed_factor, 2)}")] = ",".join(atempo_filters)

        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Video has been sped up by {speed_factor}x and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing the video: {e}")
    except FileNotFoundError:
        print("FFmpeg is not installed or not found in PATH.")


def crop_video(input_path: str, output_path: str, width: int, height: int, x: int, y: int):
    """
    Crops a specific region from a video using FFmpeg.

    :param input_path: Path to the input video file
    :param output_path: Path to the output video file
    :param width: Width of the cropped region
    :param height: Height of the cropped region
    :param x: X-coordinate of the top-left corner of the cropped region
    :param y: Y-coordinate of the top-left corner of the cropped region
    """
    try:
        # FFmpeg command for cropping
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file without confirmation
            "-i", input_path,  # Input video file
            "-vf", f"crop={width}:{height}:{x}:{y}",  # Crop filter
            output_path  # Output video file
        ]

        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Video cropped successfully and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while cropping the video: {e}")
    except FileNotFoundError:
        print("FFmpeg is not installed or not found in PATH.")

def main():
    input_video_path = './video/1match.mp4'
    output_video_path = './video/1match_cropped.mp4'
    output_video_score_path = './video/ffmpeg_output_cropped_score.mp4'
    # input_video_path = './video/test_crop.mp4'
    # output_video_path = './video/output_cropped.mp4'
    output_time_video_path = './video/output_time_cropped.mp4'
    output_video_path_with_kill_labels = './video/cropped_with_labels.mp4'
    output_video_path_with_time_labels = './video/time_labels.mp4'
    output_kills_csv_path = './all_kills.csv'
    output_time_csv_path = './all_time.csv'
    print(torch.cuda.is_available())  # Should print True
    print(torch.cuda.get_device_name(0))
    # return
    # crop_region = (1500, 0)
    start_time = time.time()
    # detect_kills(output_video_score_path, easyocr.Reader(['en'], gpu=True), output_kills_csv_path, output_video_path_with_kill_labels)
    detect_kills('./video/ffmpeg_output_cropped_time.mp4', easyocr.Reader(['en'], gpu=True), output_time_csv_path, output_video_path_with_time_labels)

    # Crop video to make score
    # crop_video(
    #     './video/ffmpeg_output_time_cropped.mp4',
    #     './video/ffmpeg_output_cropped_score.mp4',
    #     x=966,
    #     y=0,
    #     width=314,
    #     height=206
    # )

    # Crop video to make time
    # crop_video(
    #     './video/ffmpeg_output_time_cropped.mp4',
    #     './video/ffmpeg_output_cropped_time.mp4',
    #     x=546,
    #     y=0,
    #     width=187,
    #     height=40
    # )

    # convert whole video to short version
    # convertWithFfMpeg(input_video_path, './video/ffmpeg_output_time_cropped.mp4')

    # preprocess_video(input_video_path, output_video_path, crop_region=(966, 0, 314, 206), desired_fps=1)
    # preprocess_video(input_video_path, output_time_video_path, crop_region=(546, 0, 187, 40), desired_fps=1)
    # preprocess_video(output_time_video_path, output_time_video_path, crop_region=(546, 0, 187, 40), desired_fps=1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    # preprocess_video(input_video_path, output_video_path, crop_region)

    # imgpath = './images/img.png'
    # reader = easyocr.Reader(['en'], gpu=True)
    # result = reader.readtext(imgpath)
    # img_df = pd.DataFrame(result, columns=['bbox', 'text', 'conf'])
    # img_df.to_csv('result.csv', index=False)
    # print(result)

    # plt.imshow(plt.imread(imgpath))
    # plt.show()


if __name__ == '__main__':
    main()
