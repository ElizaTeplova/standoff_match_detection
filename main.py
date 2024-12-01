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
        img_df.reset_index(drop=True)
        all_dfs.append(img_df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(output_csv_path)
    cap.release()
    out.release()

def main():
    input_video_path = './video/test_crop.mp4'
    output_video_path = './video/output_cropped.mp4'
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
    detect_kills(output_video_path, easyocr.Reader(['en'], gpu=True), output_kills_csv_path, output_video_path_with_kill_labels)
    detect_kills(output_time_video_path, easyocr.Reader(['en'], gpu=True), output_time_csv_path, output_video_path_with_time_labels)
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
