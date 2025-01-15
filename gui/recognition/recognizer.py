import cv2
import easyocr
import pandas as pd


class Recognizer:
    def __init__(self, reader: easyocr.easyocr.Reader = easyocr.Reader(['en'], gpu=True)):
        self.reader = reader

    def recognize(self, video_path: str, out_video_path: str, output_csv_path: str,) -> str:
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

            result = self.reader.readtext(frame)
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
        return output_csv_path
