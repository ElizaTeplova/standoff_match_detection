from datetime import datetime

import cv2
import torch
import time
import easyocr
import subprocess

from pathlib import Path
from gui.recognition.video_preprocessor_exception import VideoPreprocessorException
# from video_preprocessor_exception import VideoPreprocessorException
from gui.contract.event_listener import EventListener
from gui.contract.event_manager import EventManager
from gui.recognition.recognizer import Recognizer


class VideoPreprocessor:
    cropped_time_prefix: str = 'cropped_time_video_'
    cropped_score_prefix: str = 'cropped_score_video_'
    accelerated_time_prefix: str = 'accelerated_time_video_'
    accelerated_score_prefix: str = 'accelerated_video_video_'
    time_filename_prefix: str = 'time_filename_'
    score_filename_prefix: str = 'score_filename_'
    csv_dir: Path = Path.cwd() / 'report'
    video_dir: Path = Path.cwd() / 'video'
    video_dir.mkdir(parents=True, exist_ok=True)

    def __init__(self, video_full_path: str, event_manager: EventManager, recognizer: Recognizer = Recognizer()):
        self.__recognizer = recognizer
        self.__event_manager = event_manager
        self.__video_postfix = str(datetime.now().timestamp()) + '.mp4'
        self.__csv_postfix: str = str(datetime.now().timestamp()) + '.csv'
        self.__video_full_path: str = video_full_path

        self.__cropped_time_video_full_path: str = str(
            self.video_dir / (self.cropped_time_prefix + self.__video_postfix))
        self.__cropped_score_video_full_path: str = str(
            self.video_dir / (self.cropped_score_prefix + self.__video_postfix))

        self.__accelerated_time_video_full_path: str = str(
            self.video_dir / (self.accelerated_time_prefix + self.__video_postfix))
        self.__accelerated_time_video_labels_full_path: str = str(
            self.video_dir / (self.accelerated_time_prefix + self.__video_postfix))

        self.__accelerated_score_video_full_path: str = str(
            self.video_dir / (self.accelerated_score_prefix + self.__video_postfix))
        self.__accelerated_score_video_labels_full_path: str = str(
            self.video_dir / (self.accelerated_score_prefix + self.__video_postfix))

        self.detected_score_csv = str(self.csv_dir / (self.time_filename_prefix + self.__csv_postfix))
        self.detected_time_csv = str(self.csv_dir / (self.time_filename_prefix + self.__csv_postfix))

    def preprocess(self) -> (str, str):
        # region Crop Video
        cropped_time_video_full_path = self.__crop_time_video()
        cropped_score_video_full_path = self.__crop_score_video()
        # endregion
        # region Accelerate Video
        accelerated_time_video_full_path = self.__accelerate_time_video(
            cropped_time_video_full_path,
            self.__accelerated_time_video_full_path
        )
        accelerated_score_video_full_path = self.__accelerate_score_video(
            cropped_score_video_full_path,
            self.__accelerated_score_video_full_path
        )
        output_time_csv = self.__recognizer.recognize(
            accelerated_time_video_full_path,
            self.__accelerated_time_video_labels_full_path,
            self.detected_time_csv
        )
        output_score_csv = self.__recognizer.recognize(
            accelerated_score_video_full_path,
            self.__accelerated_score_video_labels_full_path,
            self.detected_score_csv
        )
        # endregion
        return output_time_csv, output_score_csv

    def __crop_time_video(self) -> str:
        msg: str | None = 'Cropping time video...'
        self.__event_manager.notify(msg)
        msg = None
        try:
            return self.__crop_video(
                input_path=self.__video_full_path,
                # Directory + filename concatenation
                output_path=self.__cropped_time_video_full_path,
                x=546,
                y=0,
                width=187,
                height=40,
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred while cropping the time video: {e}"
            raise VideoPreprocessorException(msg)
        except FileNotFoundError:
            msg = 'FFmpeg is not installed or not found in PATH.'
            raise VideoPreprocessorException(msg)
        finally:  # Notify that video was processed
            msg = 'Video time cropped' if msg is None else msg
            self.__event_manager.notify(msg)

    def __crop_score_video(self) -> str:
        msg: str | None = 'Cropping score video...'
        self.__event_manager.notify(msg)
        msg = None
        try:
            return self.__crop_video(
                input_path=self.__video_full_path,
                # Directory + filename concatenation
                output_path=self.__cropped_score_video_full_path,
                x=966,
                y=0,
                width=314,
                height=206,
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred while cropping the score video: {e}"
            raise VideoPreprocessorException(msg)
        except FileNotFoundError:
            msg = 'FFmpeg is not installed or not found in PATH.'
            raise VideoPreprocessorException(msg)
        finally:  # Notify that video was processed
            msg = 'Video score cropped' if msg is None else msg
            self.__event_manager.notify(msg)

    def __accelerate_time_video(self, input_path: str, output_path: str) -> str:
        speed_factor = 30
        msg: str | None = 'Accelerating time video...'
        try:
            return self.__accelerate_video(
                input_path=input_path,
                output_path=output_path,
                speed_factor=speed_factor
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred while accelerating the video: {e}"
        except FileNotFoundError:
            msg = "FFmpeg is not installed or not found in PATH."
        finally:
            msg = f"Time Video has been sped up by {speed_factor}x and saved to {output_path}" if msg is None else msg
            self.__event_manager.notify(msg)

    def __accelerate_score_video(self, input_path: str, output_path: str) -> str:
        speed_factor = 30
        msg: str | None = 'Accelerating score video...'
        try:
            return self.__accelerate_video(
                input_path=input_path,
                output_path=output_path,
                speed_factor=speed_factor
            )
        except subprocess.CalledProcessError as e:
            msg = f"Error occurred while accelerating score video: {e}"
        except FileNotFoundError:
            msg = "FFmpeg is not installed or not found in PATH."
        finally:
            msg = f"Score Video has been sped up by {speed_factor}x and saved to {output_path}" if msg is None else msg
            self.__event_manager.notify(msg)

    @staticmethod
    def __accelerate_video(input_path: str, output_path: str, speed_factor: int = 30) -> str:
        """
        Speeds up a video by a given factor using FFmpeg.
        :param speed_factor: Factor by which to speed up the video
        """
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
        return output_path

    @staticmethod
    def __crop_video(input_path: str, output_path: str, width: int, height: int, x: int, y: int) -> str:
        """
        Crop specific region from video
        :param output_path:
        :param input_path: full path to intput video
        :param width: output width
        :param height: output height
        :param x: X-coordinate of the top-left corner of the cropped region.
        :param y: Y-coordinate of the top-left corner of the cropped region.
        :return str: output video full path
        Raises:
            subprocess.CalledProcessError: If FFmpeg fails to process the command.
            FileNotFoundError: If FFmpeg is not installed or not found in the system's PATH.
        """

        command = [
            "ffmpeg",
            "-y",  # Overwrite output file without confirmation
            "-i", input_path,  # Input video file
            "-vf", f"crop={width}:{height}:{x}:{y}",  # Crop filter
            output_path  # Output video file
        ]

        subprocess.run(command, check=True)
        return output_path
