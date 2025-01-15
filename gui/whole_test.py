from gui.recognition.video_preprocessor import VideoPreprocessor
from gui.recognition.data_reporter import DataReporter
from gui.contract.event_listener import EventListener
from gui.contract.event_manager import EventManager
import time


class ConsolListener(EventListener):
    def __init__(self):
        super().__init__()

    def update(self, message):
        print(message)


def main():
    cls = ConsolListener()
    em = EventManager()
    em.subscribe(cls)
    input_video_path = r'C:\studying\SW_7_semester\practice\textdetection\pythonProject1\video\1match.mp4'

    video_preprocessor = VideoPreprocessor(input_video_path, em)
    nicknames1 = ['N1taro', 'cosmos', 'darzyy', 'zlewis', 'v1ZUAL']
    nicknames2 = ['cLouding', 'ww fANT1M', 'nia', 'Jumper', 'packoo']

    start_time = time.time()
    accelerated_time_video_full_path, accelerated_score_video_full_path = video_preprocessor.preprocess()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    data_reporter = DataReporter(
        csv_time_path=accelerated_time_video_full_path,
        csv_score_path=accelerated_score_video_full_path,
        team1_nicknames=nicknames1,
        team2_nicknames=nicknames2,
        event_manager=em
    )
    start_time = time.time()
    full_match_score_output_path = data_reporter.report()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    print(full_match_score_output_path)


if __name__ == '__main__':
    main()
