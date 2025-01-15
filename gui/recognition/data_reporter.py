from datetime import timedelta, datetime
from pathlib import Path

import cv2
import pandas as pd
import regex as re
import easyocr

from gui.contract.event_manager import EventManager
from typing import List
from Levenshtein import ratio
from sklearn.cluster import KMeans
from .data_reporter_exception import DataReporterException
from functools import cmp_to_key
import ast


class DataReporter:
    PLAYERS_AMOUNT: int = 5
    csv_dir: Path = Path.cwd() / 'report'
    csv_dir.mkdir(parents=True, exist_ok=True)

    def __init__(
            self,
            csv_time_path: str,
            csv_score_path: str,
            team1_nicknames: List[str],
            team2_nicknames: List[str],
            event_manager: EventManager
    ):
        self.__csv_postfix: str = str(datetime.now().timestamp()) + '.csv'
        self.__full_match_score_csv_output_path: str = str(self.csv_dir / ('match_score' + self.__csv_postfix))
        self.__full_time_csv_output_path: str = str(self.csv_dir / ('full_time_' + self.__csv_postfix))

        errors = []
        if len(team1_nicknames) != self.PLAYERS_AMOUNT:
            errors.append(f'team1_nicknames must have length equal to {self.PLAYERS_AMOUNT}')
        if len(team2_nicknames) != self.PLAYERS_AMOUNT:
            errors.append(f'team2_nicknames must have length equal to {self.PLAYERS_AMOUNT}')
        if len(errors) > 0:
            raise DataReporterException(', '.join(errors))

        self.__team1_nicknames: List[str] = team1_nicknames
        self.__team2_nicknames: List[str] = team2_nicknames

        self.__csv_all_time_input_path: str = csv_time_path
        self.__csv_score_input_path: str = csv_score_path
        self.__event_manager: EventManager = event_manager

    def report(self) -> str:
        """
        Report the data into csv file
        :return: csv_file_path
        """


        all_time_df: pd.DataFrame = pd.read_csv(self.__csv_all_time_input_path)
        all_time_path = self.__time_processing(all_time_df)
        formatted_time: pd.DataFrame = pd.read_csv(all_time_path)
        all_nicknames: list = self.__team1_nicknames
        all_nicknames.extend(self.__team2_nicknames)
        all_frags_df: pd.DataFrame = pd.read_csv(self.__csv_score_input_path)
        return self.__preprocess_frags(all_frags_df, all_nicknames, formatted_time)

    def __time_processing(self, all_time_df: pd.DataFrame) -> str:
        all_time_df = all_time_df.drop(['Unnamed: 0'], axis=1)
        # region bbox to list
        all_time_df['bbox'] = all_time_df['bbox'].apply(ast.literal_eval)
        # endregion
        # region calculate bbox's center_x and center_y
        centers_x = []
        centers_y = []
        for i, row in all_time_df.iterrows():
            (top_left, top_right, bottom_right, bottom_left) = row['bbox']
            centers_x.append((top_left[0] + top_right[0]) / 2)
            centers_y.append((top_left[1] + bottom_left[1]) / 2)
        all_time_df['center_x'] = centers_x
        all_time_df['center_y'] = centers_y
        # endregion
        # region find left/center/right zone by clustering center_x
        model = KMeans(n_clusters=3)
        time_center_x: pd.Series = all_time_df['center_x']
        model.fit(time_center_x.values.reshape(-1, 1))
        all_time_df['x_clusters'] = model.labels_

        sorted_clusters_by_center_x: pd.DataFrame = all_time_df[['x_clusters', 'center_x']].groupby(
            'x_clusters').mean().sort_values(by='center_x', ascending=True)
        left_x_cluster = sorted_clusters_by_center_x.index.tolist()[0]
        center_x_cluster = sorted_clusters_by_center_x.index.tolist()[1]
        right_x_cluster = sorted_clusters_by_center_x.index.tolist()[2]
        # endregion
        formatted_time = pd.DataFrame(columns=['left_score', 'center_score', 'right_score', 'time_s'])
        # region formate dataset with flattened l/c/r region
        # 0 - bbox , 1 - text, 2 - conf, 3 - time_s, 4 - center_x, 5 - center_y, 6 - x_clusters, 7 - y_clusters
        detection_column = {
            left_x_cluster: 'left_score',
            center_x_cluster: 'center_score',
            right_x_cluster: 'right_score'
        }
        for frame, group in all_time_df.groupby('current_frame').apply(lambda x: x.values.tolist()).items():
            add_row = pd.DataFrame(columns=['frame', 'time_s', 'left_score', 'center_score', 'right_score'])
            add_row['frame'] = [int(frame)]
            for note in group:
                add_row['time_s'] = note[3]
                add_row[detection_column.get(note[7])] = [note[1]]
            formatted_time = pd.concat([formatted_time, add_row], ignore_index=True)
        formatted_time['frame'] = formatted_time['frame'].apply(int)
        # endregion
        # region fill time (center) column
        # max time starts from 02:03
        # we subtract 1s from current_time if time is not detected
        # otherwise initialize current_time again
        current_time = timedelta(minutes=2, seconds=3)
        pretty_time_score = []
        for i, row in formatted_time.iterrows():
            result = re.split('[:,.;]', str(row.center_score))
            times = list(
                filter(lambda t: t != '' and t.isnumeric() and len(t) == 2, result))  # split into 00:10 into 00 and 10
            if len(times) != 2:
                current_time -= timedelta(seconds=1)
            else:
                current_time = timedelta(minutes=int(times[0]), seconds=int(times[1]))
            hh, mm, ss = str(current_time).split(':')
            pretty_time_score.append(f'{mm}:{ss}')
        # endregion
        formatted_time['pretty_time'] = pretty_time_score
        # region fill right and left command's score
        left_pretty_score = []
        right_pretty_score = []
        left_score = 0
        right_score = 0
        for i, row in formatted_time.iterrows():
            l_score = row.left_score
            r_score = row.right_score
            # region left_score preprocessing
            if l_score is not None and str(l_score).isnumeric():
                l_score = int(l_score)
                if l_score - left_score <= 15 or r_score == str(left_score):
                    left_score = l_score
            left_pretty_score.append(left_score)
            # endregion
            # region right_score preprocessing
            if r_score is not None and str(r_score).isnumeric():
                r_score = int(r_score)
                if r_score - right_score <= 15 or r_score == l_score:
                    right_score = r_score
            right_pretty_score.append(right_score)
            # endregion
        # endregion
        formatted_time['left_pretty_score'] = left_pretty_score
        formatted_time['right_pretty_score'] = right_pretty_score
        formatted_time.to_csv(self.__full_time_csv_output_path, index=False)
        return self.__full_time_csv_output_path

    @staticmethod
    def center_comparator(lhs_row, rhs_row):
        lhs_center_x = lhs_row['center_x']
        rhs_center_x = rhs_row['center_x']
        lhs_center_y = lhs_row['center_y']
        rhs_center_y = rhs_row['center_y']

        if abs(lhs_center_y - rhs_center_y) < 5:
            return lhs_center_x - rhs_center_x
        else:
            return rhs_center_y - lhs_center_y

    def __preprocess_frags(self, all_kills_df: pd.DataFrame, nicknames: list, formatted_time: pd.DataFrame) -> str:
        # region remove falsy detection based on nicknames similarity
        matched = False
        for i, row in all_kills_df.iterrows():
            matched = False
            for nickname in nicknames:
                if ratio(nickname, row['text']) > 0.7:
                    all_kills_df.at[i, 'nickname'] = nickname
                    matched = True
                    break
            if not matched:
                all_kills_df.at[i, 'nickname'] = None
        all_kills_df = all_kills_df.dropna(subset=['nickname'])
        # endregion
        # region extract bboxes and create center_x and center_y instead
        top = 206
        centers_x = []
        centers_y = []
        for i, row in all_kills_df.iterrows():
            (top_left, top_right, bottom_right, bottom_left) = row['bbox']
            centers_x.append((top_left[0] + top_right[0]) / 2)
            centers_y.append((top - top_left[1] + top - bottom_left[1]) / 2)
        all_kills_df['center_x'] = centers_x
        all_kills_df['center_y'] = centers_y
        # endregion
        # remove redundant rows based on real placement
        all_kills_df = all_kills_df.query('center_y > 80')
        current_frame_idx = all_kills_df['current_frame'].unique()

        # region create DataFrame consisting of time, score and players statistic in current round
        match_df = pd.DataFrame(
            columns=['frame', 'time_s', 'active_player', 'helper_player', 'passive_player', 'left_score', 'time_score',
                     'right_score'])
        for i in current_frame_idx:
            # fetch all players by current frame
            current_player_score_df = all_kills_df.query(f'current_frame == {i}')
            # Formatted time doesn't consist of all time detection
            # Then Add None and fill it later
            current_time_df = formatted_time.query(f'frame == {i}')
            if current_time_df.empty:
                current_time_df = {
                    'time_s': None,
                    'left_pretty_score': None,
                    'pretty_time': None,
                    'right_pretty_score': None,
                }
            else:
                current_time_df = current_time_df.iloc[0]
            # sort them by center_y and center_x
            sorted_player_score = pd.DataFrame(
                sorted((current_player_score_df.to_dict('records')), key=cmp_to_key(self.center_comparator)))
            # flatten center_y values
            for j in range(0, len(sorted_player_score) - 1):
                if sorted_player_score.iloc[j]['center_y'] - sorted_player_score.iloc[j + 1]['center_y'] < 5:
                    sorted_player_score.at[j, 'center_y'] = sorted_player_score.iloc[j + 1]['center_y']
            # fetch all center_y values and group record by it
            center_y_idxs = sorted_player_score['center_y'].unique()

            for center_y_idx in center_y_idxs:
                # fetch 2 or 3 players in one row
                score_row = sorted_player_score.query(f'center_y == {center_y_idx}')
                players_count = len(score_row)
                if players_count < 2 or players_count > 3:
                    continue
                # choose who is active_player/helper_player/passive_player by their place
                active_player, helper_player, passive_player = None, None, None
                if players_count == 2:
                    active_player = score_row.iloc[0]['nickname']
                    passive_player = score_row.iloc[1]['nickname']
                if players_count == 3:
                    active_player = score_row.iloc[0]['nickname']
                    helper_player = score_row.iloc[1]['nickname']
                    passive_player = score_row.iloc[2]['nickname']

                new_row = pd.DataFrame({
                    'frame': [i],
                    'time_s': [sorted_player_score.iloc[0]['time_s']],
                    'active_player': [active_player],
                    'helper_player': [helper_player],
                    'passive_player': [passive_player],
                    'left_score': [current_time_df['left_pretty_score']],
                    'time_score': [current_time_df['pretty_time']],
                    'right_score': [current_time_df['right_pretty_score']]
                })
                match_df = pd.concat([match_df, pd.DataFrame(new_row)])
        # endregion
        match_df = match_df.sort_values(by=['frame'])
        # region fill remain time scores
        match_time_score = list(match_df.time_score)
        match_timedelta = []
        current_time = timedelta(minutes=2, seconds=3)
        full_time_score = [match_df.iloc[0]['time_score'] or current_time]

        for i in range(1, len(match_df)):
            time_score = match_df.iloc[i]['time_score']
            if time_score is None:
                current_time -= timedelta(seconds=match_df.iloc[i]['frame'] - match_df.iloc[i - 1]['frame'])
            else:
                result = re.split('[:,.;]', str(time_score))
                current_time = timedelta(minutes=int(result[0]), seconds=int(result[1]))
            hh, mm, ss = str(current_time).split(':')
            full_time_score.append(f'{mm}:{ss}')

        match_df['time_score'] = full_time_score
        # endregion
        # region fill remain left and right score
        full_left_score = [match_df.iloc[0]['left_score'] or 0]
        full_right_score = [match_df.iloc[0]['left_score'] or 0]
        for i in range(1, len(match_df)):
            if match_df.iloc[i]['left_score'] is None:
                full_left_score.append(full_left_score[i - 1])
            else:
                full_left_score.append(match_df.iloc[i]['left_score'])
            if match_df.iloc[i]['right_score'] is None:
                full_right_score.append(full_right_score[i - 1])
            else:
                full_right_score.append(match_df.iloc[i]['right_score'])
        match_df['right_score'] = full_right_score
        match_df['left_score'] = full_left_score
        # endregion
        # drop duplicated frags in the same round exclusive assistants
        match_df = match_df.drop_duplicates(
            subset=['active_player', 'helper_player', 'passive_player', 'left_score', 'right_score'])
        # fetch rows with assistance
        assistance_df = match_df[match_df['helper_player'].notna()]
        # find duplicated rows without assistance
        exceed_rows = []
        for i, assistance_row in assistance_df.iterrows():
            found = match_df.query(
                f'active_player == "{assistance_row["active_player"]}" and passive_player == "{assistance_row["passive_player"]}" and left_score == {assistance_row["left_score"]} and right_score == {assistance_row["right_score"]} and helper_player != "{assistance_row["helper_player"]}"')
            if not found.empty:
                exceed_rows.append(found.iloc[0]['frame'])
        match_df = match_df[~match_df['frame'].isin(exceed_rows)]
        # endregion
        match_df.to_csv(self.__full_match_score_csv_output_path)
        return self.__full_match_score_csv_output_path
