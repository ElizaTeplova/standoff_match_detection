import dearpygui.dearpygui as dpg
import config as cfg
import label as lbl
import mimetypes
import threading
import time
from typing import Tuple, Dict
import pandas as pd
from model.player import Player

running = False
paused = False
progress = 0
selected_math_full_path = None
errors = []
players: Dict[str, Player] = {}


def run_task():
    global running
    global paused
    global progress
    print("Running...")
    dpg.set_item_label(cfg.BTN_SAVE_TAG, 'Pause')
    for i in range(1, 101):
        while paused:
            time.sleep(0.1)
        if not running:
            return
        progress = i
        print(i)
        dpg.set_value(cfg.PROGRESS_BAR_TAG, 1 / 100 * i)
        dpg.configure_item(cfg.PROGRESS_BAR_TAG, overlay=f'{i}%')
        time.sleep(0.05)
    dpg.set_item_label(cfg.BTN_SAVE_TAG, 'Save')
    print('Finished')
    running = False


def draw_player_stats(players: Dict[str, Player]) -> None:
    nicknames = players.keys()
    values_group_series = []
    for player in players.values():
        values_group_series.append(player.helper_score)
    for player in players.values():
        values_group_series.append(player.passive_score)
    for player in players.values():
        values_group_series.append(player.active_score)

    ilabels = ["Helper Score", "Passive Score", "Active Score"]  # Score categories
    glabels = [(nickname, idx) for idx, nickname in enumerate(nicknames)]
    glabels = tuple(glabels)
    group_c = len(ilabels)  # Number of Score Categories

    with dpg.plot(label='Player Score', height=400, width=-1):
        dpg.add_plot_legend(location=dpg.mvPlot_Location_SouthEast)
        # region X-axis
        dpg.add_plot_axis(dpg.mvYAxis, label='Player', tag='y_axis_bar_group', no_gridlines=True)
        dpg.set_axis_ticks(dpg.last_item(), glabels)
        # endregion
        # region Y-axis
        with dpg.plot_axis(dpg.mvXAxis, label='Score', tag='x_axis_bar_group', auto_fit=True):
            dpg.set_axis_limits(dpg.last_item(), 0, max(values_group_series))
            dpg.add_bar_group_series(
                values=values_group_series,
                label_ids=ilabels,
                group_size=group_c,
                tag='bar_group_series',
                label='Scores',
                horizontal=True
            )
        #endregion


def draw_table(match_full_path: str) -> None:
    # global players
    players = {}
    match_df = pd.read_csv('C:\\studying\SW_7_semester\practice\\textdetection\pythonProject1\match.csv')
    with dpg.tree_node(label='Math Stats'):
        with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit, row_background=True, reorderable=True,
                       resizable=True, no_host_extendX=False, hideable=True,
                       borders_innerV=True, delay_search=True, borders_outerV=True, borders_innerH=True,
                       borders_outerH=True):
            dpg.add_table_column(label='N')
            dpg.add_table_column(label='Active Player')
            dpg.add_table_column(label='Helper Player')
            dpg.add_table_column(label='Defeated Player')

            for i, row in match_df.iterrows():
                with dpg.table_row():
                    active_player: str = row.loc['active_player']
                    helper_player = None if pd.isna(row.loc['helper_player']) else row.loc['helper_player']
                    passive_player: str = row.loc['passive_player']
                    # region Active Player
                    if not players.__contains__(active_player):
                        players[active_player] = Player(
                            nickname=active_player,
                            active_score=1,
                        )
                    else:
                        players.get(active_player).inc_active()
                    # endregion
                    # region Passive Player
                    if not players.__contains__(passive_player):
                        players[passive_player] = Player(
                            nickname=passive_player,
                            passive_score=1,
                        )
                    else:
                        players.get(passive_player).inc_passive()
                    # endregion
                    # region Helper Player
                    if not players.__contains__(helper_player):
                        players[helper_player] = Player(
                            nickname=helper_player,
                            helper_score=1,
                        )
                    else:
                        players.get(helper_player).inc_helper()

                    dpg.add_text(str(i))
                    dpg.add_text(active_player)
                    dpg.add_text(helper_player)
                    dpg.add_text(passive_player)

    players.pop(None)
    sorted_players = dict(sorted(players.items(), key=lambda item: item[1].active_score, reverse=False))
    for v in players.values():
        print(v.__str__())
    with dpg.tree_node(label='Player Score'):
        draw_player_stats(sorted_players)


def save_callback():
    global selected_math_full_path, errors
    player1_nicknames, player2_nicknames = get_players()

    if len(player1_nicknames) != cfg.TEAM_PLAYERS_AMOUNT:
        errors.append(f"Team1 players doesnt consist of 5 players, {len(player1_nicknames)} given")

    if len(player2_nicknames) != cfg.TEAM_PLAYERS_AMOUNT:
        errors.append(f"Team2 players doesnt consist of 5 players, {len(player2_nicknames)} given")

    full_path = selected_math_full_path
    if not isinstance(full_path, str):
        errors.append(f"File wasn't selected")
    else:
        [file_type, file_encoding] = mimetypes.guess_type(full_path, strict=True)
        # if file_type != 'video/mp4':
        #     errors.append(f'File type must be video/mp4, but {file_type} given')
    if errors:
        error_msg = ''
        for i in range(len(errors)):
            error_msg += f"{i + 1}) {errors[i]}\n"
        with dpg.popup(cfg.BTN_SAVE_TAG, modal=True, mousebutton=dpg.mvMouseButton_Left, tag=cfg.ERROR_POPUP_TAG):
            dpg.add_text(error_msg)
        errors.clear()
        return

    global running
    global paused
    if not running:
        print('Started')
        running = True
        paused = False
        thread = threading.Thread(target=run_task(), args=(), daemon=True)
        thread.start()

    else:
        if not paused:
            print('Paused...')
            paused = True
            dpg.set_item_label(cfg.BTN_SAVE_TAG, 'Save')
            return
        print('Resuming...')
        paused = False
        dpg.set_item_label(cfg.BTN_SAVE_TAG, 'Pause')


def get_players() -> Tuple[list, list]:
    player1_nicknames = []
    player2_nicknames = []
    for tag in cfg.team1_player_tags:
        nickname = str(dpg.get_value(tag))
        if len(nickname) != 0:
            player1_nicknames.append(nickname)

    for tag in cfg.team2_player_tags:
        nickname = str(dpg.get_value(tag))
        if len(nickname) != 0:
            player2_nicknames.append(nickname)
    return player1_nicknames, player2_nicknames


# sender - id, app_data - dict
def file_selected_callback(sender: str, app_data: dict):
    global selected_math_full_path
    full_path = app_data.get('file_path_name')
    [file_type, file_encoding] = mimetypes.guess_type(full_path, strict=True)
    selected_math_full_path = full_path
    print(f"type sender {type(sender)}, app_data {type(app_data)}")
    print(f" {sender}, {app_data}")


def file_cancelled_callback(sender, app_data):
    print(f"type sender {type(sender)}, app_data {type(app_data)}")
    print(f" {sender}, {app_data}")


dpg.create_context()

with dpg.window(
        tag=cfg.PRIMARY_WINDOW_TAG,
        label=lbl.VIEWPORT_INPUT_LABEL,
        pos=cfg.VP_IN_START_POSITION,
        # width=cfg.VP_IN_WIDTH
):
    with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=file_selected_callback,
            id=cfg.FILE_SELECTOR_TAG,
            width=700,
            height=400
    ) as file_dialog:
        dpg.add_file_extension(".mp4", color=(255, 0, 0, 255))
        dpg.add_file_extension(".py", color=(0, 255, 0, 255))

    with dpg.group(horizontal=True):
        with dpg.group(horizontal=False):
            dpg.add_text('Team 1')
            for i in range(cfg.TEAM_PLAYERS_AMOUNT):
                dpg.add_input_text(tag=cfg.team1_player_tags[i], default_value=f'Player{i + 1}',
                                   width=cfg.INPUT_PLAYER_WIDTH)

        with dpg.group(horizontal=False):
            dpg.add_text('Team 2')
            for i in range(cfg.TEAM_PLAYERS_AMOUNT):
                dpg.add_input_text(tag=cfg.team2_player_tags[i], default_value=f'Player{i + 1}',
                                   width=cfg.INPUT_PLAYER_WIDTH)

    dpg.add_button(
        tag=cfg.BTN_SELECT_FILE_TAG,
        label='Select Match',
        callback=lambda: dpg.show_item(cfg.FILE_SELECTOR_TAG)
    )
    dpg.add_button(tag=cfg.BTN_SAVE_TAG, label='Save', callback=save_callback)
    dpg.add_progress_bar(
        tag=cfg.PROGRESS_BAR_TAG,
        default_value=0,
        width=-1,
        overlay='0%'
    )

# region right window
with dpg.window(
        label=lbl.VIEWPORT_OUTPUT_LABEL,
        pos=cfg.VP_OUT_START_POSITION,
        width=cfg.VP_OUT_WIDTH,
        height=cfg.VIEWPORT_HEIGHT
):
    draw_table('')
# endregion

dpg.create_viewport(title='StandoffReviewer', width=cfg.VIEWPORT_WIDTH, height=cfg.VIEWPORT_HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
# dpg.set_primary_window(cfg.PRIMARY_WINDOW_TAG, True)
dpg.start_dearpygui()
dpg.destroy_context()
