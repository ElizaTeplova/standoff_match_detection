import dearpygui.dearpygui as dpg
import config as cfg
import label as lbl
import mimetypes


def save_callback():
    val = dpg.get_value(cfg.LABEL_TEAM1_TITLE_TAG)
    print(f"Saving file... {val}")


# sender - id, app_data - dict
def file_selected_callback(sender: str, app_data: dict):
    full_path = app_data.get('file_path_name')
    [file_type, file_encoding] = mimetypes.guess_type(full_path, strict=True)
    if file_type != 'video/mp4':
        return
    print(f"type sender {type(sender)}, app_data {type(app_data)}")
    print(f" {sender}, {app_data}")


def file_cancelled_callback(sender, app_data):
    print(f"type sender {type(sender)}, app_data {type(app_data)}")
    print(f" {sender}, {app_data}")


print(cfg.team1_player_tags)

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

    with dpg.group(horizontal=True):
        with dpg.group(horizontal=False):
            dpg.add_text('Team 1')
            for i in range(cfg.TEAM_PLAYERS_AMOUNT):
                dpg.add_input_text(tag=cfg.team1_player_tags[i], default_value=f'Player{i + 1}')

        with dpg.group(horizontal=False):
            dpg.add_text('Team 2')
            for i in range(cfg.TEAM_PLAYERS_AMOUNT):
                dpg.add_input_text(tag=cfg.team2_player_tags[i], default_value=f'Player{i + 1}')

        dpg.add_button(
            tag=cfg.BTN_SELECT_FILE_TAG,
            label='Select Match',
            callback=lambda: dpg.show_item(cfg.FILE_SELECTOR_TAG)
        )
        dpg.add_button(tag=cfg.BTN_SAVE_TAG, label='Save', callback=save_callback)

    dpg.add_input_text(
        tag=cfg.LABEL_TEAM1_TITLE_TAG,
        label='String',
        default_value='Quick brown fox'
    )
    dpg.add_slider_float(label='Float', default_value=5, min_value=0.0, max_value=10.0)

with dpg.window(
        label=lbl.VIEWPORT_OUTPUT_LABEL,
        pos=cfg.VP_OUT_START_POSITION,
        width=cfg.VP_OUT_WIDTH
):
    dpg.add_text('Hello, world')
    dpg.add_button(label='Save')
    dpg.add_input_text(label='String', default_value='Quick brown fox')
    dpg.add_slider_float(label='Float', default_value=5, min_value=0.0, max_value=10.0)

dpg.create_viewport(title='StandoffReviewer', width=cfg.VIEWPORT_WIDTH, height=cfg.VIEWPORT_HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
# dpg.set_primary_window(cfg.PRIMARY_WINDOW_TAG, True)
dpg.start_dearpygui()
dpg.destroy_context()
