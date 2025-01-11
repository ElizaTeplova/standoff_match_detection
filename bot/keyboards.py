from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,  # buttons below the message
    InlineKeyboardButton
)


class KeyboardConfiguration:
    START_MATCH_RECOGNITION: str = 'Start recognition'
    BACK_TO_INPUT_FIRST_TEAM: str = 'Back to input team 1'
    BACK_TO_INPUT_SECOND_TEAM: str = 'Back to input team 2'
    BACK_TO_INPUT_FIRST_TEAM_NICKNAMES: str = 'Back to input team 1 nicknames'
    BACK_TO_INPUT_SECOND_TEAM_NICKNAMES: str = 'Back to input team 2 nicknames'
    CONFIRM_INPUT = 'Start report formation'


main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=KeyboardConfiguration.START_MATCH_RECOGNITION), KeyboardButton(text='Second Button')],
        [KeyboardButton(text='Third Button'), KeyboardButton(text='Fouth Button')],
    ],
    resize_keyboard=True,
    one_time_keyboard=True,
    input_field_placeholder='Select button',
    selective=True  # Show keyboard for each user separately
)
