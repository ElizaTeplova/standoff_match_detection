from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

sub_channel = InlineKeyboardMarkup(
    inline_keyboard=[[
        InlineKeyboardButton(text='Subscribe', url='tg://resolve?domain=rankoffofficial')
    ]]
)