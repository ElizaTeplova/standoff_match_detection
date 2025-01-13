from aiogram.fsm.state import StatesGroup, State


class Form(StatesGroup):
    t1_title: State = State()
    t2_title: State = State()
    t1_nicknames: State = State()
    t2_nicknames: State = State()
    video_match: State = State()
