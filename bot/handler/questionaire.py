from aiogram import Router, F
from aiogram.types import Message, Video
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext

from utils.state import Form

# from keyboard.builder import match

router = Router()

TEAM_MEMBERS = 5


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        f'Hello, {message.from_user.first_name}',
    )


@router.message(Command('match'))
async def fill_match(message: Message, state: FSMContext):
    await state.set_state(Form.t1_title)
    await message.answer("Enter team1 title:")


@router.message(Form.t1_title)
async def form_t1_title(message: Message, state: FSMContext):
    title = message.text.strip()
    if len(title) == 0:
        await message.answer("Incorrect title. Try again.")
        return

    await state.update_data(t1_title=message.text.strip())
    await state.set_state(Form.t2_title)
    await message.answer("Enter team2 title:")


@router.message(Form.t2_title)
async def form_t2_title(message: Message, state: FSMContext):
    title = message.text.strip()
    if len(title) == 0:
        await message.answer("Incorrect title. Try again.")
        return

    await state.update_data(t2_title=message.text.strip())
    await state.set_state(Form.t1_nicknames)
    await message.answer("Enter team1 nicknames:")


@router.message(Form.t1_nicknames)
async def form_t2_title(message: Message, state: FSMContext):
    players = [nickname.strip() for nickname in message.text.split(',')]
    if len(players) != TEAM_MEMBERS:
        await message.answer(
            f"Incorrect number of players. Must be {TEAM_MEMBERS} but {len(players)} given. Try again.")

    await state.update_data(t1_nicknames=players)
    await state.set_state(Form.t2_nicknames)
    await message.answer("Enter team2 nicknames:")


@router.message(Form.t2_nicknames)
async def form_t2_title(message: Message, state: FSMContext):
    players = [nickname.strip() for nickname in message.text.split(',')]
    if len(players) != TEAM_MEMBERS:
        await message.answer(
            f"Incorrect number of players. Must be {TEAM_MEMBERS} but {len(players)} given. Try again.")

    await state.update_data(t2_nicknames=players)
    await state.set_state(Form.video_match)
    await message.answer("Upload match (.mp4):")


@router.message(Form.video_match, F.video)
async def form_video_match(message: Message, state: FSMContext):

    if message.video.mime_type != 'video/mp4':
        await message.answer(f"Incorrect video type. Must be video/mp4 but {message.video.mime_type} given. Try again.")
    if message.video.width > 1280 or message.video.height > 720:
        await message.answer(
            f"Incorrect video resolution. Must be 1280x720 but {message.video.width}x{message.video.height} given. Try again.")
    video_id = message.video[-1].id
    await state.update_data(video_match=video_id)
    data = await state.get_data()
    form_info = (f"{data['t1_title']}: {data['t1_nicknames']}\n"
                 f"{data['t2_title']}: {data['t2_nicknames']}")

    await message.answer_video(video_id, caption=form_info)
