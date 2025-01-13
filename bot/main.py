import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

from bot.handler import questionaire
from middleware.check_subscription import CheckSubscriptionMiddleware
import handler.questionaire
from aiogram import Bot, Dispatcher, F, html
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command, CommandObject
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties

import keyboards

load_dotenv()
dispatcher = Dispatcher()


# @dispatcher.message(CommandStart())
# async def start(message: Message) -> None:
#     await message.answer(
#         f'Hello {html.bold(message.from_user.first_name)}',
#         # reply_markup=keyboards.main_kb
#     )
#
#
# @dispatcher.message(Command(commands=['ping']))
# async def pingpong(message: Message, command: CommandObject) -> None:
#     players = [nickname.strip() for nickname in command.args.split(',')]
#     await message.answer(f'Pong! Players: {players} Len: {len(players)}')
#
#
# @dispatcher.message()
# async def echo(message: Message) -> None:
#     await message.answer(f'Command not found')


async def main():
    logging.info('Bot was started')
    bot = Bot(
        token=os.getenv('BOT_TOKEN'),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )

    logging.info(questionaire.router)
    dispatcher.include_routers(questionaire.router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dispatcher.start_polling(bot)


if __name__ == '__main__':
    logging.basicConfig(
        filename='bot.log',
        level=logging.INFO,
    )
    asyncio.run(main())
