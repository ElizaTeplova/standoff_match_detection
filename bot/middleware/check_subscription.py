from typing import Callable, Awaitable, Dict, Any
from aiogram import BaseMiddleware
from aiogram.types import Message
from keyboard.inline import sub_channel

class CheckSubscriptionMiddleware(BaseMiddleware):
    NOT_SUBSCRIBED: str = 'left'
    async def __call__(
            self,
            handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
            event: Message,
            data: Dict[str, Any]
    ) -> Any:
        chat_member = await event.bot.get_chat_member('@rankoffofficial', event.from_user.id)

        if chat_member.status == CheckSubscriptionMiddleware.NOT_SUBSCRIBED:
            await event.answer(
                'Subscribe to use bot',
                reply_markup=sub_channel
            )
        else:
            return await handler(event, data)