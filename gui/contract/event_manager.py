from typing import List
from .event_listener import EventListener


class EventManager:
    def __init__(self):
        self.listeners: List[EventListener] = []

    def subscribe(self, listener: EventListener) -> None:
        self.listeners.append(listener)

    def unsubscribe(self, listener: EventListener) -> None:
        self.listeners.remove(listener)

    def notify(self, message: str) -> None:
        for listener in self.listeners:
            listener.update(message)
