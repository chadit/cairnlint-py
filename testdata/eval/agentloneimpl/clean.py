from typing import Protocol


class Sender(Protocol):
    def send(self, message: str) -> str: ...


class EmailSender:
    def send(self, message: str) -> str:
        return f"mail:{message}"


class SmsSender:
    def send(self, message: str) -> str:
        return f"sms:{message}"
