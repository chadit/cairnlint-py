from typing import Protocol


class Storer(Protocol):
    def load(self, key: str) -> bytes: ...


class FileStore:
    def load(self, key: str) -> bytes:
        return key.encode()
