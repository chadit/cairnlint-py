from abc import ABC, abstractmethod


class Loader(ABC):
    @abstractmethod
    def load(self, key: str) -> bytes:
        raise NotImplementedError


def require_name(name: str) -> str:
    if not name:
        raise ValueError("name is required")
    return name.strip()


def reset() -> None:
    pass
