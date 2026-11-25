class Ledger:
    def __init__(self, entries: list[int]) -> None:
        self._entries: list[int] = entries
        self._reads: int = 0

    def append(self, entry: int) -> None:
        return self._entries.append(entry)

    def total(self) -> int:
        self._reads += 1
        return sum(self._entries)
