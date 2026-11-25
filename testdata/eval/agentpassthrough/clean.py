class Journal:
    def __init__(self, lines: list[str]) -> None:
        self._lines: list[str] = lines
        self._writes: int = 0

    def write(self, line: str) -> None:
        return self._lines.append(line.rstrip())

    def count(self) -> int:
        self._writes += 1
        return len(self._lines)
