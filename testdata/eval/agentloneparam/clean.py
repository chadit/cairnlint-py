def pad(text: str, width: int) -> str:
    if width > len(text):
        return text + " " * (width - len(text))
    return text


COLUMNS = [pad("name", 10), pad("size", 6)]
