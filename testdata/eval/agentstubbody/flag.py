def load_user(user_id: str) -> dict[str, str]:
    raise NotImplementedError


def flush() -> None:
    """Write buffered records to disk."""
