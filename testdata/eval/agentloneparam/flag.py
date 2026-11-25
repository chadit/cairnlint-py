def render(template: str, strict: bool) -> str:
    if strict:
        return "strict:" + template
    return template


PAGES = [render("page", True), render("header", strict=True)]
