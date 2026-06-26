import re


def _normalize_fraction(text: str) -> str:
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    return text


def _extract_equation_values(text: str) -> str | None:
    values = re.findall(r"(?:x(?:_\d+)?|x_\{?\d+\}?)=(-?\d+(?:\.\d+)?)", text)
    if not values:
        return None
    ordered = sorted(set(values), key=lambda value: float(value))
    return "{" + ",".join(ordered) + "}"


def _normalize_set(text: str) -> str:
    if not (text.startswith("{") and text.endswith("}")):
        return text
    values = [value for value in text.strip("{}").split(",") if value]
    if not values:
        return "{}"
    try:
        values = sorted(set(values), key=lambda value: float(value))
    except ValueError:
        values = sorted(set(values))
    return "{" + ",".join(values) + "}"


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""

    text = str(answer).strip().lower()
    text = _normalize_fraction(text)
    replacements = {
        "$": "",
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "。": "",
        "，": ",",
        "、": ",",
        "；": ";",
        "：": ":",
        "π": "pi",
        "\\pi": "pi",
        "α": "alpha",
        "\\alpha": "alpha",
        "−": "-",
        "→": "->",
        "⇒": "->",
        " ": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"最终答案[:：]?", "", text)
    text = re.sub(r"答案[:：]?", "", text)
    text = text.replace("非", "not")
    text = text.replace("¬", "not")
    text = text.replace("是一个", "是")
    text = text.replace("是的", "是")
    text = text.replace("常数", "c")

    equation_values = _extract_equation_values(text)
    if equation_values:
        return equation_values

    text = re.sub(r"\bx_\{?(\d+)\}?", r"x_\1", text)
    text = re.sub(r"[\[\]()]", "", text)
    text = _normalize_set(text)
    return text
