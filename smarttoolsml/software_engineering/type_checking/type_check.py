# pip install mypy
# mypy type_check.py

from collections import Counter


def mode_using_counter(list_of_numbers: list[float]) -> float:
    c = Counter(list_of_numbers)
    return c.most_common(1)[0][0]


# mypy checks if types are correctly defined. (list_of_numbers: list[float])
