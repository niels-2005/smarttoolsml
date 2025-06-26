import math

import pytest

# Parameterizing is useful for Data Driven Tests


# call the function four times with given inputs, like test_param01(82) and so on.
@pytest.mark.parametrize("test_input", [82, 78, 45, 66])
def test_param01(test_input):
    assert test_input >= 45


# call the function three times with given inputs and expected outputs.
@pytest.mark.parametrize("input, output", [(2, 4), (3, 27), (4, 256)])
def test_param02(input, output):
    assert (input**input) == output


# Define the Data outside
data = [([2, 3, 4], "sum", 9), ([2, 3, 4], "prod", 24)]


@pytest.mark.parametrize("a, b, c", data)
def test_param03(a: list, b: str, c: int):
    if b == "sum":
        assert sum(a) == c
    elif b == "prod":
        assert math.prod(a) == c
