# conftest.py
import pytest

# other modules can use these fixtures.


@pytest.fixture(scope="session")
def weekdays1():
    return ["mon", "tue", "wed"]


@pytest.fixture(scope="session")
def weekdays2():
    return ["fri", "sat", "sun"]
