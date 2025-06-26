import pytest

weekdays1 = ["mon", "tue", "wed"]
weekdays2 = ["fri", "sat", "sun"]


@pytest.fixture()
def setup01():
    wk1 = weekdays1.copy()
    wk1.append("thur")
    yield wk1
    # Yet Teardown, executed after test, e.g. for deleting files, closing db connections etc.
    print("\n After yield in setup01 Fixture.")
    del wk1  # del wk1 isnt important because python automatically clean. (but for db connections as example)


@pytest.fixture()
def setup02():
    wk2 = weekdays2.copy()
    wk2.insert(0, "thur")
    yield wk2
    print("\n After yield in setup02 Fixture.")
    del wk2


def test_extendlist(setup01):
    setup01.extend(weekdays2)
    assert setup01 == ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]


# test above fixtures (multiple fixtures)
def test_len(setup01, setup02):
    assert len(setup01) == 4
    assert len(setup02) == 4
