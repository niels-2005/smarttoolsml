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


def test_extendlist(setup01):
    setup01.extend(weekdays2)
    assert setup01 == ["mon", "tue", "wed", "thur", "fri", "sat", "sun"]
