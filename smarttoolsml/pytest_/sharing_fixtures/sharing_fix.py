# weekdays1, weekdays2 used from conftest.py


def test_add_thursday(weekdays1):
    days = weekdays1.copy()
    days.append("thur")
    assert days == ["mon", "tue", "wed", "thur"]


def test_insert_thursday(weekdays2):
    days = weekdays2.copy()
    days.insert(0, "thur")
    assert days[0] == "thur"
