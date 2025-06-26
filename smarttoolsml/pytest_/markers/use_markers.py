import sys

import pytest

# Markers need to be registered in pytest.ini file.

# mark whole module as mark
# pytest -v -m "markerstest"
pytestmark = [pytest.mark.markerstest, pytest.mark.testmarkers]

# how to run

# only sanity tests: pytest -v -m "sanity"
# all tests but no sanity: pytest -v -m "no sanity"
# run multiple markers: pytest -v -m "sanity and str"


@pytest.mark.sanity
def test_str01():
    num = 9 / 4
    s1 = "I like " + "pytest automation"
    assert str(num) == "2.25"
    assert s1 == "I like pytest automation"
    assert s1 + str(num) == "I like pytest automation2.25"


@pytest.mark.sanity
def test_str02():
    letters = "abcdefghijklmnopqrstuvwxyz"
    assert len(letters) == 26


@pytest.mark.sanity
@pytest.mark.str
def test_str03():
    letters = "abcdefghijklmnopqrstuvwxyz"
    assert letters[0] == "a"
    assert letters[-1] == "z" == letters[25]


# expected to fail, if it would pass -> terminal = XPASS
@pytest.mark.xfail(reason="Expected to fail because XYY")
def test_str04():
    letters = "xihdjhfjhwkjdkwjdekfjfk"
    assert letters[0] == "x"


# mark as fail when function not working on linux
@pytest.mark.xfail(sys.platform == "linux", reason="doesnt work on linux")
def test_str05():
    letters = "abcd"
    num = 1234
    result = letters + str(num)
    assert result == "abcd12345"


def test_strslice():
    letters = "abcdefghijklmnopqrstuvwxyz"
    assert letters[:] == letters
    assert letters[10:] == "klmnopqrstuvwxyz"
    assert letters[-3:] == "xyz"
    assert letters[:21:5] == "afkpu"


def test_strsplit():
    s1 = "Python,Pytest and Automation"
    assert s1.split() == ["Python,Pytest", "and", "Automation"]
    assert s1.split(",") == ["Python", "Pytest and Automation"]


def test_strjoin():
    pass
    s1 = "Python,Pytest and Automation"
    l1 = ["Python,Pytest", "and", "Automation"]
    l2 = ["Python", "Pytest and Automation"]
