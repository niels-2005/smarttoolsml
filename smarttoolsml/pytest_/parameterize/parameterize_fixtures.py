import pytest


# fixture will run two times
# ids are the displayed name in terminal.
@pytest.fixture(
    scope="session", params=[(3, 4), (3, 5)], ids=["tuple (3, 4)", "tuple (3, 5)"]
)
def fixture01(request):
    # first request.param == (3, 4) and so on.
    return request.param


@pytest.fixture(scope="session", params=["access", "slice"], ids=["access", "slice"])
def fixture02(request):
    return request.param


# think like a invisible loop here. (two times fixture01, one time with (3,4), second time with (3, 5))
def test_fix_param01(fixture01):
    assert type(fixture01) == tuple


# function with two different fixtures. runs four times because each fixture got two params.
def test_fix_param02(fixture01, fixture02):
    if fixture02 == "access":
        assert type(fixture01[0]) == int
    elif fixture02 == "slice":
        assert type(fixture01[0]) == int
