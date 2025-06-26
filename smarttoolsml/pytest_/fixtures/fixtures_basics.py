import pytest

# Fixture uses: Initialize db connection, open files, etc.
# its optional to return something from fixtures.


@pytest.fixture()
def setup_list():
    print("\n in fixtures... \n")
    cities = ["New York", "London", "Riyadh", "Singapore", "Mumbai"]
    return cities


# setup_list is called above as fixture!
def test_get_item(setup_list):
    assert setup_list[0] == "New York"
    assert setup_list[::2] == ["New York", "Riyadh", "Mumbai"]


# setup_list is called above as fixture
def test_reverse_list(setup_list):
    assert setup_list[::-2] == ["Mumbai", "Riyadh", "New York"]
