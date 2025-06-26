import pytest


# function in a function = factorie
@pytest.fixture()
def setup01():
    def get_structure(name):
        if name == "list":
            return [1, 2, 3]
        elif name == "tuple":
            return (1, 2, 3)

    return get_structure


# setup01 is a few times callable
def test_fact_fixture(setup01):
    assert type(setup01("list")) == list
    assert type(setup01("tuple")) == tuple
