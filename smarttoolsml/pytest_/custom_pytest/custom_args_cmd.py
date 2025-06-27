# needs to be in conftest.py!

import pytest

qa_config = "qa.txt"
prod_config = "prod.txt"


# add custom arg, example: pytest -v -s --cmdopt=QA
def pytest_addoption(parser):
    parser.addoption("--cmdopt", default="QA")


@pytest.fixture()
def cmdopt(pytestconfig):
    opt = pytestconfig.getoption("cmdopt")
    if opt == "QA":
        f = open(qa_config, "r")
    else:
        f = open(prod_config, "r")
    yield f


# Example test in other file.
def test_argtest01(cmdopt):
    print(f"Read config file: {cmdopt.readline()}")
