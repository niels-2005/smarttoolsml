import sys

import pytest

# Skipping complete module

# if sys.platform == "linux":
#     pytest.skip("Skipping entire module on Linux", allow_module_level=True)


const = 9 / 5


def cent_to_fah(cent: int = 0):
    return (cent * const) + 32


# skip tests for a reason
@pytest.mark.skip(reason="Skipping for no reason, just fun")
def test_case01():
    assert type(const) == float


# Skip test if it's not working with python version above 3.5
@pytest.mark.skipif(
    sys.version_info > (3, 5), reason="Doesn't work on Python versions above 3.5"
)
def test_case02():
    assert cent_to_fah() == 32


# Skip if test is not working with higher pytest versions
@pytest.mark.skipif(
    pytest.__version__ > "8.3.1", reason="Doesn't work on higher pytest versions"
)
def test_case03():
    assert cent_to_fah(38) == 100.4
