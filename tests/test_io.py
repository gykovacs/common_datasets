"""
Testing the IO and encoding functionalities
"""

from common_datasets._io import coalesce

def test_coalesce():
    """
    Testing the coalesce function
    """

    assert coalesce(1, 2) == 1

    assert coalesce(None, 2) == 2
