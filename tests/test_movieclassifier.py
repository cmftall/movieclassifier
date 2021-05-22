# tests/test_movieclassifier.py

"""Test cases for the movieclassifier package."""
from movieclassifier import __version__


def test_version() -> None:
    """Test version of the movieclassifier."""
    assert __version__ == "0.1.0"
