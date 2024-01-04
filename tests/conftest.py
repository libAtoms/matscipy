import os.path

import pytest

@pytest.fixture
def datafile_directory():
    """Absolute path to the `tests` directory (which also contains the data files)"""
    return os.path.dirname(__file__)