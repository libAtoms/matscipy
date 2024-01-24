import os.path

import pytest

@pytest.fixture
def datafile_directory():
    """Absolute path to the `tests` directory (which also contains the data files)"""
    return os.path.dirname(__file__)


def pytest_make_parametrize_id(config, val, argname):
    '''
    Define how pytest should label parametrized tests
    '''
    if "matscipy." in str(val):
        # Assume val is a path to a class or function
        # e.g. matscipy.dislocation.BCCEdge111Dislocation
        v = str(val).split(".")[-1]

        if "'>" in v:
            v = v[:-2]
    else:
        v = str(val)

    return f"{argname}={v}"