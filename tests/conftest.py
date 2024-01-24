import os.path

import pytest
import inspect
import numpy as np


@pytest.fixture
def datafile_directory():
    """Absolute path to the `tests` directory (which also contains the data files)"""
    return os.path.dirname(__file__)


def pytest_make_parametrize_id(config, val, argname):
    '''
    Define how pytest should label parametrized tests
    '''
    
    if inspect.isclass(val):
        # Strip full path (e.g. matscipy.dislocation.X)
        # And also the <class 'x'> wrapper
        v = str(val).split(".")[-1][:-2]

        return f"{argname}={v}"
    
    elif type(val) in [bool, str] or \
        np.issubdtype(type(val), np.integer) or \
        np.issubdtype(type(val), np.floating):
        # Common rule for nice class names
        return f"{argname}={val}"

    # Allow pytest default behaviour
    return None