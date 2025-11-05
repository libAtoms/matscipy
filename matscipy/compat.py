import scipy
from packaging.version import Version


def compat_cg_parameters(cg_parameters):
    cg_parameters = cg_parameters.copy()
    if Version(scipy.__version__) >= Version('1.12.0'):
        # scipy >= 1.12.0 uses 'rtol' instead of 'tol'
        if 'tol' in cg_parameters:
            cg_parameters['rtol'] = cg_parameters['tol']
            del cg_parameters['tol']
        return cg_parameters
    else:
        # scipy < 1.12.0 uses 'tol'
        if 'rtol' in cg_parameters:
            cg_parameters['tol'] = cg_parameters['rtol']
            del cg_parameters['rtol']
        return cg_parameters
