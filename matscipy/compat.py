import scipy
from packaging.version import Version


def compat_cg_parameters(cg_parameters):
    if Version(scipy.__version__) >= Version('1.12.0'):
        return cg_parameters
    else:
        cg_parameters = cg_parameters.copy()
        if 'rtol' in cg_parameters:
            cg_parameters['tol'] = cg_parameters['rtol']
            del cg_parameters['rtol']
        return cg_parameters
