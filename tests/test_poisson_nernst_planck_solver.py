import numpy as np
import os.path
import pytest

from matscipy.electrochemistry import PoissonNernstPlanckSystem
rel_err_norm_tol = 1.0e-10 # relative tolerance when comparing test data against reference
abs_err_tol = 1.0e-12

@pytest.fixture
def ref_npz_interface(shared_datadir):
    """Provides 0.1 mM NaCl solution at 0.05 V across 100 nm open half space reference data from binary npz file"""
    return np.load( os.path.join(shared_datadir,'NaCl_c_0.1_mM_0.1_mM_z_+1_-1_L_1e-7_u_0.05_V_seg_200_interface.npz') )

def test_poisson_nernt_planck_solver_std_interface_bc(ref_npz_interface):
    """Test PNP solver against simple interfacial BC"""
    pnp = PoissonNernstPlanckSystem(
        c=[0.1,0.1], z=[1,-1], L=1e-7, delta_u=0.05,
        N=200, e=1e-12, maxit=20)
    pnp.useStandardInterfaceBC()
    pnp.init()
    pnp.solve()

    assert np.all( pnp.grid - ref_npz_interface['x'] < abs_err_tol )
    assert np.all( pnp.potential - ref_npz_interface['u'] < abs_err_tol )
    assert np.all( pnp.concentration - ref_npz_interface['c'] < abs_err_tol )
