import matscipytest
import numpy as np
import os.path
import unittest

from matscipy.electrochemistry import PoissonNernstPlanckSystem

class PoissonNernstPlanckSolverTest(matscipytest.MatSciPyTestCase):

    def setUp(self):
        """Provides 0.1 mM NaCl solution at 0.05 V across 100 nm open half space reference data from binary npz file"""
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.ref_data = np.load(
            os.path.join(self.test_path, 'electrochemistry_data',
            'NaCl_c_0.1_mM_0.1_mM_z_+1_-1_L_1e-7_u_0.05_V_seg_200_interface.npz') )

    def test_poisson_nernst_planck_solver_std_interface_bc(self):
        """Tests PNP solver against simple interfacial BC"""
        pnp = PoissonNernstPlanckSystem(
            c=[0.1,0.1], z=[1,-1], L=1e-7, delta_u=0.05,
            N=200, e=1e-12, maxit=20)
        pnp.useStandardInterfaceBC()
        pnp.solve()

        self.assertArrayAlmostEqual(pnp.grid, self.ref_data ['x'])
        self.assertArrayAlmostEqual(pnp.potential, self.ref_data ['u'])
        self.assertArrayAlmostEqual(pnp.concentration, self.ref_data ['c'])

if __name__ == '__main__':
    unittest.main()
