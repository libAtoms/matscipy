#
# Copyright 2019-2020 Johannes Hoermann (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
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
        self.assertArrayAlmostEqual(pnp.concentration, self.ref_data ['c'], 1e-6)

if __name__ == '__main__':
    unittest.main()
