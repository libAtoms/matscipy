import unittest
import matscipytest
import numpy as np
from ase.lattice.cubic import Diamond
from matscipy.gamma_surface import *

from quippy.potential import Potential


class GammaSurfaceTest(matscipytest.MatSciPyTestCase):
    def setUp(self) -> None:
        self.fmax = 1e-3
        self.at0 = Diamond('Si', latticeconstant=5.43)

        # Si testing framework SW model string
        # https://github.com/libAtoms/silicon-testing-framework/blob/master/models/SW/model.py
        self.model = Potential('IP SW', param_str="""
            <SW_params n_types="1">
            <per_type_data type="1" atomic_num="14" />
            <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
                p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
            <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14"
                lambda="21.0" gamma="1.20" eps="2.1675" />
            </SW_params>
            """)
    
    def test_gamma_surface(self):
        surface = GammaSurface(self.at0, np.array([0, 0, 1], dtype=int), np.array([1, 0, 0], dtype=int))
        surface.generate_images(3, 3)
        surface.relax_images(self.model, self.fmax)
        Es = surface.get_surface_energies(self.model)

        assert np.min(Es) >= 0.0
        print("Gamma Surface: ", np.max(Es))
        assert np.allclose(np.max(Es), 0.20510077433464793)

    def test_stacking_fault(self):
        surface = StackingFault(self.at0, np.array([1, 1, 0]), np.array([-1, 1, 0]))
        surface.generate_images(9, z_replications=1, path_ylims=[0, 0.5])
        surface.relax_images(self.model, self.fmax)
        Es = surface.get_surface_energies(self.model)[0, :]
        print("Stacking Fault: ", np.max(Es))
        assert np.allclose(np.max(Es), 0.07150004280093887)

    def test_disloc_stacking_fault(self):
        from matscipy.dislocation import DiamondGlideScrew
        surface = StackingFault(self.at0, DiamondGlideScrew)
        surface.generate_images(9, z_replications=1, path_ylims=[0, 0.5])
        surface.relax_images(self.model, self.fmax)
        Es = surface.get_surface_energies(self.model)[0, :]
if __name__ == '__main__':
    unittest.main()