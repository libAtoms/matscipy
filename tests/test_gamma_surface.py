import unittest
import matscipytest
import numpy as np
from ase.lattice.cubic import Diamond
from matscipy.gamma_surface import *
from matscipy.calculators.manybody.explicit_forms.tersoff_brenner import TersoffBrenner, \
                                                                         Brenner_PRB_42_9458_C_I
from matscipy.calculators.manybody import Manybody
from matscipy.dislocation import get_elastic_constants

class GammaSurfaceTest(matscipytest.MatSciPyTestCase):
    def setUp(self) -> None:
        self.fmax = 1e-3
        self.model = Manybody(**TersoffBrenner(Brenner_PRB_42_9458_C_I))
        alat, _, __, ___ = get_elastic_constants(calculator=self.model, symbol="C", verbose=False)
        self.at0 = Diamond('C', latticeconstant=alat)
        
    
    def test_gamma_surface(self):
        surface = GammaSurface(self.at0, np.array([0, 0, 1], dtype=int), np.array([1, 0, 0], dtype=int))
        surface.generate_images(3, 3)
        surface.relax_images(self.model, self.fmax)
        Es = surface.get_energy_densities(self.model)

        assert np.min(Es) >= 0.0
        assert np.allclose(np.max(Es), 0.3340102322222366)

    def test_stacking_fault(self):
        surface = StackingFault(self.at0, np.array([1, 1, 0]), np.array([-1, 1, 0]))
        surface.generate_images(9, z_reps=1, path_ylims=[0, 1])
        surface.relax_images(self.model, self.fmax)
        Es = surface.get_energy_densities(self.model)[0, :]
        print(np.max(Es))
        assert np.allclose(np.max(Es), 0.2628608260182023)

    def test_disloc_stacking_fault(self):
        from matscipy.dislocation import DiamondGlideScrew
        surface = StackingFault(self.at0, DiamondGlideScrew)
        surface.generate_images(9, z_reps=1, path_ylims=[0, 0.5])
        surface.relax_images(self.model, self.fmax)
        Es = surface.get_energy_densities(self.model)[0, :]
if __name__ == '__main__':
    unittest.main()