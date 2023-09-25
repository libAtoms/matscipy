import unittest
import matscipytest
import numpy as np
from ase.lattice.cubic import Diamond
from matscipy.gamma_surface import *

from quippy.potential import Potential


class GammaSurfaceTest(matscipytest.MatSciPyTestCase):
    def setUp(self) -> None:
            self.fmax = 1e-4
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
          # Test functionality of gamma surface code + using model to relax

            surface = GammaSurface(self.at0, np.array([0, 0, 1], dtype=int), np.array([1, 0, 0], dtype=int))
            surface.generate_images(7, 7)
            surface.relax_images(self.model, self.fmax)
            Es = surface.get_surface_energies(self.model)

            assert np.min(Es) >= 0.0
            print(np.max(Es))
    

    def test_stacking_fault(self):
          import matplotlib.pyplot as plt
          from ase.io import write
          # Test that stacking fault reproduces Si results from Si testing framework for SW potential
          surface = StackingFault(self.at0, np.array([0, 0, 1]), np.array([1, 1, 0]), compress=True)
          surface.generate_images(21, z_replications=1)
          surface.relax_images(self.model, self.fmax)
          Es = surface.get_surface_energies(self.model)
          
          surface.plot_gamma_surface()
          plt.savefig("Stack_Compressed.png")

          Es1 = Es.copy()[0, :]
          l1 = len(surface.images[0])

          surf1 = surface.surface_area

          end_E = surface.images[4].get_potential_energy()
        #   write("Surface_structs_compressed.xyz", surface.images)
          
          xd1 = surface.x_disp
          yd1 = surface.y_disp
          sep1 = surface.surface_separation

          cell1 = surface.images[0].cell
          surface1 = surface

          surface = StackingFault(self.at0, np.array([0, 0, 1]), np.array([1, 1, 0]), compress=False)
          surface.generate_images(21, z_replications=1)
          surface.relax_images(self.model, self.fmax)
          Es = surface.get_surface_energies(self.model)[0, :]
          l2 = len(surface.images[0])

          surface.plot_gamma_surface()
          plt.savefig("Stack_Uncompressed.png")


          

          surface = GammaSurface(self.at0, np.array([0, 0, 1]), np.array([1, 1, 0]), compress=True)
          surface.generate_images(21, 21, z_replications=1)
          surface.relax_images(self.model, self.fmax)
          Es = surface.get_surface_energies(self.model)[0, :]
          l2 = len(surface.images[0])

          surface.plot_gamma_surface()
          plt.savefig("Gamma_Compressed.png")

          surface = GammaSurface(self.at0, np.array([0, 0, 1]), np.array([1, 1, 0]), compress=False)
          surface.generate_images(21, 21, z_replications=1)
          surface.relax_images(self.model, self.fmax)
          Es = surface.get_surface_energies(self.model)[0, :]
          l2 = len(surface.images[0])

          surface.plot_gamma_surface()
          plt.savefig("Gamma_Uncompressed.png")


          print(xd1)
          print(yd1)

          print()

          print(surface.x_disp)
          print(surface.y_disp)

          print()
          print(surface1.mapping)

        #   print(Es1.shape, Es.shape)

        #   print(Es)
        #   print(Es1)

        #   print(l1, l2)
        #   print(cell1[:, :])
        #   print(surface.images[0].cell[:, :])

        #   write("Surface_structs_uncompressed.xyz", surface.images)

          print(np.max(Es1), np.max(Es))

          print(np.max(Es1) * surf1, np.max(Es) * surface.surface_area)


tst = GammaSurfaceTest()
tst.setUp()
#tst.test_gamma_surface()
tst.test_stacking_fault()

## Si paper gives 2.17 J/m**2 == 0.1354 eV/A**2 for (0 0 1)
## compressed result = 0.16738092
## uncompressed = 0.07231819518

#Smaller
## Compressed = 0.1673890
## Uncompressed = 0.723372