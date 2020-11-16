from __future__ import print_function

import matscipytest
from matscipy.calculators.AbellTersoffBrenner import KumagaiTersoff, Kumagai_CompMaterSci_39_457_Si_py_var
from ase import Atoms


class TestKumagaiTersoff(matscipytest.MatSciPyTestCase):

    def get_potential_energy_test(self):
        d = 2  # Si2 bondlength
        a = Atoms([14]*4, [(d, 0, d), (0, 0, 0), (d, 0, 0), (0, 0, d)],
                  cell=(100, 100, 100))
        a.center(vacuum=10.0)
        calculator = KumagaiTersoff(**Kumagai_CompMaterSci_39_457_Si_py_var)
        a.set_calculator(calculator)
        ene = a.get_potential_energy()
        # print(ene)
        assert (ene + 6.956526471027033) < 0.01  # TODO: compare with LAMMPS
