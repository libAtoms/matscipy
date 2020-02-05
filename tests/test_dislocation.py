import os
import unittest
import matscipytest

import matscipy.dislocation as sd
import numpy as np

from ase.calculators.lammpslib import LAMMPSlib
from scipy.optimize import minimize


class TestDislocation(matscipytest.MatSciPyTestCase):


    def test_core_position(self):

        dft_alat = 3.19

        dft_C11 = 488
        dft_C12 = 200
        dft_C44 = 137


        dft_elastic_param = [dft_alat,
                             dft_C11,
                             dft_C12,
                             dft_C44]


        cent_x = np.sqrt(6.0)*dft_alat/3.0
        center = (cent_x, 0.0, 0.0)

        # make the cell with dislocation core not in center

        disloc, bulk, u = sd.make_screw_cyl(dft_alat, dft_C11, dft_C12, dft_C44,
                                            cylinder_r=40,
                                            center=center)
        res = minimize(sd.cost_function,
                       (1.0, 0.5),
                       args=(disloc,
                             bulk,
                             40,
                             dft_elastic_param,
                             False),
                       method='Nelder-Mead')

        self.assertArrayAlmostEqual(res.x, center[:2], tol=1e-4)

    def test_elastic_constants_lammpslib(self):
        """
        Test the get_elastic_constants() function using lammpslib calculator
        If lammps crashes, check "lammps.log" file in the tests directory
        See lammps and ASE documentation on how to make lammpslib work
        """

        target_values = np.array([3.14339177996466, # alat
                                  523.0266819809012, # C11
                                  202.1786296941397, # C12
                                  160.88179872237012]) # C44 for eam4

        pot_name = "w_eam4.fs"

        lammps = LAMMPSlib(lmpcmds=["pair_style eam/fs",
                   "pair_coeff * * %s W" % pot_name],
                    atom_types={'W': 1}, keep_alive=True,
                    log_file="lammps.log")

        obtained_valeus = sd.get_elastic_constants(calculator=lammps,
                                                   delta=1.0e-3)

        os.remove("lammps.log")

        self.assertArrayAlmostEqual(obtained_valeus, target_values, tol=1e-4)

if __name__ == '__main__':
    unittest.main()
