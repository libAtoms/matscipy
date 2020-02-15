import os
import unittest
import matscipytest
import sys

import matscipy.dislocation as sd
import numpy as np

from ase.calculators.lammpslib import LAMMPSlib
from scipy.optimize import minimize
from matscipy.calculators.eam import EAM

try:
    import matplotlib
except ImportError:
    print("matplotlib not found: will skip some tests")

try:
    import lammps
except ImportError:
    print("lammps not found: will skip some tests")

class TestDislocation(matscipytest.MatSciPyTestCase):
    """Class to store test for dislocation.py module."""

    def test_core_position(self):
        """Make screw dislocation and fit a core position to it.

        Tests that the fitted core position is same as created
        """
        print("Fitting the core, takes about 10 seconds")
        dft_alat = 3.19

        dft_C11 = 488
        dft_C12 = 200
        dft_C44 = 137

        dft_elastic_param = [dft_alat,
                             dft_C11,
                             dft_C12,
                             dft_C44]

        cent_x = np.sqrt(6.0)*dft_alat/3.0
        center = np.array((cent_x, 0.0, 0.0))

        # make the cell with dislocation core not in center
        disloc, bulk, u = sd.make_screw_cyl(dft_alat, dft_C11,
                                            dft_C12, dft_C44,
                                            cylinder_r=40,
                                            center=center)

        #  initial guess is the center of the cell
        initial_guess = np.diagonal(disloc.cell).copy()[:2] / 2.0

        res = minimize(sd.cost_function,
                       initial_guess,
                       args=(disloc,
                             bulk,
                             40,
                             dft_elastic_param,
                             False, False),
                       method='Nelder-Mead')

        self.assertArrayAlmostEqual(res.x, initial_guess + center[:2], tol=1e-4)

    def test_elastic_constants_EAM(self):
        """Test the get_elastic_constants() function using matscipy EAM calculator."""
        target_values = np.array([3.14339177996466,  # alat
                                  523.0266819809012,  # C11
                                  202.1786296941397,  # C12
                                  160.88179872237012])  # C44 for eam4

        pot_name = "w_eam4.fs"
        calc_EAM = EAM(pot_name)
        obtained_values = sd.get_elastic_constants(calculator=calc_EAM,
                                                   delta=1.0e-3)

        self.assertArrayAlmostEqual(obtained_values, target_values, tol=1e-4)

    # This function tests the lammpslib and LAMMPS installation and thus skipped during automated testing
    @unittest.skipIf("lammps" not in sys.modules,
                     "LAMMPS installation is required and thus is not good for automated testing")
    def test_elastic_constants_lammpslib(self):
        """Test the get_elastic_constants() function using lammpslib.

        If lammps crashes, check "lammps.log" file in the tests directory
        See lammps and ASE documentation on how to make lammpslib work
        """
        print("WARNING: In case lammps crashes no error message is printed: ",
              "check 'lammps.log' file in test folder")
        target_values = np.array([3.14339177996466,  # alat
                                  523.0266819809012,  # C11
                                  202.1786296941397,  # C12
                                  160.88179872237012])  # C44 for eam4

        pot_name = "w_eam4.fs"

        lammps = LAMMPSlib(lmpcmds=["pair_style eam/fs",
                                    "pair_coeff * * %s W" % pot_name],
                           atom_types={'W': 1}, keep_alive=True,
                           log_file="lammps.log")

        obtained_values = sd.get_elastic_constants(calculator=lammps,
                                                   delta=1.0e-3)

        os.remove("lammps.log")

        self.assertArrayAlmostEqual(obtained_values, target_values, tol=1e-4)

    # This function tests the lammpslib and LAMMPS installation and thus skipped during automated testing
    @unittest.skipIf("lammps" not in sys.modules,
                     "LAMMPS installation is required and thus is not good for automated testing")
    def test_screw_cyl_lammpslib(self):
        """Test make_crew_cyl() and call lammpslib caclulator.

        If lammps crashes, check "lammps.log" file in the tests directory
        See lammps and ASE documentation on how to make lammpslib work
        """
        print("WARNING: In case lammps crashes no error message is printed: ",
              "check 'lammps.log' file in test folder")

        alat = 3.14339177996466
        C11 = 523.0266819809012
        C12 = 202.1786296941397
        C44 = 160.88179872237012

        cylinder_r = 40

        cent_x = np.sqrt(6.0)*alat/3.0
        center = (cent_x, 0.0, 0.0)

        pot_name = "w_eam4.fs"
        target_toten = -13086.484626  # Target value for w_eam4

        lammps = LAMMPSlib(lmpcmds=["pair_style eam/fs",
                                    "pair_coeff * * %s W" % pot_name],
                           atom_types={'W': 1}, keep_alive=True,
                           log_file="lammps.log")

        disloc_ini, bulk_ini, __ = sd.make_screw_cyl(alat, C11, C12, C44,
                                                     cylinder_r=cylinder_r,
                                                     l_extend=center)

        disloc_ini.set_calculator(lammps)
        ini_toten = disloc_ini.get_potential_energy()
        self.assertAlmostEqual(ini_toten, target_toten, places=4)

        disloc_fin, __, __ = sd.make_screw_cyl(alat, C11, C12, C44,
                                               cylinder_r=cylinder_r,
                                               center=center)
        disloc_fin.set_calculator(lammps)
        fin_toten = disloc_fin.get_potential_energy()
        self.assertAlmostEqual(fin_toten, target_toten, places=4)
        os.remove("lammps.log")

    # Also requires version of atomman higher than 1.3.1.1
    @unittest.skipIf(not "matplotlib" in sys.modules,
        "Requires matplotlib which is not a part of automated testing environment")
    def test_differential_displacement(self):
        """Test differential_displacement() function from atomman

        """
        alat = 3.14339177996466
        C11 = 523.0266819809012
        C12 = 202.1786296941397
        C44 = 160.88179872237012

        cylinder_r = 40

        cent_x = np.sqrt(6.0) * alat / 3.0
        center = (cent_x, 0.0, 0.0)

        disloc_ini, bulk_ini, __ = sd.make_screw_cyl(alat, C11, C12, C44,
                                                     cylinder_r=cylinder_r,
                                                     l_extend=center)

        disloc_fin, __, __ = sd.make_screw_cyl(alat, C11, C12, C44,
                                               cylinder_r=cylinder_r,
                                               center=center)

        fig = sd.show_NEB_configurations([disloc_ini, disloc_fin], bulk_ini,
                                         xyscale=5.0, show=False)
        print("'dd_test.png' will be created: check the displacement map")
        fig.savefig("dd_test.png")

    def test_read_dislo_QMMM(self):
        """Test read_dislo_QMMM() function"""

        alat = 3.14339177996466
        C11 = 523.0266819809012
        C12 = 202.1786296941397
        C44 = 160.88179872237012

        target_values = {"Nat": 1443,  # total number of atoms
                         "QM": 12,  # number of atoms in QM region
                         "MM": 876,  # number of atoms in MM region
                         "fixed": 555}  # number of fixed atoms

        cylinder_r = 40

        disloc, __, __ = sd.make_screw_cyl(alat, C11, C12, C44,
                                           cylinder_r=cylinder_r)

        x, y, _ = disloc.positions.T
        R = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2)

        R_cut = alat * np.sqrt(6.0) / 2.0 + 0.2  # radius for 12 QM atoms
        QM_mask = R < R_cut

        region = disloc.get_array("region")
        region[QM_mask] = np.full_like(region[QM_mask], "QM")
        disloc.set_array("region", region)

        disloc.write("test_read_dislo.xyz")

        test_disloc, __ = sd.read_dislo_QMMM("test_read_dislo.xyz")
        Nat = len(test_disloc)

        self.assertEqual(Nat, target_values["Nat"])

        total_Nat_type = 0
        for atom_type in ["QM", "MM", "fixed"]:

            Nat_type = np.count_nonzero(region == atom_type)
            total_Nat_type += Nat_type
            self.assertEqual(Nat_type, target_values[atom_type])

        self.assertEqual(Nat, total_Nat_type)  # total number of atoms in region is equal to Nat (no atoms unmapped)
        # TODO
        #  self.assertAtomsAlmostEqual(disloc, test_disloc) - gives an error of _cell attribute new ase version?

    def test_stroh_solution(self):
        """Builds isotropic Stroh solution and compares it to Volterra solution"""

        alat = 3.14
        C11 = 523.0
        C12 = 202.0
        C44 = 160.49

        # A = 2. * C44 / (C11 - C12)
        # print(A) # A = 0.999937 very isotropic material
        cylinder_r = 40
        burgers = alat * np.sqrt(3.0) / 2.

        __, W_bulk, u_stroh = sd.make_screw_cyl(alat, C11, C12, C44,
                                                cylinder_r=cylinder_r)
        x, y, __ = W_bulk.positions.T
        # make center of the cell at the dislocation core
        x -= x.mean()
        y -= y.mean()
        u_volterra = np.arctan2(y, x) * burgers / (2.0 * np.pi)

        # compare x and y components with zeros - isotropic solution
        self.assertArrayAlmostEqual(np.zeros_like(u_volterra), u_stroh[:, 0], tol=1e-5)
        self.assertArrayAlmostEqual(np.zeros_like(u_volterra), u_stroh[:, 1], tol=1e-5)
        #  compare z component with simple Volterra solution
        self.assertArrayAlmostEqual(u_volterra, u_stroh[:, 2])


if __name__ == '__main__':
    unittest.main()
