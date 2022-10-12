#
# Copyright 2018, 2020-2021 Petr Grigorev (Warwick U.)
#           2020 James Kermode (Warwick U.)
#           2019 Lars Pastewka (U. Freiburg)
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
    print("matplotlib not found: skipping some tests")

try:
    import lammps
except ImportError:
    print("lammps not found: skipping some tests")

try:
    import atomman
except ImportError:
    print("atomman not found: skipping some tests")

try:
    import ovito
except ImportError:
    print("ovito not found: skipping some tests")


class TestDislocation(matscipytest.MatSciPyTestCase):
    """Class to store test for dislocation.py module."""

    @unittest.skipIf("atomman" not in sys.modules, 'requires Stroh solution from atomman to run')
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

        cent_x = np.sqrt(6.0) * dft_alat / 3.0
        center = np.array((cent_x, 0.0, 0.0))

        # make the cell with dislocation core not in center
        disloc, bulk, u = sd.make_screw_cyl(dft_alat, dft_C11,
                                            dft_C12, dft_C44,
                                            cylinder_r=40,
                                            center=center)

        #  initial guess is the center of the cell
        initial_guess = np.diagonal(disloc.cell).copy()[:2] / 2.0

        core_pos = sd.fit_core_position(disloc, bulk, dft_elastic_param,
                                        origin=initial_guess,
                                        hard_core=False, core_radius=40)
        self.assertArrayAlmostEqual(core_pos, initial_guess + center[:2], tol=1e-4)

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
    @unittest.skipIf("lammps" not in sys.modules or
                     "atomman" not in sys.modules,
                     "LAMMPS installation and Stroh solution are required")
    def test_screw_cyl_lammpslib(self):
        """Test make_crew_cyl() and call lammpslib calculator.

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

        cent_x = np.sqrt(6.0) * alat / 3.0
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

        disloc_ini.calc = lammps
        ini_toten = disloc_ini.get_potential_energy()
        self.assertAlmostEqual(ini_toten, target_toten, places=4)

        disloc_fin, __, __ = sd.make_screw_cyl(alat, C11, C12, C44,
                                               cylinder_r=cylinder_r,
                                               center=center)
        disloc_fin.calc = lammps
        fin_toten = disloc_fin.get_potential_energy()
        self.assertAlmostEqual(fin_toten, target_toten, places=4)
        os.remove("lammps.log")

    # Also requires version of atomman higher than 1.3.1.1
    @unittest.skipIf("matplotlib" not in sys.modules or "atomman" not in sys.modules,
                     "Requires matplotlib and atomman which is not a part of automated testing environment")
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

    @unittest.skipIf("atomman" not in sys.modules, 'requires Stroh solution from atomman to run')
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
        os.remove("test_read_dislo.xyz")
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

    @unittest.skipIf("atomman" not in sys.modules, 'requires Stroh solution from atomman to run')
    def test_stroh_solution(self):
        """Builds isotropic Stroh solution and compares it to Volterra solution"""

        alat = 3.14
        C11 = 523.0
        C12 = 202.05
        C44 = 160.49

        # A = 2. * C44 / (C11 - C12)
        # print(A) # A = 0.999937 very isotropic material.
        # At values closer to 1.0 Stroh solution is numerically unstable and does not pass checks
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
        self.assertArrayAlmostEqual(np.zeros_like(u_volterra), u_stroh[:, 0], tol=1e-4)
        self.assertArrayAlmostEqual(np.zeros_like(u_volterra), u_stroh[:, 1], tol=1e-4)
        #  compare z component with simple Volterra solution
        self.assertArrayAlmostEqual(u_volterra, u_stroh[:, 2])

    def test_make_screw_quadrupole_kink(self):
        """Test the total number of atoms in the quadrupole double kink configuration"""

        alat = 3.14
        n1u = 5
        kink_length = 20

        kink, _, _ = sd.make_screw_quadrupole_kink(alat=alat, n1u=n1u, kink_length=kink_length)
        quadrupole_base, _, _, _ = sd.make_screw_quadrupole(alat=alat, n1u=n1u)

        self.assertEqual(len(kink), len(quadrupole_base) * 2 * kink_length)

    @unittest.skipIf("atomman" not in sys.modules, 'requires Stroh solution from atomman to run')
    def test_make_screw_cyl_kink(self):
        """Test the total number of atoms and number of fixed atoms in the cylinder double kink configuration"""

        alat = 3.14339177996466
        C11 = 523.0266819809012
        C12 = 202.1786296941397
        C44 = 160.88179872237012

        cent_x = np.sqrt(6.0) * alat / 3.0
        center = [cent_x, 0.0, 0.0]

        cylinder_r = 40
        kink_length = 26

        kink, large_disloc, straight_bulk = sd.make_screw_cyl_kink(alat, C11, C12, C44, kink_length=kink_length,
                                                                   cylinder_r=cylinder_r, kind="double")

        # check the total number of atoms as compared to make_screw_cyl()
        disloc, _, _ = sd.make_screw_cyl(alat, C11, C12, C12, cylinder_r=cylinder_r, l_extend=center)

        self.assertEqual(len(kink), len(disloc) * 2 * kink_length)

        kink_fixed_atoms = kink.constraints[0].get_indices()
        reference_fixed_atoms = kink.constraints[0].get_indices()

        # check that the same number of atoms is fixed
        self.assertEqual(len(kink_fixed_atoms), len(reference_fixed_atoms))
        # check that the fixed atoms are the same and with same positions
        self.assertArrayAlmostEqual(kink_fixed_atoms, reference_fixed_atoms)
        self.assertArrayAlmostEqual(kink.positions[kink_fixed_atoms],
                                    large_disloc.positions[reference_fixed_atoms])

    def test_slice_long_dislo(self):
        """Function to test slicing tool"""

        alat = 3.14339177996466
        b = np.sqrt(3.0) * alat / 2.0
        n1u = 5
        kink_length = 20

        kink, straight_dislo, kink_bulk = sd.make_screw_quadrupole_kink(alat=alat, n1u=n1u, kink_length=kink_length)
        quadrupole_base, _, _, _ = sd.make_screw_quadrupole(alat=alat, n1u=n1u)

        sliced_kink, core_positions = sd.slice_long_dislo(kink, kink_bulk, b)

        # check the number of sliced configurations is equal to length of 2 * kink_length * 3 (for double kink)
        self.assertEqual(len(sliced_kink), kink_length * 3 * 2)

        # check that the bulk and kink slices are the same size
        bulk_sizes = [len(slice[0]) for slice in sliced_kink]
        kink_sizes = [len(slice[1]) for slice in sliced_kink]
        self.assertArrayAlmostEqual(bulk_sizes, kink_sizes)

        # check that the size of slices are the same as single b configuration
        self.assertArrayAlmostEqual(len(quadrupole_base), len(sliced_kink[0][0]))

        right_kink, straight_dislo, kink_bulk = sd.make_screw_quadrupole_kink(alat=alat, n1u=n1u, kind="right",
                                                                              kink_length=kink_length)
        sliced_right_kink, _ = sd.slice_long_dislo(right_kink, kink_bulk, b)
        # check the number of sliced configurations is equal to length of kink_length * 3 - 2 (for right kink)
        self.assertEqual(len(sliced_right_kink), kink_length * 3 - 2)

        left_kink, straight_dislo, kink_bulk = sd.make_screw_quadrupole_kink(alat=alat, n1u=n1u, kind="left",
                                                                              kink_length=kink_length)
        sliced_left_kink, _ = sd.slice_long_dislo(left_kink, kink_bulk, b)
        # check the number of sliced configurations is equal to length of kink_length * 3 - 1 (for left kink)
        self.assertEqual(len(sliced_left_kink), kink_length * 3 - 1)

    def check_disloc(self, cls, ref_angle, structure="BCC", test_u=True,
                     burgers=0.5 * np.array([1.0, 1.0, 1.0]), tol=10.0):
        alat = 3.14339177996466
        C11 = 523.0266819809012
        C12 = 202.1786296941397
        C44 = 160.88179872237012

        d = cls(alat, C11, C12, C44)
        bulk, disloc = d.build_cylinder(20.0)
        assert len(bulk) == len(disloc)

        if test_u:
            # test the consistency
            # displacement = disloc.positions - bulk.positions
            stroh_displacement = d.displacements(bulk.positions,
                                                 np.diag(bulk.cell) / 2.0,
                                                 self_consistent=d.self_consistent)

            displacement = disloc.positions - bulk.positions

            np.testing.assert_array_almost_equal(displacement, stroh_displacement)

        results = sd.ovito_dxa_straight_dislo_info(disloc, structure=structure)
        assert len(results) == 1
        position, b, line, angle = results[0]
        self.assertArrayAlmostEqual(np.abs(b), burgers)  # signs can change

        err = angle - ref_angle
        print(f'angle = {angle} ref_angle = {ref_angle} err = {err}')
        assert abs(err) < tol

    @unittest.skipIf("atomman" not in sys.modules or 
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_screw_dislocation(self):
        self.check_disloc(sd.BCCScrew111Dislocation, 0.0)

    @unittest.skipIf("atomman" not in sys.modules or 
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_edge_dislocation(self):        
        self.check_disloc(sd.BCCEdge111Dislocation, 90.0)

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_edge100_dislocation(self,):
        self.check_disloc(sd.BCCEdge100Dislocation, 90.0,
                          burgers=np.array([1.0, 0.0, 0.0]))

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_edge100110_dislocation(self,):
        self.check_disloc(sd.BCCEdge100110Dislocation, 90.0,
                          burgers=np.array([1.0, 0.0, 0.0]))

    @unittest.skipIf("atomman" not in sys.modules or 
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_mixed_dislocation(self):
        self.check_disloc(sd.BCCMixed111Dislocation, 70.5)

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_30degree_diamond_partial_dislocation(self,):
        self.check_disloc(sd.DiamondGlide30degreePartial, 30.0,
                          structure="Diamond",
                          burgers=(1.0 / 6.0) * np.array([1.0, 2.0, 1.0]))

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_90degree_diamond_partial_dislocation(self,):
        self.check_disloc(sd.DiamondGlide90degreePartial, 90.0,
                          structure="Diamond",
                          burgers=(1.0 / 6.0) * np.array([2.0, 1.0, 1.0]))


    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_screw_diamond_dislocation(self,):
        self.check_disloc(sd.DiamondGlideScrew, 0.0,
                          structure="Diamond", test_u=False,
                          burgers=(1.0 / 2.0) * np.array([0.0, 1.0, 1.0]))


    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_60degree_diamond_dislocation(self,):
        self.check_disloc(sd.DiamondGlide60Degree, 60.0,
                          structure="Diamond", test_u=False,
                          burgers=(1.0 / 2.0) * np.array([1.0, 0.0, 1.0]))

    def check_glide_configs(self, cls, structure="BCC"):
        alat = 3.14339177996466
        C11 = 523.0266819809012
        C12 = 202.1786296941397
        C44 = 160.88179872237012

        d = cls(alat, C11, C12, C44)
        bulk, disloc_ini, disloc_fin = d.build_glide_configurations(radius=40)

        assert len(bulk) == len(disloc_ini)
        assert len(disloc_ini) == len(disloc_fin)

        assert all(disloc_ini.get_array("fix_mask") ==
                   disloc_fin.get_array("fix_mask"))

        results = sd.ovito_dxa_straight_dislo_info(disloc_ini,
                                                   structure=structure)
        assert len(results) == 1
        ini_x_position = results[0][0][0]

        results = sd.ovito_dxa_straight_dislo_info(disloc_fin,
                                                   structure=structure)
        assert len(results) == 1
        fin_x_position = results[0][0][0]
        # test that difference between initial and final positions are
        # roughly equal to glide distance.
        # Since the configurations are unrelaxed dxa gives
        # a very rough estimation (especially for edge dislcoations)
        # thus tolerance is taken to be ~1 Angstom
        np.testing.assert_almost_equal(fin_x_position - ini_x_position,
                                       d.glide_distance, decimal=0)

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_screw_glide(self):
        self.check_glide_configs(sd.BCCScrew111Dislocation)

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_edge_glide(self):
        self.check_glide_configs(sd.BCCEdge111Dislocation)

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_mixed_glide(self):
        self.check_glide_configs(sd.BCCMixed111Dislocation)

    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_edge100_glide(self):
        self.check_glide_configs(sd.BCCEdge100Dislocation)


    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_edge100110_glide(self):
            self.check_glide_configs(sd.BCCEdge100110Dislocation)


    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_30degree_diamond_partial_glide(self):
            self.check_glide_configs(sd.DiamondGlide30degreePartial,
                                     structure="Diamond")


    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_90degree_diamond_partial_glide(self):
            self.check_glide_configs(sd.DiamondGlide90degreePartial,
                                     structure="Diamond")


    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_screw_diamond_partial_glide(self):
            self.check_glide_configs(sd.DiamondGlideScrew,
                                     structure="Diamond")


    @unittest.skipIf("atomman" not in sys.modules or
                     "ovito" not in sys.modules,
                     "requires atomman and ovito")
    def test_60degree_diamond_partial_glide(self):
            self.check_glide_configs(sd.DiamondGlide60Degree,
                                     structure="Diamond")

    def test_fixed_line_atoms(self):

        from ase.build import bulk

        pot_name = "w_eam4.fs"
        calc_EAM = EAM(pot_name)
        # slightly pressurised cell to avoid exactly zero forces
        W = bulk("W", a=0.9 * 3.143392, cubic=True)

        W = W * [2, 2, 2]
        del W[0]
        W.calc = calc_EAM
        for line_direction in [[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1],
                               [1, 1, 0],
                               [0, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]]:

            fixed_mask = W.positions.T[0] > W.cell[0, 0] / 2.0 - 0.1
            W.set_constraint(sd.FixedLineAtoms(fixed_mask, line_direction))

            line_dir_mask = np.array(line_direction, dtype=bool)
            # forces in direction other than line dir are zero
            assert (W.get_forces()[fixed_mask].T[~line_dir_mask] == 0.0).all()
            # forces in line direction are non zero
            assert (W.get_forces()[fixed_mask].T[line_dir_mask] != 0.0).all()
            # forces on unconstrained atoms are non zero
            assert (W.get_forces()[~fixed_mask] != 0.0).all()


    @unittest.skipIf("lammps" not in sys.modules,
                     "LAMMPS installation is required and thus is not good for automated testing")
    def test_gamma_line(self):

        # eam_4 parameters
        eam4_elastic_param = 3.143392, 527.025604, 206.34803, 165.092165
        dislocation = sd.BCCEdge111Dislocation(*eam4_elastic_param)
        unit_cell = dislocation.unit_cell

        pot_name = "w_eam4.fs"
        # calc_EAM = EAM(pot_name) eam calculator is way too slow
        lammps = LAMMPSlib(lmpcmds=["pair_style eam/fs",
                                    "pair_coeff * * %s W" % pot_name],
                           atom_types={'W': 1}, keep_alive=True,
                           log_file="lammps.log")

        unit_cell.calc = lammps
        shift, E = sd.gamma_line(unit_cell, surface=1, factor=5)

        # target values corresponding to fig 2(a) of
        # J.Phys.: Condens.Matter 25(2013) 395502
        # http://iopscience.iop.org/0953-8984/25/39/395502
        target_E = [0.00000000e+00, 1.57258669e-02,
                    5.31974533e-02, 8.01241031e-02,
                    9.37911067e-02, 9.54010452e-02,
                    9.37911067e-02, 8.01241032e-02,
                    5.31974533e-02, 1.57258664e-02,
                    4.33907234e-14]

        target_shift = [0., 0.27222571, 0.54445143, 0.81667714,
                        1.08890285, 1.36112857, 1.63335428,
                        1.90557999, 2.17780571, 2.45003142,
                        2.72225714]

        np.testing.assert_almost_equal(E, target_E, decimal=3)
        np.testing.assert_almost_equal(shift, target_shift, decimal=3)

if __name__ == '__main__':
    unittest.main()
