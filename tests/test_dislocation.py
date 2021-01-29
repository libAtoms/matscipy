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


def ovito_dxa(atoms, replicate_z=3):
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import ReplicateModifier, DislocationAnalysisModifier
    from ovito.pipeline import StaticSource, Pipeline
    
    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(ReplicateModifier(num_z=replicate_z))
    dxa = DislocationAnalysisModifier(input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC)
    pipeline.modifiers.append(dxa)

    data = pipeline.compute()
    return (np.array(data.dislocations.segments[0].true_burgers_vector),
            data.dislocations.segments[0].length / replicate_z,
            data.dislocations.segments[0])

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
        C12 = 202.0
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
        self.assertArrayAlmostEqual(np.zeros_like(u_volterra), u_stroh[:, 0], tol=1e-5)
        self.assertArrayAlmostEqual(np.zeros_like(u_volterra), u_stroh[:, 1], tol=1e-5)
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

    def check_disloc(self, cls, ref_angle,
                     burgers=0.5 * np.array([1.0, 1.0, 1.0]), tol=10.0):
        alat = 3.14339177996466
        C11 = 523.0266819809012
        C12 = 202.1786296941397
        C44 = 160.88179872237012

        d = cls(alat, C11, C12, C44)
        bulk, disloc = d.build_cylinder(20.0)
        assert len(bulk) == len(disloc)
        # test the consistency
        # displacement = disloc.positions - bulk.positions
        stroh_displacement = d.displacements(bulk.positions,
                                             np.diag(bulk.cell) / 2.0,
                                             self_consistent=True)

        displacement = disloc.positions - bulk.positions

        np.testing.assert_array_almost_equal(displacement, stroh_displacement)

        del disloc.arrays['fix_mask']  # logical properties not supported by Ovito
        b, length, segment = ovito_dxa(disloc)
        self.assertArrayAlmostEqual(np.abs(b), burgers)  # 1/2[111], signs can change
        assert abs(length - disloc.cell[2, 2]) < 0.01
        
        b_hat = np.array(segment.spatial_burgers_vector)
        b_hat /= np.linalg.norm(b_hat)
        
        lines = np.diff(segment.points, axis=0)
        for line in lines:
            t_hat = line / np.linalg.norm(line)
            dot = np.abs(np.dot(t_hat, b_hat))
            angle = np.degrees(np.arccos(dot))
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


    def check_glide_configs(self, cls):
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

    @unittest.skipIf("atomman" not in sys.modules,
                     "requires atomman")
    def test_screw_glide(self):
        self.check_glide_configs(sd.BCCScrew111Dislocation)

    @unittest.skipIf("atomman" not in sys.modules,
                     "requires atomman")
    def test_edge_glide(self):
        self.check_glide_configs(sd.BCCEdge111Dislocation)

    @unittest.skipIf("atomman" not in sys.modules,
                     "requires atomman")
    def test_mixed_glide(self):
        self.check_glide_configs(sd.BCCMixed111Dislocation)

    @unittest.skipIf("atomman" not in sys.modules,
                     "requires atomman")
    def test_edge100_glide(self):
        self.check_glide_configs(sd.BCCEdge100Dislocation)

    @unittest.skipIf("atomman" not in sys.modules,
                         "requires atomman")
    def test_edge100110_glide(self):
            self.check_glide_configs(sd.BCCEdge100110Dislocation)

if __name__ == '__main__':
    unittest.main()
