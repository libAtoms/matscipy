#
# Copyright 2023 Andreas Klemenz (Fraunhofer IWM)
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

import sys
import unittest
import matscipytest

import numpy as np
import ase
import ase.io
import matscipy.opls
import matscipy.io.opls
import ase.calculators.lammpsrun


class TestOPLSIO(matscipytest.MatSciPyTestCase):
    def test_read_extended_xyz(self):
        struct = matscipy.io.opls.read_extended_xyz('opls_extxyz.xyz')

        self.assertIsInstance(struct, matscipy.opls.OPLSStructure)
        self.assertListEqual(list(struct.numbers), [1, 1])
        self.assertArrayAlmostEqual(
            struct.positions,
            [[4.5, 5.0, 5.0],
             [5.5, 5.0, 5.0]],
            tol=0.01
            )
        self.assertListEqual(list(struct.get_array('molid')), [1, 1])
        self.assertListEqual(list(struct.types), ['H1'])
        self.assertListEqual(list(struct.get_array('type')), ['H1', 'H1'])
        self.assertListEqual(list(struct.get_array('tags')), [0, 0])


    def test_read_block(self):
        data = matscipy.io.opls.read_block('opls_parameters.in', 'Dihedrals')
        self.assertDictionariesEqual(
            data,
            {'H1-C1-C1-H1': [0.00, 0.00, 0.01, 0.00]},
            ignore_case=False
            )

        with self.assertRaises(RuntimeError):
            matscipy.io.opls.read_block('opls_parameters.in', 'Charges')


    def test_read_cutoffs(self):
        cutoffs = matscipy.io.opls.read_cutoffs('opls_cutoffs.in')

        self.assertIsInstance(cutoffs, matscipy.opls.CutoffList)
        self.assertDictionariesEqual(
            cutoffs.nvh,
            {'C1-C1': 1.85, 'C1-H1': 1.15},
            ignore_case=False
            )


    def test_read_parameter_file(self):
        cutoffs, ljq, bonds, angles, dihedrals = matscipy.io.opls.read_parameter_file('opls_parameters.in')

        self.assertIsInstance(cutoffs, matscipy.opls.CutoffList)
        self.assertDictionariesEqual(
            cutoffs.nvh,
            {'C1-C1': 1.85, 'C1-H1': 1.15, 'H1-H1': 0.0},
            ignore_case=False
            )

        self.assertIsInstance(ljq, dict)
        self.assertDictionariesEqual(
            ljq,
            {'C1': [0.001, 3.500, -0.010], 'H1': [0.001, 2.500, 0.010]},
            ignore_case=False
            )
        self.assertIsInstance(ljq.lj_cutoff, float)
        self.assertIsInstance(ljq.c_cutoff, float)
        self.assertAlmostEqual(ljq.lj_cutoff, 12.0, places=2)
        self.assertAlmostEqual(ljq.c_cutoff,  15.0, places=2)
        self.assertIsInstance(ljq.lj_pairs, dict)
        self.assertDictionariesEqual(
            ljq.lj_pairs,
            {'C1-H1': [0.001, 3.4, 11.0]},
            ignore_case=False
            )

        self.assertIsInstance(bonds, matscipy.opls.BondData)
        self.assertDictionariesEqual(
            bonds.nvh,
            {'C1-C1': [10.0, 1.0], 'C1-H1': [10.0, 1.0]},
            ignore_case=False
            )

        self.assertIsInstance(angles, matscipy.opls.AnglesData)
        self.assertDictionariesEqual(
            angles.nvh,
            {'H1-C1-C1': [1.0, 100.0], 'H1-C1-H1': [1.0, 100.0]},
            ignore_case=False
            )

        self.assertIsInstance(dihedrals, matscipy.opls.DihedralsData)
        self.assertDictionariesEqual(
            dihedrals.nvh,
            {'H1-C1-C1-H1': [0.00, 0.00, 0.01, 0.00]},
            ignore_case=False
            )


    def test_read_lammps_data(self):
        test_structure = matscipy.io.opls.read_lammps_data('opls_test.atoms', 'opls_test.parameters')

        self.assertArrayAlmostEqual(test_structure.cell,
                                    [[10.0, 0.0, 0.0],
                                     [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]],
                                    tol=0.01)
        self.assertEqual(len(test_structure), 4)
        self.assertListEqual(list(test_structure.numbers), [1, 6, 6, 1])
        self.assertArrayAlmostEqual(test_structure.positions,
                                    [[3.5, 5.0, 5.0],
                                     [4.5, 5.0, 5.0],
                                     [5.5, 5.0, 5.0],
                                     [6.5, 5.0, 5.0]],
                                    tol=0.01)
        test_velocities = ase.calculators.lammpsrun.convert(
            test_structure.get_velocities(),
            'velocity',
            'ASE',
            'metal'
        )
        self.assertArrayAlmostEqual(test_velocities,
                                    [[0.1, 0.2, 0.3],
                                     [0.0, 0.0, 0.0],
                                     [0.4, 0.5, 0.6],
                                     [0.0, 0.0, 0.0]
                                    ],
                                    tol=0.01)
        self.assertArrayAlmostEqual(test_structure.get_masses(),
                                    [1.008, 12.011, 12.011, 1.008],
                                    tol=0.0001)
        self.assertArrayAlmostEqual(test_structure.get_charges(),
                                    [0.01, -0.01, -0.01, 0.01],
                                    tol=0.001)
        self.assertListEqual(
            list(test_structure.get_array('molid')), [1, 1, 1, 1]
            )
        self.assertEqual(len(test_structure.get_types()), 2)
        self.assertTrue('C1' in test_structure.get_types())
        self.assertTrue('H1' in test_structure.get_types())

        self.assertEqual(len(test_structure.bond_types), 2)
        self.assertTrue('C1-C1' in test_structure.bond_types)
        self.assertTrue('C1-H1' in test_structure.bond_types or
                        'H1-C1' in test_structure.bond_types)
        self.assertTupleEqual(test_structure.bond_list.shape, (3, 3))
        self.assertTrue(
            (test_structure.bond_list[0] == [0, 0, 1]).all() or
            (test_structure.bond_list[0] == [0, 1, 0]).all()
            )
        self.assertTrue(
            (test_structure.bond_list[1] == [1, 1, 2]).all() or
            (test_structure.bond_list[1] == [1, 2, 1]).all()
            )
        self.assertTrue(
            (test_structure.bond_list[2] == [0, 2, 3]).all() or
            (test_structure.bond_list[2] == [0, 3, 2]).all()
            )

        self.assertEqual(len(test_structure.ang_types), 1)
        self.assertTrue('H1-C1-C1' in test_structure.ang_types or
                        'C1-C1-H1' in test_structure.ang_types)
        self.assertTupleEqual(test_structure.ang_list.shape, (2, 4))
        self.assertTrue(
            (test_structure.ang_list[0] == [0, 0, 1, 2]).all() or
            (test_structure.ang_list[0] == [0, 2, 1, 0]).all()
            )
        self.assertTrue(
            (test_structure.ang_list[1] == [0, 1, 2, 3]).all() or
            (test_structure.ang_list[1] == [0, 3, 2, 1]).all()
            )

        self.assertEqual(len(test_structure.dih_types), 1)
        self.assertListEqual(test_structure.dih_types, ['H1-C1-C1-H1'])
        self.assertTupleEqual(test_structure.dih_list.shape, (1, 5))
        self.assertTrue(
            (test_structure.dih_list[0] == [0, 0, 1, 2, 3]).all() or
            (test_structure.dih_list[0] == [0, 3, 2, 1, 0]).all()
            )


    def test_write_lammps_atoms(self):
        c2h2 = ase.Atoms('HC2H', cell=[10., 10., 10.])
        c2h2.set_positions([
            [0., 0., 0.],
            [1., 0., 0.],
            [2., 0., 0.],
            [3., 0., 0.]
            ])

        opls_c2h2 = matscipy.opls.OPLSStructure(c2h2)
        opls_c2h2.set_types(['H1', 'C1', 'C1', 'H1'])

        cutoffs, ljq, bonds, angles, dihedrals = matscipy.io.opls.read_parameter_file('opls_parameters.in')

        opls_c2h2.set_cutoffs(cutoffs)
        opls_c2h2.set_atom_data(ljq)
        opls_c2h2.get_bonds(bonds)
        opls_c2h2.get_angles(angles)
        opls_c2h2.get_dihedrals(dihedrals)

        matscipy.io.opls.write_lammps_atoms('temp', opls_c2h2)
        matscipy.io.opls.write_lammps_definitions('temp', opls_c2h2)


        # Read written structure
        c2h2_written = matscipy.io.opls.read_lammps_data('temp.atoms', 'temp.opls')

        self.assertArrayAlmostEqual(c2h2_written.cell,
                                    [[10.0, 0.0, 0.0],
                                     [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]],
                                    tol=0.01)
        self.assertEqual(len(c2h2_written), 4)
        self.assertListEqual(list(c2h2_written.numbers), [1, 6, 6, 1])
        self.assertArrayAlmostEqual(c2h2_written.positions,
                                    [[0.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [2.0, 0.0, 0.0],
                                     [3.0, 0.0, 0.0]],
                                    tol=0.01)
        self.assertArrayAlmostEqual(c2h2_written.get_velocities(),
                                    np.zeros([4, 3], dtype=float),
                                    tol=0.01)
        self.assertArrayAlmostEqual(c2h2_written.get_masses(),
                                    [1.008, 12.011, 12.011, 1.008],
                                    tol=0.0001)
        self.assertArrayAlmostEqual(c2h2_written.get_charges(),
                                    [0.01, -0.01, -0.01, 0.01],
                                    tol=0.001)
        self.assertListEqual(list(c2h2_written.get_array('molid')),
                             [1, 1, 1, 1])
        self.assertEqual(len(c2h2_written.get_types()), 2)
        self.assertTrue('C1' in c2h2_written.get_types())
        self.assertTrue('H1' in c2h2_written.get_types())

        self.assertEqual(len(c2h2_written.bond_types), 2)
        self.assertTrue('C1-C1' in c2h2_written.bond_types)
        self.assertTrue('C1-H1' in c2h2_written.bond_types or
                        'H1-C1' in c2h2_written.bond_types)
        self.assertTupleEqual(c2h2_written.bond_list.shape, (3, 3))
        bonds = c2h2_written.bond_list[:, 1:].tolist()
        self.assertTrue([0, 1] in bonds or
                        [1, 0] in bonds)
        self.assertTrue([1, 2] in bonds or
                        [2, 1] in bonds)
        self.assertTrue([2, 3] in bonds or
                        [3, 2] in bonds)

        self.assertEqual(len(c2h2_written.ang_types), 1)
        self.assertTrue('H1-C1-C1' in c2h2_written.ang_types or
                        'C1-C1-H1' in c2h2_written.ang_types)
        self.assertTupleEqual(c2h2_written.ang_list.shape, (2, 4))
        angles = c2h2_written.ang_list[:, 1:].tolist()
        self.assertTrue([0, 1, 2] in angles or
                        [2, 1, 0] in angles)
        self.assertTrue([1, 2, 3] in angles or
                        [3, 2, 1] in angles)

        self.assertEqual(len(c2h2_written.dih_types), 1)
        self.assertListEqual(c2h2_written.dih_types, ['H1-C1-C1-H1'])
        self.assertTupleEqual(c2h2_written.dih_list.shape, (1, 5))
        self.assertTrue(
            (c2h2_written.dih_list[0] == [0, 0, 1, 2, 3]).all() or
            (c2h2_written.dih_list[0] == [0, 3, 2, 1, 0]).all()
            )


    def test_write_lammps_definitions(self):
        c2h2 = ase.Atoms('HC2H', cell=[10., 10., 10.])
        c2h2.set_positions(
            [[0., 0., 0.],
             [1., 0., 0.],
             [2., 0., 0.],
             [3., 0., 0.]]
            )

        opls_c2h2 = matscipy.opls.OPLSStructure(c2h2)
        opls_c2h2.set_types(['H1', 'C1', 'C1', 'H1'])

        cutoffs, ljq, bonds, angles, dihedrals = matscipy.io.opls.read_parameter_file('opls_parameters.in')

        opls_c2h2.set_cutoffs(cutoffs)
        opls_c2h2.set_atom_data(ljq)
        opls_c2h2.get_bonds(bonds)
        opls_c2h2.get_angles(angles)
        opls_c2h2.get_dihedrals(dihedrals)

        matscipy.io.opls.write_lammps_definitions('temp', opls_c2h2)


        # Read written parameters
        pair_coeff = []
        bond_coeff = []
        angle_coeff = []
        dihedral_coeff = []
        charges = []

        with open('temp.opls', 'r') as fileobj:
            for line in fileobj.readlines():
                if line.startswith('pair_style'):
                    lj_cutoff = line.split()[2]
                    q_cutoff = line.split()[3]
                elif line.startswith('pair_coeff'):
                    pair_coeff.append(line.split())
                elif line.startswith('bond_coeff'):
                    bond_coeff.append(line.split())
                elif line.startswith('angle_coeff'):
                    angle_coeff.append(line.split())
                elif line.startswith('dihedral_coeff'):
                    dihedral_coeff.append(line.split())
                elif len(line.split()) > 3:
                    if line.split()[3] == 'charge':
                        charges.append(line.split())

        self.assertEqual(len(charges), 2)
        for charge in charges:
            if charge[6] == 'C1':
                self.assertAlmostEqual(float(charge[4]), -0.01, places=2)
            elif charge[6] == 'H1':
                self.assertAlmostEqual(float(charge[4]), 0.01, places=2)

        self.assertAlmostEqual(float(lj_cutoff), 12.0, places=1)
        self.assertAlmostEqual(float(q_cutoff), 15.0, places=1)
        self.assertEqual(len(pair_coeff), 3)
        for pair in pair_coeff:
            if pair[6] == 'C1':
                self.assertAlmostEqual(float(pair[3]), 0.001, places=3)
                self.assertAlmostEqual(float(pair[4]), 3.5, places=1)
            elif pair[6] == 'H1':
                self.assertAlmostEqual(float(pair[3]), 0.001, places=3)
                self.assertAlmostEqual(float(pair[4]), 2.5, places=1)
            elif pair[7] == 'H1-C1' or pair[7] == 'C1-H1':
                self.assertAlmostEqual(float(pair[3]), 0.001, places=3)
                self.assertAlmostEqual(float(pair[4]), 3.4, places=1)
                self.assertAlmostEqual(float(pair[5]), 11.0, places=1)

        self.assertEqual(len(bond_coeff), 2)
        for bond in bond_coeff:
            if bond[5] == 'C1-C1':
                self.assertAlmostEqual(float(bond[2]), 10.0, places=1)
                self.assertAlmostEqual(float(bond[3]), 1.0, places=1)
            elif bond[5] == 'H1-C1' or bond[5] == 'C1-H1':
                self.assertAlmostEqual(float(bond[2]), 10.0, places=1)
                self.assertAlmostEqual(float(bond[3]), 1.0, places=1)

        self.assertEqual(len(angle_coeff), 1)
        self.assertAlmostEqual(float(angle_coeff[0][2]), 1.0, places=1)
        self.assertAlmostEqual(float(angle_coeff[0][3]), 100.0, places=1)

        self.assertEqual(len(dihedral_coeff), 1)
        self.assertAlmostEqual(float(dihedral_coeff[0][2]), 0.0, places=1)
        self.assertAlmostEqual(float(dihedral_coeff[0][3]), 0.0, places=1)
        self.assertAlmostEqual(float(dihedral_coeff[0][4]), 0.01, places=2)
        self.assertAlmostEqual(float(dihedral_coeff[0][5]), 0.0, places=1)


if __name__ == '__main__':
    unittest.main()
