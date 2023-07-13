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
import io
import unittest

import numpy as np
import ase
import matscipy.opls


class TestOPLS(unittest.TestCase):
    def test_bond_data(self):
        """Test the matscipy.opls.BondData class."""

        bond_data = matscipy.opls.BondData({'AB-CD': [1, 2],
                                            'EF-GH': [3, 4]})
        self.assertDictEqual(bond_data.nvh, {'AB-CD': [1, 2],
                                             'EF-GH': [3, 4]})

        # check correct handling of permutations and missing values
        bond_data.set_names(['A1-A2', 'A2-A1'])
        self.assertSetEqual(bond_data.names, {'AB-CD', 'EF-GH', 'A1-A2'})

        self.assertIsNone(bond_data.get_name('A0', 'A0'))
        self.assertMultiLineEqual(bond_data.get_name('A1', 'A2'), 'A1-A2')
        self.assertMultiLineEqual(bond_data.get_name('A2', 'A1'), 'A1-A2')

        # check return values
        self.assertTupleEqual(bond_data.name_value('A0', 'A0'),
                              (None, None))
        self.assertTupleEqual(bond_data.name_value('AB', 'CD'),
                              ('AB-CD', [1, 2]))
        self.assertTupleEqual(bond_data.name_value('CD', 'AB'),
                              ('AB-CD', [1, 2]))

        self.assertListEqual(bond_data.get_value('AB', 'CD'), [1, 2])

    def test_cutoff_data(self):
        """Test the matscipy.opls.CutoffList class."""

        cutoff_data = matscipy.opls.CutoffList({'AB-CD': 1.0, 'EF-GH': 2.0})
        self.assertAlmostEqual(cutoff_data.max(), 2.0, places=5)

    def test_angles_data(self):
        """Test the matscipy.opls.AnglesData class."""

        angles_data = matscipy.opls.AnglesData({'A1-A2-A3': [1, 2],
                                                'A4-A5-A6': [3, 4]})
        self.assertDictEqual(angles_data.nvh, {'A1-A2-A3': [1, 2],
                                               'A4-A5-A6': [3, 4]})

        # check correct handling of permutations and missing values
        angles_data.set_names(['A7-A8-A9', 'A9-A8-A7'])
        self.assertSetEqual(angles_data.names, {'A1-A2-A3',
                                                'A4-A5-A6',
                                                'A7-A8-A9'})

        angles_data.add_name('B1', 'B2', 'B3')
        angles_data.add_name('B3', 'B2', 'B1')
        self.assertSetEqual(angles_data.names, {'A1-A2-A3', 'A4-A5-A6',
                                                'A7-A8-A9', 'B1-B2-B3'})

        self.assertIsNone(angles_data.get_name('A0', 'A0', 'A0'))
        self.assertMultiLineEqual(angles_data.get_name('A1', 'A2', 'A3'),
                                  'A1-A2-A3')
        self.assertMultiLineEqual(angles_data.get_name('A3', 'A2', 'A1'),
                                  'A1-A2-A3')

        # check return values
        self.assertTupleEqual(angles_data.name_value('A0', 'A0', 'A0'),
                              (None, None))
        self.assertTupleEqual(
            angles_data.name_value('A1', 'A2', 'A3'), ('A1-A2-A3', [1, 2])
            )
        self.assertTupleEqual(
            angles_data.name_value('A3', 'A2', 'A1'), ('A1-A2-A3', [1, 2])
            )

    def test_dihedrals_data(self):
        """Test the matscipy.opls.DihedralsData class."""

        dih_data = matscipy.opls.DihedralsData({'A1-A2-A3-A4': [1, 2, 3, 4],
                                                'B1-B2-B3-B4': [5, 6, 7, 8]})
        self.assertDictEqual(dih_data.nvh, {'A1-A2-A3-A4': [1, 2, 3, 4],
                                            'B1-B2-B3-B4': [5, 6, 7, 8]})

        # check correct handling of permutations and missing values
        dih_data.set_names(['C1-C2-C3-C4', 'C4-C3-C2-C1'])
        self.assertSetEqual(
            dih_data.names, {'A1-A2-A3-A4', 'B1-B2-B3-B4', 'C1-C2-C3-C4'}
            )

        dih_data.add_name('D1', 'D2', 'D3', 'D4')
        dih_data.add_name('D4', 'D3', 'D2', 'D1')
        self.assertSetEqual(dih_data.names, {'A1-A2-A3-A4', 'B1-B2-B3-B4',
                                             'C1-C2-C3-C4', 'D1-D2-D3-D4'})

        self.assertIsNone(dih_data.get_name('A0', 'A0', 'A0', 'A0'))
        self.assertMultiLineEqual(
            dih_data.get_name('A1', 'A2', 'A3', 'A4'), 'A1-A2-A3-A4'
            )
        self.assertMultiLineEqual(
            dih_data.get_name('A4', 'A3', 'A2', 'A1'), 'A1-A2-A3-A4'
            )

        # check return values
        self.assertTupleEqual(
            dih_data.name_value('A0', 'A0', 'A0', 'A0'), (None, None)
            )
        self.assertTupleEqual(
            dih_data.name_value('A1', 'A2', 'A3', 'A4'),
            ('A1-A2-A3-A4', [1, 2, 3, 4])
            )
        self.assertTupleEqual(
            dih_data.name_value('A4', 'A3', 'A2', 'A1'),
            ('A1-A2-A3-A4', [1, 2, 3, 4])
            )

    def test_opls_structure(self):
        """Test the matscipy.opls.OPLSStructure class."""

        # check initialization
        c2h6 = ase.Atoms('C2H6', cell=[10., 10., 10.])
        opls_c2h6 = matscipy.opls.OPLSStructure(c2h6)
        self.assertListEqual(opls_c2h6.get_types().tolist(), ['C', 'H'])

        # check correct initial assignment of tags
        self.assertListEqual(opls_c2h6.types[opls_c2h6.get_tags()].tolist(),
                             opls_c2h6.get_chemical_symbols())

        # check correct initial assignment of
        # tags after definition of atom types
        types = ['C1', 'C1', 'H1', 'H1', 'H1', 'H2', 'H2', 'H2']
        opls_c2h6.set_types(types)
        self.assertListEqual(
            opls_c2h6.get_types()[opls_c2h6.get_tags()].tolist(),
            types
            )

        opls_c2h6.set_positions([
            [1.,  0.,  0.],
            [2.,  0.,  0.],
            [0.,  0.,  0.],
            [1.,  1.,  0.],
            [1., -1.,  0.],
            [2.,  0.,  1.],
            [2.,  0., -1.],
            [3.,  0.,  0.]
            ])
        opls_c2h6.center()

        # check that a runtime error is raised
        # in case some cutoffs are not defined
        cutoffs = matscipy.opls.CutoffList(
            {'C1-H1': 1.1, 'C1-H2': 1.1,
             'H1-H2': 0.1, 'H1-H1': 0.1, 'H2-H2': 0.1}
            )
        opls_c2h6.set_cutoffs(cutoffs)

        with self.assertRaises(RuntimeError):
            opls_c2h6.get_neighbors()

        # check for correct neighborlist when all cutoffs are defined
        cutoffs = matscipy.opls.CutoffList(
            {'C1-C1': 1.1, 'C1-H1': 1.1, 'C1-H2': 1.1,
             'H1-H2': 0.1, 'H1-H1': 0.1, 'H2-H2': 0.1}
            )
        opls_c2h6.set_cutoffs(cutoffs)

        opls_c2h6.get_neighbors()

        pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
                 (1, 0), (1, 5), (1, 6), (1, 7),
                 (2, 0), (3, 0), (4, 0),
                 (5, 1), (6, 1), (7, 1)
                 ]

        for i, j in zip(opls_c2h6.ibond, opls_c2h6.jbond):
            self.assertTrue((i, j) in pairs)

        # check atomic charges
        opls_c2h6.set_atom_data({'C1': [1, 2, 3],
                                 'H1': [4, 5, 6],
                                 'H2': [7, 8, 9]})

        for charge, charge_target in zip(opls_c2h6.get_charges(),
                                         [3., 3., 6., 6., 6., 9., 9., 9.]):
            self.assertAlmostEqual(charge, charge_target, places=5)

        # Check correct construction of bond type list and bond list
        # Case 1: Some cutoffs are not defined - check for runtime error
        cutoffs = matscipy.opls.CutoffList(
            {'C1-H1': 1.1, 'C1-H2': 1.1,
             'H1-H2': 0.1, 'H1-H1': 0.1, 'H2-H2': 0.1}
            )
        opls_c2h6.set_cutoffs(cutoffs)

        with self.assertRaises(RuntimeError):
            opls_c2h6.get_bonds()

        # Case 2: All cutoffs are defined
        cutoffs = matscipy.opls.CutoffList(
            {'C1-C1': 1.1, 'C1-H1': 1.1, 'C1-H2': 1.1,
             'H1-H2': 0.1, 'H1-H1': 0.1, 'H2-H2': 0.1}
            )
        opls_c2h6.set_cutoffs(cutoffs)

        # Case 2.1: no bond data provided
        # Case 2.2: bond data is provided and complete
        # Valid lists should be created in both cases
        bond_data = matscipy.opls.BondData({'C1-C1': [1, 2],
                                            'C1-H1': [3, 4],
                                            'C1-H2': [5, 6]})
        for bond_types, bond_list in [opls_c2h6.get_bonds(),
                                      opls_c2h6.get_bonds(bond_data)]:

            # Check for correct list of bond types
            self.assertEqual(bond_types.shape[0], 3)
            self.assertTrue('C1-C1' in bond_types)
            self.assertTrue('H1-C1' in bond_types or 'C1-H1' in bond_types)
            self.assertTrue('H2-C1' in bond_types or 'C1-H2' in bond_types)

            # Check for correct list of bonds
            type_index_c1c1 = np.where(bond_types == 'C1-C1')[0][0]

            if 'H1-C1' in bond_types:
                type_index_c1h1 = np.where(bond_types == 'H1-C1')[0][0]
            else:
                type_index_c1h1 = np.where(bond_types == 'C1-H1')[0][0]

            if 'H2-C1' in bond_types:
                type_index_c1h2 = np.where(bond_types == 'H2-C1')[0][0]
            else:
                type_index_c1h2 = np.where(bond_types == 'C1-H2')[0][0]

            self.assertEqual(bond_list.shape[0], 7)

            bond_list = bond_list.tolist()
            self.assertTrue([type_index_c1c1, 0, 1] in bond_list or
                            [type_index_c1c1, 1, 0] in bond_list)
            self.assertTrue([type_index_c1h1, 0, 2] in bond_list or
                            [type_index_c1h1, 2, 0] in bond_list)
            self.assertTrue([type_index_c1h1, 0, 3] in bond_list or
                            [type_index_c1h1, 3, 0] in bond_list)
            self.assertTrue([type_index_c1h1, 0, 4] in bond_list or
                            [type_index_c1h1, 4, 0] in bond_list)
            self.assertTrue([type_index_c1h2, 1, 5] in bond_list or
                            [type_index_c1h2, 5, 1] in bond_list)
            self.assertTrue([type_index_c1h2, 1, 6] in bond_list or
                            [type_index_c1h2, 6, 1] in bond_list)
            self.assertTrue([type_index_c1h2, 1, 7] in bond_list or
                            [type_index_c1h2, 7, 1] in bond_list)

        # check that a runtime error is raised in
        # case bond data is provided but incomplete
        buf = io.StringIO()
        sys.stdout = buf  # suppress STDOUT while running test
        with self.assertRaises(RuntimeError):
            opls_c2h6.get_bonds(matscipy.opls.BondData({'C1-C1': [1, 2],
                                                        'C1-H1': [3, 4]}))
        sys.stdout = sys.__stdout__

        # Check correct construction of angle type list and angle list

        # Case 1: no angular potentials provided
        # Case 2: angular potentials are provided and complete
        # Valid lists should be created in both cases
        angles_data = matscipy.opls.AnglesData(
            {'H1-C1-H1': [1, 2], 'H2-C1-H2': [3, 4],
             'C1-C1-H1': [5, 6], 'C1-C1-H2': [7, 8]}
            )
        for angle_types, angle_list in [opls_c2h6.get_angles(),
                                        opls_c2h6.get_angles(angles_data)]:

            # Check for correct list of angle types
            self.assertEqual(len(angle_types), 4)
            self.assertTrue('H1-C1-H1' in angle_types)
            self.assertTrue('H2-C1-H2' in angle_types)
            self.assertTrue('C1-C1-H1' in angle_types or
                            'H1-C1-C1' in angle_types)
            self.assertTrue('C1-C1-H2' in angle_types or
                            'H2-C1-C1' in angle_types)

            # Check for correct list of angles
            type_index_h1c1h1 = angle_types.index('H1-C1-H1')
            type_index_h2c1h2 = angle_types.index('H2-C1-H2')

            if 'C1-C1-H1' in angle_types:
                type_index_c1c1h1 = angle_types.index('C1-C1-H1')
            else:
                type_index_c1c1h1 = angle_types.index('H1-C1-C1')

            if 'C1-C1-H2' in angle_types:
                type_index_c1c1h2 = angle_types.index('C1-C1-H2')
            else:
                type_index_c1c1h2 = angle_types.index('H2-C1-C1')

            self.assertEqual(len(angle_list), 12)

            self.assertTrue([type_index_c1c1h1, 2, 0, 1] in angle_list or
                            [type_index_c1c1h1, 1, 0, 2] in angle_list)
            self.assertTrue([type_index_c1c1h1, 3, 0, 1] in angle_list or
                            [type_index_c1c1h1, 1, 0, 3] in angle_list)
            self.assertTrue([type_index_c1c1h1, 4, 0, 1] in angle_list or
                            [type_index_c1c1h1, 1, 0, 4] in angle_list)
            self.assertTrue([type_index_c1c1h2, 5, 1, 0] in angle_list or
                            [type_index_c1c1h2, 0, 1, 5] in angle_list)
            self.assertTrue([type_index_c1c1h2, 6, 1, 0] in angle_list or
                            [type_index_c1c1h2, 0, 1, 6] in angle_list)
            self.assertTrue([type_index_c1c1h2, 7, 1, 0] in angle_list or
                            [type_index_c1c1h2, 0, 1, 7] in angle_list)
            self.assertTrue([type_index_h1c1h1, 2, 0, 3] in angle_list or
                            [type_index_h1c1h1, 3, 0, 2] in angle_list)
            self.assertTrue([type_index_h1c1h1, 2, 0, 4] in angle_list or
                            [type_index_h1c1h1, 4, 0, 2] in angle_list)
            self.assertTrue([type_index_h1c1h1, 3, 0, 4] in angle_list or
                            [type_index_h1c1h1, 4, 0, 3] in angle_list)
            self.assertTrue([type_index_h2c1h2, 6, 1, 5] in angle_list or
                            [type_index_h2c1h2, 5, 1, 6] in angle_list)
            self.assertTrue([type_index_h2c1h2, 7, 1, 5] in angle_list or
                            [type_index_h2c1h2, 5, 1, 7] in angle_list)
            self.assertTrue([type_index_h2c1h2, 7, 1, 6] in angle_list or
                            [type_index_h2c1h2, 6, 1, 7] in angle_list)

        # check that a runtime error is raised in case
        # angular potentials are provided but incomplete
        buf = io.StringIO()
        sys.stdout = buf  # suppress STDOUT while running test
        angles_data = matscipy.opls.AnglesData(
            {'H1-C1-H1': [1, 2], 'H2-C1-H2': [3, 4],
             'C1-C1-H1': [5, 6]}
            )
        with self.assertRaises(RuntimeError):
            angle_types, angle_list = opls_c2h6.get_angles(angles_data)
        sys.stdout = sys.__stdout__

        # Check correct construction of dihedral type list and dihedral list

        # Case 1: no dihedral potentials provided
        # Case 2: dihedral potentials are provided and complete
        # Valid lists should be created in both cases
        dih_data = matscipy.opls.DihedralsData({'H1-C1-C1-H2': [1, 2, 3, 4]})

        for dih_types, dih_list in [opls_c2h6.get_dihedrals(),
                                    opls_c2h6.get_dihedrals(dih_data)]:

            # Check for correct list of dihedral types
            self.assertEqual(len(dih_types), 1)
            self.assertTrue('H1-C1-C1-H2' in dih_types or
                            'H2-C1-C1-H1' in dih_types)

            # Check for correct list of dihedrals
            self.assertEqual(len(dih_list), 9)

            self.assertTrue([0, 2, 0, 1, 5] in dih_list or
                            [0, 5, 1, 0, 2] in dih_list)
            self.assertTrue([0, 3, 0, 1, 5] in dih_list or
                            [0, 5, 1, 0, 3] in dih_list)
            self.assertTrue([0, 4, 0, 1, 5] in dih_list or
                            [0, 5, 1, 0, 4] in dih_list)
            self.assertTrue([0, 2, 0, 1, 6] in dih_list or
                            [0, 6, 1, 0, 2] in dih_list)
            self.assertTrue([0, 3, 0, 1, 6] in dih_list or
                            [0, 6, 1, 0, 3] in dih_list)
            self.assertTrue([0, 4, 0, 1, 6] in dih_list or
                            [0, 6, 1, 0, 4] in dih_list)
            self.assertTrue([0, 2, 0, 1, 7] in dih_list or
                            [0, 7, 1, 0, 2] in dih_list)
            self.assertTrue([0, 3, 0, 1, 7] in dih_list or
                            [0, 7, 1, 0, 3] in dih_list)
            self.assertTrue([0, 4, 0, 1, 7] in dih_list or
                            [0, 7, 1, 0, 4] in dih_list)


if __name__ == '__main__':
    unittest.main()
