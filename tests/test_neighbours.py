#
# Copyright 2014-2015, 2017-2021 Lars Pastewka (U. Freiburg)
#           2020 Jonas Oldenstaedt (U. Freiburg)
#           2014 James Kermode (Warwick U.)
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

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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
# ======================================================================

import unittest

import numpy as np

import ase
import ase.io as io
import ase.lattice.hexagonal
from ase.build import bulk, molecule

import matscipytest
from matscipy.neighbours import (
    neighbour_list,
    first_neighbours,
    triplet_list,
    mic,
    CutoffNeighbourhood,
    MolecularNeighbourhood,
    get_jump_indicies,
)
from matscipy.fracture_mechanics.idealbrittlesolid import triangular_lattice_slab
from matscipy.molecules import Molecules

###


class TestNeighbours(matscipytest.MatSciPyTestCase):

    def test_neighbour_list(self):
        for pbc in [True, False, [True, False, True]]:
            a = io.read('aC.cfg')
            a.set_pbc(pbc)
            j, dr, i, abs_dr, shift = neighbour_list("jDidS", a, 1.85)

            self.assertTrue((np.bincount(i) == np.bincount(j)).all())

            r = a.get_positions()
            dr_direct = mic(r[j]-r[i], a.cell)
            self.assertArrayAlmostEqual(r[j]-r[i]+shift.dot(a.cell), dr_direct)

            abs_dr_from_dr = np.sqrt(np.sum(dr*dr, axis=1))
            abs_dr_direct = np.sqrt(np.sum(dr_direct*dr_direct, axis=1))

            self.assertTrue(np.all(np.abs(abs_dr-abs_dr_from_dr) < 1e-12))
            self.assertTrue(np.all(np.abs(abs_dr-abs_dr_direct) < 1e-12))

            self.assertTrue(np.all(np.abs(dr-dr_direct) < 1e-12))

    def test_neighbour_list_atoms_outside_box(self):
        for pbc in [True, False, [True, False, True]]:
            a = io.read('aC.cfg')
            a.set_pbc(pbc)
            a.positions[100, :] += a.cell[0, :]
            a.positions[200, :] += a.cell[1, :]
            a.positions[300, :] += a.cell[2, :]
            j, dr, i, abs_dr, shift = neighbour_list("jDidS", a, 1.85)

            self.assertTrue((np.bincount(i) == np.bincount(j)).all())

            r = a.get_positions()
            dr_direct = mic(r[j]-r[i], a.cell)
            self.assertArrayAlmostEqual(r[j]-r[i]+shift.dot(a.cell), dr_direct)

            abs_dr_from_dr = np.sqrt(np.sum(dr*dr, axis=1))
            abs_dr_direct = np.sqrt(np.sum(dr_direct*dr_direct, axis=1))

            self.assertTrue(np.all(np.abs(abs_dr-abs_dr_from_dr) < 1e-12))
            self.assertTrue(np.all(np.abs(abs_dr-abs_dr_direct) < 1e-12))

            self.assertTrue(np.all(np.abs(dr-dr_direct) < 1e-12))

    def test_neighbour_list_triangular(self):
        a = triangular_lattice_slab(1.0, 2, 2)
        i, j, D, S = neighbour_list('ijDS', a, 1.2)
        D2 = a.positions[j] - a.positions[i] + S.dot(a.cell)
        self.assertArrayAlmostEqual(D, D2)

    def test_small_cell(self):
        a = ase.Atoms('C', positions=[[0.5, 0.5, 0.5]], cell=[1, 1, 1],
                      pbc=True)
        i, j, dr, shift = neighbour_list("ijDS", a, 1.1)
        assert np.bincount(i)[0] == 6
        assert (dr == shift).all()

        i, j = neighbour_list("ij", a, 1.5)
        assert np.bincount(i)[0] == 18

        a.set_pbc(False)
        i = neighbour_list("i", a, 1.1)
        assert len(i) == 0

        a.set_pbc([True, False, False])
        i = neighbour_list("i", a, 1.1)
        assert np.bincount(i)[0] == 2

        a.set_pbc([True, False, True])
        i = neighbour_list("i", a, 1.1)
        assert np.bincount(i)[0] == 4

    def test_out_of_cell_small_cell(self):
        a = ase.Atoms('CC', positions=[[0.5, 0.5, 0.5],
                                       [1.1, 0.5, 0.5]],
                      cell=[1, 1, 1], pbc=False)
        i1, j1, r1 = neighbour_list("ijd", a, 1.1)
        a.set_cell([2, 1, 1])
        i2, j2, r2 = neighbour_list("ijd", a, 1.1)

        self.assertArrayAlmostEqual(i1, i2)
        self.assertArrayAlmostEqual(j1, j2)
        self.assertArrayAlmostEqual(r1, r2)

    def test_out_of_cell_large_cell(self):
        a = ase.Atoms('CC', positions=[[9.5, 0.5, 0.5],
                                       [10.1, 0.5, 0.5]],
                      cell=[10, 10, 10], pbc=False)
        i1, j1, r1 = neighbour_list("ijd", a, 1.1)
        a.set_cell([20, 10, 10])
        i2, j2, r2 = neighbour_list("ijd", a, 1.1)

        self.assertArrayAlmostEqual(i1, i2)
        self.assertArrayAlmostEqual(j1, j2)
        self.assertArrayAlmostEqual(r1, r2)

    def test_hexagonal_cell(self):
        for sx in range(3):
            a = ase.lattice.hexagonal.Graphite('C', latticeconstant=(2.5, 10.0),
                                               size=[sx+1, sx+1, 1])
            i = neighbour_list("i", a, 1.85)
            self.assertTrue(np.all(np.bincount(i)==3))

    def test_first_neighbours(self):
        i = [1,1,1,1,3,3,3]
        self.assertArrayAlmostEqual(first_neighbours(5, i), [-1,0,4,4,7,7])
        i = [0,1,2,3,4,5]
        self.assertArrayAlmostEqual(first_neighbours(6, i), [0,1,2,3,4,5,6])
        i = [0,1,2,3,5,6]
        self.assertArrayAlmostEqual(first_neighbours(8, i), [0,1,2,3,4,4,5,6,6])
        i = [0,1,2,3,3,5,6]
        self.assertArrayAlmostEqual(first_neighbours(8, i), [0,1,2,3,5,5,6,7,7])

    def test_multiple_elements(self):
        a = molecule('HCOOH')
        a.center(vacuum=5.0)
        io.write('HCOOH.cfg', a)
        i = neighbour_list("i", a, 1.85)
        self.assertArrayAlmostEqual(np.bincount(i), [2,3,1,1,1])

        cutoffs = {(1, 6): 1.2}
        i = neighbour_list("i", a, cutoffs)
        self.assertArrayAlmostEqual(np.bincount(i), [0,1,0,0,1])

        cutoffs = {(6, 8): 1.4}
        i = neighbour_list("i", a, cutoffs)
        self.assertArrayAlmostEqual(np.bincount(i), [1,2,1])

        cutoffs = {('H', 'C'): 1.2, (6, 8): 1.4}
        i = neighbour_list("i", a, cutoffs)
        self.assertArrayAlmostEqual(np.bincount(i), [1,3,1,0,1])

    def test_noncubic(self):
        a = bulk("Al", cubic=False)
        i, j, d = neighbour_list("ijd", a, 3.1)
        self.assertArrayAlmostEqual(np.bincount(i), [12])
        self.assertArrayAlmostEqual(d, [2.86378246]*12)

    def test_out_of_bounds(self):
        nat = 10
        atoms = ase.Atoms(numbers=range(nat),
                          cell=[(0.2, 1.2, 1.4),
                                (1.4, 0.1, 1.6),
                                (1.3, 2.0, -0.1)])
        atoms.set_scaled_positions(3 * np.random.random((nat, 3)) - 1)

        for p1 in range(2):
            for p2 in range(2):
                for p3 in range(2):
                    atoms.set_pbc((p1, p2, p3))
                    i, j, d, D, S = neighbour_list("ijdDS", atoms, atoms.numbers * 0.2 + 0.5)
                    c = np.bincount(i, minlength=nat)
                    atoms2 = atoms.repeat((p1 + 1, p2 + 1, p3 + 1))
                    i2, j2, d2, D2, S2 = neighbour_list("ijdDS", atoms2, atoms2.numbers * 0.2 + 0.5)
                    c2 = np.bincount(i2, minlength=nat)
                    c2.shape = (-1, nat)
                    dd = d.sum() * (p1 + 1) * (p2 + 1) * (p3 + 1) - d2.sum()
                    dr = np.linalg.solve(atoms.cell.T, (atoms.positions[1]-atoms.positions[0]).T).T+np.array([0,0,3])
                    self.assertTrue(abs(dd) < 1e-10)
                    self.assertTrue(not (c2 - c).any())

    def test_wrong_number_of_cutoffs(self):
        nat = 10
        atoms = ase.Atoms(numbers=range(nat),
                          cell=[(0.2, 1.2, 1.4),
                                (1.4, 0.1, 1.6),
                                (1.3, 2.0, -0.1)])
        atoms.set_scaled_positions(3 * np.random.random((nat, 3)) - 1)
        exception_thrown = False
        try:
            i, j, d, D, S = neighbour_list("ijdDS", atoms, np.ones(len(atoms)-1))
        except TypeError:
            exception_thrown = True
        self.assertTrue(exception_thrown)

    def test_shrink_wrapped_direct_call(self):
        a = io.read('aC.cfg')
        r = a.positions
        j, dr, i, abs_dr, shift = neighbour_list("jDidS", positions=r,
                                                 cutoff=1.85)

        self.assertTrue((np.bincount(i) == np.bincount(j)).all())

        dr_direct = r[j]-r[i]
        abs_dr_from_dr = np.sqrt(np.sum(dr*dr, axis=1))
        abs_dr_direct = np.sqrt(np.sum(dr_direct*dr_direct, axis=1))

        self.assertTrue(np.all(np.abs(abs_dr-abs_dr_from_dr) < 1e-12))
        self.assertTrue(np.all(np.abs(abs_dr-abs_dr_direct) < 1e-12))

        self.assertTrue(np.all(np.abs(dr-dr_direct) < 1e-12))


class TestTriplets(matscipytest.MatSciPyTestCase):

    def test_get_jump_indicies(self):
        first_triplets = get_jump_indicies([0, 0, 0, 1, 1, 1, 1, 1, 2, 2,
                                            2, 3, 3, 3, 3, 4, 4, 4, 4])
        first_triplets_comp = [0, 3, 8, 11, 15, 19]
        assert np.all(first_triplets == first_triplets_comp)
        first_triplets = get_jump_indicies([0])
        first_triplets_comp = [0, 1]
        # print(first_triplets, first_triplets_comp)
        assert np.all(first_triplets == first_triplets_comp)

    def test_triplet_list(self):
        ij_t_comp = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
                     9, 10, 10, 11, 11]
        ik_t_comp = [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4, 7, 8, 6, 8, 6, 7, 10,
                     11, 9, 11, 9, 10]
        i_n = [0]*2+[1]*4+[2]*4

        first_i = get_jump_indicies(i_n)
        ij_t_comp = [0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
                     7, 7, 7, 8, 8, 8, 9, 9, 9]
        ik_t_comp = [1, 0, 3, 4, 5, 2, 4, 5, 2, 3, 5, 2, 3, 4, 7, 8, 9,
                     6, 8, 9, 6, 7, 9, 6, 7, 8]

        a = triplet_list(first_i)
        assert np.alltrue(a[0] == ij_t_comp)
        assert np.alltrue(a[1] == ik_t_comp)

        first_i = np.array([0, 2, 6, 10], dtype='int32')
        a = triplet_list(first_i, [2.2]*4+[3.0]*2+[2.0]*4, 2.6)
        ij_t_comp = [0, 1, 2, 3, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
        ik_t_comp = [1, 0, 3, 2, 7, 8, 9, 6, 8, 9, 6, 7, 9, 6, 7, 8]
        assert np.all(a[0] == ij_t_comp)
        assert np.all(a[1] == ik_t_comp)

        first_i = np.array([0, 2, 6, 10], dtype='int32')
        a = triplet_list(first_i)
        ij_t_comp = [0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
                     5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
        ik_t_comp = [1, 0, 3, 4, 5, 2, 4, 5, 2, 3, 5, 2, 3,
                     4, 7, 8, 9, 6, 8, 9, 6, 7, 9, 6, 7, 8]
        assert np.all(a[0] == ij_t_comp)
        assert np.all(a[1] == ik_t_comp)

    def test_triplet_list_with_cutoff(self):
        first_i = np.array([0, 2, 6, 10], dtype='int32')
        a = triplet_list(first_i, [2.2]*9+[3.0], 2.6)
        ij_t_comp = [0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
                     5, 6, 6, 7, 7, 8, 8]
        ik_t_comp = [1, 0, 3, 4, 5, 2, 4, 5, 2, 3, 5, 2, 3,
                     4, 7, 8, 6, 8, 6, 7]
        assert np.all(a[0] == ij_t_comp)
        assert np.all(a[1] == ik_t_comp)

        first_i = np.array([0, 2, 6, 10], dtype='int32')
        a = triplet_list(first_i, [2.2]*4+[3.0]*2+[2.0]*4, 2.6)
        ij_t_comp = [0, 1, 2, 3, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
        ik_t_comp = [1, 0, 3, 2, 7, 8, 9, 6, 8, 9, 6, 7, 9, 6, 7, 8]
        assert np.all(a[0] == ij_t_comp)
        assert np.all(a[1] == ik_t_comp)


class TestNeighbourhood(matscipytest.MatSciPyTestCase):
    theta0 = np.pi / 3
    atoms = ase.Atoms("H2O",
                      positions=[[-1, 0, 0],
                                 [0, 0, 0],
                                 [np.cos(theta0), np.sin(theta0), 0]],
                      cell=ase.cell.Cell.fromcellpar([10, 10, 10, 90, 90, 90]))
    molecules = Molecules(bonds_connectivity=[[0, 1], [2, 1], [0, 2]],
                          bonds_types=[1, 2, 3],
                          angles_connectivity=[
                              [0, 1, 2],
                              [0, 2, 1],
                              [1, 0, 2],
                          ],
                          angles_types=[1, 2, 3])

    cutoff = CutoffNeighbourhood(cutoff=10.)
    molecule = MolecularNeighbourhood(molecules)

    def test_pairs(self):
        cutoff_d = self.cutoff.get_pairs(self.atoms, "ijdD")
        molecule_d = self.molecule.get_pairs(self.atoms, "ijdD")
        p = np.array([1, 0, 2, 3, 5, 4])
        mask_extra_bonds = self.molecule.connectivity["bonds"]["type"] >= 0

        # print("CUTOFF", cutoff_d)
        # print("MOLECULE", molecule_d)

        for c, m in zip(cutoff_d, molecule_d):
            # print("c =", c)
            # print("m =", m[mask_extra_bonds])
            self.assertArrayAlmostEqual(c, m[mask_extra_bonds][p], tol=1e-10)

    def test_triplets(self):
        cutoff_pairs = np.array(self.cutoff.get_pairs(self.atoms, "ij")).T
        molecules_pairs = np.array(self.molecule.get_pairs(self.atoms, "ij")).T
        cutoff_d = self.cutoff.get_triplets(self.atoms, "ijk")
        molecule_d = self.molecule.get_triplets(self.atoms, "ijk")
        p = np.array([0, 1, 3, 2, 4, 5])

        # We compare the refered pairs, not the triplet info directly
        for c, m in zip(cutoff_d, molecule_d):
            # print("c =", cutoff_pairs[:][c])
            # print("m =", molecules_pairs[:][m])
            self.assertArrayAlmostEqual(cutoff_pairs[:, 0][c],
                                        molecules_pairs[:, 0][m][p], tol=1e-10)
            self.assertArrayAlmostEqual(cutoff_pairs[:, 1][c],
                                        molecules_pairs[:, 1][m][p], tol=1e-10)

        # Testing computed distances and vectors
        cutoff_d = self.cutoff.get_triplets(self.atoms, "dD")
        molecule_d = self.cutoff.get_triplets(self.atoms, "dD")

        # TODO why no permutation?
        for c, m in zip(cutoff_d, molecule_d):
            self.assertArrayAlmostEqual(c, m, tol=1e-10)

    def test_pair_types(self):
        pass


if __name__ == '__main__':
    unittest.main()
