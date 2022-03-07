#
# Copyright 2014-2016, 2021 Lars Pastewka (U. Freiburg)
#           2014 James Kermode (Warwick U.)
#           2022 Lucas Frérot (U. Freiburg)
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
#                  Adrien Gola, Karlsruhe Institute of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

import unittest

import numpy as np

from matscipy.io import loadtbl, savetbl
from matscipy.io.lammps_data import LAMMPSData, read_molecules_from_lammps_data
from os import remove

import matscipytest


class TestEAMIO(matscipytest.MatSciPyTestCase):
    def test_savetbl_loadtbl(self):
        n = 123
        a = np.random.random(n)
        b = np.random.random(n)
        poe = np.random.random(n)
        savetbl('test.out', a=a, b=b, poe=poe)

        data = loadtbl('test.out')
        self.assertArrayAlmostEqual(a, data['a'])
        self.assertArrayAlmostEqual(b, data['b'])
        self.assertArrayAlmostEqual(poe, data['poe'])


class TestLAMMPSData(matscipytest.MatSciPyTestCase):
    def test_read_write_lammps_data(self):
        filename = "lammps_text.data"

        data = LAMMPSData(style='full')
        data['atoms'] = [
            [0, 0, 0],
            [0, 0, 1],
            [1.1, 2, 1.1]
        ]
        data['velocities'] = [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]

        data['atom types'] = [1, 1, 2]
        data['atoms']['charge'] = [1, -1, 1]
        data['atoms']['mol'] = 1
        data['masses'] = [2, 3]

        data['bonds'] = [
            [1, 3]
        ]

        data['bond types'] = [1]

        data['angles'] = [
            [1, 2, 3]
        ]
        data['angle types'] = [1]
        data.ranges = [[-1, 1], [-1, 1], [-1, 1]]
        data.write(filename)

        read_data = LAMMPSData(style='full')
        read_data.read(filename)

        assert np.all(np.array(data.ranges) == np.array(read_data.ranges))
        assert np.all(data['atoms'] == read_data['atoms'])
        assert np.all(data['bonds'] == read_data['bonds'])
        assert np.all(data['angles'] == read_data['angles'])
        assert np.all(data['masses'] == read_data['masses'])
        assert np.all(data['velocities'] == read_data['velocities'])

        mols = read_molecules_from_lammps_data(filename)

        # Correct for type offset
        for label in ["bonds", "angles", "dihedrals"]:
            data[label]["atoms"] -= 1

        assert np.all(data["bonds"] == mols.bonds)
        assert np.all(data["angles"] == mols.angles)
        assert np.all(data["dihedrals"] == mols.dihedrals)

        try:
            remove(filename)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    unittest.main()
