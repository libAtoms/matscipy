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

from ase.io import read
from matscipy.io import loadtbl, savetbl
from matscipy.io.lammpsdata import LAMMPSData, read_molecules_from_lammps_data
from matscipy.molecules import Molecules

import pytest
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

    def test_savetbl_loadtbl_text(self):
        n = 12
        a = np.random.random(n)
        b = np.random.random(n)
        t = ['a'*(i+1) for i in range(n)]
        savetbl('test2.out', a=a, b=b, t=t)

        a2, t2, b2 = loadtbl('test2.out', usecols=['a', 't', 'b'], types={'t': np.str_})
        self.assertArrayAlmostEqual(a, a2)
        self.assertArrayAlmostEqual(b, b2)
        assert (t == t2).all()


###

@pytest.fixture
def lammps_data(tmp_path):
    filename = tmp_path / "lammps_text.data"

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
        [1, 3],
        [2, 3],
    ]

    data['bond types'] = [1, 2]

    data['angles'] = [
        [1, 2, 3],
        [2, 3, 1],
    ]
    data['angle types'] = [1, 2]
    data.ranges = [[-1, 1], [-1, 1], [-1, 1]]
    data.write(filename)

    return data, filename


def test_read_write_lammps_data(lammps_data):
    data, filename = lammps_data
    read_data = LAMMPSData(style='full')
    read_data.read(filename)

    assert np.all(np.array(data.ranges) == np.array(read_data.ranges))
    assert np.all(data['atoms'] == read_data['atoms'])
    assert np.all(data['bonds'] == read_data['bonds'])
    assert np.all(data['angles'] == read_data['angles'])
    assert np.all(data['masses'] == read_data['masses'])
    assert np.all(data['velocities'] == read_data['velocities'])


@pytest.fixture
def mols_from_lammps_data(lammps_data):
    # Correct for type offset
    for label in ["bonds", "angles", "dihedrals"]:
        lammps_data[0][label]["atoms"] -= 1

    return lammps_data[0], read_molecules_from_lammps_data(lammps_data[1])


@pytest.fixture
def mols_from_atoms(lammps_data):
    data, filename = lammps_data
    atoms = read(filename, format='lammps-data', sort_by_id=True,
                 units='metal', style='full')

    # Correct for type offset
    for label in ["bonds", "angles", "dihedrals"]:
        data[label]["atoms"] -= 1

    return data, Molecules.from_atoms(atoms)


def test_read_molecules_from_lammps_data(mols_from_lammps_data):
    data, mols = mols_from_lammps_data
    assert np.all(data["bonds"] == mols.bonds)
    assert np.all(data["angles"] == mols.angles)
    assert np.all(data["dihedrals"] == mols.dihedrals)


def test_read_molecules_from_atoms(mols_from_atoms):
    data, mols = mols_from_atoms
    assert np.all(data["bonds"] == mols.bonds)
    assert np.all(data["angles"] == mols.angles)
    assert np.all(data["dihedrals"] == mols.dihedrals)


if __name__ == '__main__':
    unittest.main()
