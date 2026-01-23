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
from ase import Atoms


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
    data["angles"]["atoms"] = data["angles"]["atoms"][:, (1, 0, 2)]
    assert np.all(data["bonds"] == mols.bonds)
    assert np.all(data["angles"] == mols.angles)
    assert np.all(data["dihedrals"] == mols.dihedrals)


def test_read_molecules_from_atoms(mols_from_atoms):
    data, mols = mols_from_atoms
    data["angles"]["atoms"] = data["angles"]["atoms"][:, (1, 0, 2)]
    assert np.all(data["bonds"] == mols.bonds)
    assert np.all(data["angles"] == mols.angles)
    assert np.all(data["dihedrals"] == mols.dihedrals)


def test_molecules_roundtrip():
    """Test that from_atoms and to_arrays are inverse operations."""
    # Create an ASE Atoms object with connectivity information
    natoms = 4
    atoms = Atoms('CHCH', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    
    # Add bonds array (atoms 0-1, 1-2, 2-3 with types 1, 2, 1)
    atoms.arrays['bonds'] = np.array([
        '1(1)',     # atom 0 bonded to atom 1 with type 1
        '0(1),2(2)', # atom 1 bonded to atoms 0 and 2 with types 1 and 2
        '1(2),3(1)', # atom 2 bonded to atoms 1 and 3 with types 2 and 1
        '2(1)'      # atom 3 bonded to atom 2 with type 1
    ])
    
    # Add angles array (angle 0-1-2 with type 1, angle 1-2-3 with type 2)
    atoms.arrays['angles'] = np.array([
        '_',        # atom 0 has no angles
        '0-2(1)',   # atom 1 has angle involving atoms 0-2 with type 1
        '1-3(2)',   # atom 2 has angle involving atoms 1-3 with type 2
        '_'         # atom 3 has no angles
    ])
    
    # Add dihedrals array (dihedral 0-1-2-3 with type 1)
    atoms.arrays['dihedrals'] = np.array([
        '_',          # atom 0 has no dihedrals
        '0-2-3(1)',   # atom 1 has dihedral involving atoms 0-2-3 with type 1
        '_',          # atom 2 has no dihedrals
        '_'           # atom 3 has no dihedrals
    ])
    
    # Convert to Molecules object
    molecules = Molecules.from_atoms(atoms)
    
    # Convert back to arrays format
    arrays = molecules.to_arrays(natoms)
    
    # Verify the roundtrip preserves the original data
    assert 'bonds' in arrays
    assert 'angles' in arrays
    assert 'dihedrals' in arrays
    
    # For bonds, the order might be different due to the way connectivity is stored
    # So we need to compare sets of bond strings for each atom
    for i in range(natoms):
        original_bonds = set(atoms.arrays['bonds'][i].split(',')) if atoms.arrays['bonds'][i] != '_' else {'_'}
        roundtrip_bonds = set(arrays['bonds'][i].split(',')) if arrays['bonds'][i] != '_' else {'_'}
        assert original_bonds == roundtrip_bonds, f"Bond mismatch for atom {i}: {original_bonds} != {roundtrip_bonds}"
    
    # Angles and dihedrals should match exactly since they're stored on specific atoms
    for i in range(natoms):
        assert atoms.arrays['angles'][i] == arrays['angles'][i], f"Angle mismatch for atom {i}"
        assert atoms.arrays['dihedrals'][i] == arrays['dihedrals'][i], f"Dihedral mismatch for atom {i}"


def test_molecules_roundtrip_empty():
    """Test roundtrip with atoms that have no connectivity."""
    natoms = 2
    atoms = Atoms('HH', positions=[[0, 0, 0], [1, 0, 0]])
    
    # Add empty connectivity arrays
    atoms.arrays['bonds'] = np.array(['_', '_'])
    atoms.arrays['angles'] = np.array(['_', '_'])
    atoms.arrays['dihedrals'] = np.array(['_', '_'])
    
    # Convert to Molecules object and back
    molecules = Molecules.from_atoms(atoms)
    arrays = molecules.to_arrays(natoms)
    
    # When all connectivity is empty ('_'), the to_arrays method returns empty dict
    # because there are no actual bonds/angles/dihedrals to store
    # This is expected behavior - the Molecules object will have empty arrays
    assert len(molecules.bonds) == 0
    assert len(molecules.angles) == 0  
    assert len(molecules.dihedrals) == 0
    
    # The returned arrays dict should be empty since there's no connectivity
    assert len(arrays) == 0


def test_molecules_roundtrip_bonds_only():
    """Test roundtrip with only bonds (no angles or dihedrals)."""
    natoms = 3
    atoms = Atoms('CHH', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    
    # Add only bonds array
    atoms.arrays['bonds'] = np.array([
        '1(1),2(2)',  # atom 0 bonded to atoms 1 and 2
        '0(1)',       # atom 1 bonded to atom 0
        '0(2)'        # atom 2 bonded to atom 0
    ])
    
    # Convert to Molecules object and back
    molecules = Molecules.from_atoms(atoms)
    arrays = molecules.to_arrays(natoms)
    
    # Should only get bonds back
    assert 'bonds' in arrays
    assert 'angles' not in arrays
    assert 'dihedrals' not in arrays
    
    # Verify bonds
    for i in range(natoms):
        original_bonds = set(atoms.arrays['bonds'][i].split(',')) if atoms.arrays['bonds'][i] != '_' else {'_'}
        roundtrip_bonds = set(arrays['bonds'][i].split(',')) if arrays['bonds'][i] != '_' else {'_'}
        assert original_bonds == roundtrip_bonds, f"Bond mismatch for atom {i}: {original_bonds} != {roundtrip_bonds}"


if __name__ == '__main__':
    unittest.main()
