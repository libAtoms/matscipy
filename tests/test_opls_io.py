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

import os

import ase
import ase.calculators.lammpsrun
import ase.io
import numpy as np
import pytest

import matscipy.io.opls
import matscipy.opls


def assert_dictionaries_equal(d1, d2, ignore_case=True):
    """Helper function to compare dictionaries with array values."""

    def lower_if_ignore_case(k):
        if ignore_case:
            return k.lower()
        else:
            return k

    d1_normalized = dict([(lower_if_ignore_case(k), v) for (k, v) in d1.items()])
    d2_normalized = dict([(lower_if_ignore_case(k), v) for (k, v) in d2.items()])

    assert sorted(d1_normalized.keys()) == sorted(d2_normalized.keys()), (
        f"Dictionaries differ: d1.keys() ({d1_normalized.keys()}) != d2.keys() "
        f"({d2_normalized.keys()})"
    )

    for key in d1_normalized:
        v1, v2 = d1_normalized[key], d2_normalized[key]
        if isinstance(v1, (list, np.ndarray)):
            np.testing.assert_allclose(v1, v2, rtol=1e-7, atol=1e-7)
        else:
            assert v1 == v2, f"Dictionaries differ: key={key} value1={v1} value2={v2}"


def test_read_extended_xyz(datafile_directory):
    file_path = os.path.join(datafile_directory, "opls_extxyz.xyz")
    struct = matscipy.io.opls.read_extended_xyz(file_path)

    assert isinstance(struct, matscipy.opls.OPLSStructure)
    assert list(struct.numbers) == [1, 1]
    np.testing.assert_allclose(
        struct.positions, [[4.5, 5.0, 5.0], [5.5, 5.0, 5.0]], atol=0.01, rtol=0
    )
    assert list(struct.get_array("molid")) == [1, 1]
    assert list(struct.types) == ["H1"]
    assert list(struct.get_array("type")) == ["H1", "H1"]
    assert list(struct.get_array("tags")) == [0, 0]


def test_read_block(datafile_directory):
    file_path = os.path.join(datafile_directory, "opls_parameters.in")
    data = matscipy.io.opls.read_block(file_path, "Dihedrals")
    assert_dictionaries_equal(
        data, {"H1-C1-C1-H1": [0.00, 0.00, 0.01, 0.00]}, ignore_case=False
    )

    with pytest.raises(RuntimeError):
        matscipy.io.opls.read_block(file_path, "Charges")


def test_read_cutoffs(datafile_directory):
    file_path = os.path.join(datafile_directory, "opls_cutoffs.in")
    cutoffs = matscipy.io.opls.read_cutoffs(file_path)

    assert isinstance(cutoffs, matscipy.opls.CutoffList)
    assert_dictionaries_equal(
        cutoffs.nvh, {"C1-C1": 1.85, "C1-H1": 1.15}, ignore_case=False
    )


def test_read_parameter_file(datafile_directory):
    file_path = os.path.join(datafile_directory, "opls_parameters.in")
    cutoffs, ljq, bonds, angles, dihedrals = matscipy.io.opls.read_parameter_file(
        file_path
    )

    assert isinstance(cutoffs, matscipy.opls.CutoffList)
    assert_dictionaries_equal(
        cutoffs.nvh, {"C1-C1": 1.85, "C1-H1": 1.15, "H1-H1": 0.0}, ignore_case=False
    )

    assert isinstance(ljq, dict)
    assert_dictionaries_equal(
        ljq,
        {"C1": [0.001, 3.500, -0.010], "H1": [0.001, 2.500, 0.010]},
        ignore_case=False,
    )
    assert isinstance(ljq.lj_cutoff, float)
    assert isinstance(ljq.c_cutoff, float)
    assert ljq.lj_cutoff == pytest.approx(12.0, abs=0.01)
    assert ljq.c_cutoff == pytest.approx(15.0, abs=0.01)
    assert isinstance(ljq.lj_pairs, dict)
    assert_dictionaries_equal(
        ljq.lj_pairs, {"C1-H1": [0.001, 3.4, 11.0]}, ignore_case=False
    )

    assert isinstance(bonds, matscipy.opls.BondData)
    assert_dictionaries_equal(
        bonds.nvh, {"C1-C1": [10.0, 1.0], "C1-H1": [10.0, 1.0]}, ignore_case=False
    )

    assert isinstance(angles, matscipy.opls.AnglesData)
    assert_dictionaries_equal(
        angles.nvh,
        {"H1-C1-C1": [1.0, 100.0], "H1-C1-H1": [1.0, 100.0]},
        ignore_case=False,
    )

    assert isinstance(dihedrals, matscipy.opls.DihedralsData)
    assert_dictionaries_equal(
        dihedrals.nvh, {"H1-C1-C1-H1": [0.00, 0.00, 0.01, 0.00]}, ignore_case=False
    )


def test_read_lammps_data(datafile_directory):
    atoms_path = os.path.join(datafile_directory, "opls_test.atoms")
    params_path = os.path.join(datafile_directory, "opls_test.parameters")
    test_structure = matscipy.io.opls.read_lammps_data(atoms_path, params_path)

    np.testing.assert_allclose(
        test_structure.cell,
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        atol=0.01,
        rtol=0,
    )
    assert len(test_structure) == 4
    assert list(test_structure.numbers) == [1, 6, 6, 1]
    np.testing.assert_allclose(
        test_structure.positions,
        [[3.5, 5.0, 5.0], [4.5, 5.0, 5.0], [5.5, 5.0, 5.0], [6.5, 5.0, 5.0]],
        atol=0.01,
        rtol=0,
    )
    test_velocities = ase.calculators.lammpsrun.convert(
        test_structure.get_velocities(), "velocity", "ASE", "metal"
    )
    np.testing.assert_allclose(
        test_velocities,
        [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0]],
        atol=0.01,
        rtol=0,
    )
    np.testing.assert_allclose(
        test_structure.get_masses(), [1.008, 12.011, 12.011, 1.008], atol=0.0001, rtol=0
    )
    np.testing.assert_allclose(
        test_structure.get_charges(), [0.01, -0.01, -0.01, 0.01], atol=0.001, rtol=0
    )
    assert list(test_structure.get_array("molid")) == [1, 1, 1, 1]
    assert len(test_structure.get_types()) == 2
    assert "C1" in test_structure.get_types()
    assert "H1" in test_structure.get_types()

    assert len(test_structure.bond_types) == 2
    assert "C1-C1" in test_structure.bond_types
    assert "C1-H1" in test_structure.bond_types or "H1-C1" in test_structure.bond_types
    assert test_structure.bond_list.shape == (3, 3)
    assert (test_structure.bond_list[0] == [0, 0, 1]).all() or (
        test_structure.bond_list[0] == [0, 1, 0]
    ).all()
    assert (test_structure.bond_list[1] == [1, 1, 2]).all() or (
        test_structure.bond_list[1] == [1, 2, 1]
    ).all()
    assert (test_structure.bond_list[2] == [0, 2, 3]).all() or (
        test_structure.bond_list[2] == [0, 3, 2]
    ).all()

    assert len(test_structure.ang_types) == 1
    assert (
        "H1-C1-C1" in test_structure.ang_types or "C1-C1-H1" in test_structure.ang_types
    )
    assert test_structure.ang_list.shape == (2, 4)
    assert (test_structure.ang_list[0] == [0, 0, 1, 2]).all() or (
        test_structure.ang_list[0] == [0, 2, 1, 0]
    ).all()
    assert (test_structure.ang_list[1] == [0, 1, 2, 3]).all() or (
        test_structure.ang_list[1] == [0, 3, 2, 1]
    ).all()

    assert len(test_structure.dih_types) == 1
    assert test_structure.dih_types == ["H1-C1-C1-H1"]
    assert test_structure.dih_list.shape == (1, 5)
    assert (test_structure.dih_list[0] == [0, 0, 1, 2, 3]).all() or (
        test_structure.dih_list[0] == [0, 3, 2, 1, 0]
    ).all()


def test_write_lammps_atoms(datafile_directory):
    c2h2 = ase.Atoms("HC2H", cell=[10.0, 10.0, 10.0])
    c2h2.set_positions(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )

    opls_c2h2 = matscipy.opls.OPLSStructure(c2h2)
    opls_c2h2.set_types(["H1", "C1", "C1", "H1"])

    params_path = os.path.join(datafile_directory, "opls_parameters.in")
    cutoffs, ljq, bonds, angles, dihedrals = matscipy.io.opls.read_parameter_file(
        params_path
    )

    opls_c2h2.set_cutoffs(cutoffs)
    opls_c2h2.set_atom_data(ljq)
    opls_c2h2.get_bonds(bonds)
    opls_c2h2.get_angles(angles)
    opls_c2h2.get_dihedrals(dihedrals)

    matscipy.io.opls.write_lammps_atoms("temp", opls_c2h2)
    matscipy.io.opls.write_lammps_definitions("temp", opls_c2h2)

    # Read written structure
    c2h2_written = matscipy.io.opls.read_lammps_data("temp.atoms", "temp.opls")

    np.testing.assert_allclose(
        c2h2_written.cell,
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        atol=0.01,
        rtol=0,
    )
    assert len(c2h2_written) == 4
    assert list(c2h2_written.numbers) == [1, 6, 6, 1]
    np.testing.assert_allclose(
        c2h2_written.positions,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        atol=0.01,
        rtol=0,
    )
    np.testing.assert_allclose(
        c2h2_written.get_velocities(), np.zeros([4, 3], dtype=float), atol=0.01, rtol=0
    )
    np.testing.assert_allclose(
        c2h2_written.get_masses(), [1.008, 12.011, 12.011, 1.008], atol=0.0001, rtol=0
    )
    np.testing.assert_allclose(
        c2h2_written.get_charges(), [0.01, -0.01, -0.01, 0.01], atol=0.001, rtol=0
    )
    assert list(c2h2_written.get_array("molid")) == [1, 1, 1, 1]
    assert len(c2h2_written.get_types()) == 2
    assert "C1" in c2h2_written.get_types()
    assert "H1" in c2h2_written.get_types()

    assert len(c2h2_written.bond_types) == 2
    assert "C1-C1" in c2h2_written.bond_types
    assert "C1-H1" in c2h2_written.bond_types or "H1-C1" in c2h2_written.bond_types
    assert c2h2_written.bond_list.shape == (3, 3)
    bonds_list = c2h2_written.bond_list[:, 1:].tolist()
    assert [0, 1] in bonds_list or [1, 0] in bonds_list
    assert [1, 2] in bonds_list or [2, 1] in bonds_list
    assert [2, 3] in bonds_list or [3, 2] in bonds_list

    assert len(c2h2_written.ang_types) == 1
    assert "H1-C1-C1" in c2h2_written.ang_types or "C1-C1-H1" in c2h2_written.ang_types
    assert c2h2_written.ang_list.shape == (2, 4)
    angles_list = c2h2_written.ang_list[:, 1:].tolist()
    assert [0, 1, 2] in angles_list or [2, 1, 0] in angles_list
    assert [1, 2, 3] in angles_list or [3, 2, 1] in angles_list

    assert len(c2h2_written.dih_types) == 1
    assert c2h2_written.dih_types == ["H1-C1-C1-H1"]
    assert c2h2_written.dih_list.shape == (1, 5)
    assert (c2h2_written.dih_list[0] == [0, 0, 1, 2, 3]).all() or (
        c2h2_written.dih_list[0] == [0, 3, 2, 1, 0]
    ).all()


def test_write_lammps_definitions(datafile_directory):
    c2h2 = ase.Atoms("HC2H", cell=[10.0, 10.0, 10.0])
    c2h2.set_positions(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )

    opls_c2h2 = matscipy.opls.OPLSStructure(c2h2)
    opls_c2h2.set_types(["H1", "C1", "C1", "H1"])

    params_path = os.path.join(datafile_directory, "opls_parameters.in")
    cutoffs, ljq, bonds, angles, dihedrals = matscipy.io.opls.read_parameter_file(
        params_path
    )

    opls_c2h2.set_cutoffs(cutoffs)
    opls_c2h2.set_atom_data(ljq)
    opls_c2h2.get_bonds(bonds)
    opls_c2h2.get_angles(angles)
    opls_c2h2.get_dihedrals(dihedrals)

    matscipy.io.opls.write_lammps_definitions("temp", opls_c2h2)

    # Read written parameters
    pair_coeff = []
    bond_coeff = []
    angle_coeff = []
    dihedral_coeff = []
    charges = []

    with open("temp.opls", "r") as fileobj:
        for line in fileobj.readlines():
            if line.startswith("pair_style"):
                lj_cutoff = line.split()[2]
                q_cutoff = line.split()[3]
            elif line.startswith("pair_coeff"):
                pair_coeff.append(line.split())
            elif line.startswith("bond_coeff"):
                bond_coeff.append(line.split())
            elif line.startswith("angle_coeff"):
                angle_coeff.append(line.split())
            elif line.startswith("dihedral_coeff"):
                dihedral_coeff.append(line.split())
            elif len(line.split()) > 3:
                if line.split()[3] == "charge":
                    charges.append(line.split())

    assert len(charges) == 2
    for charge in charges:
        if charge[6] == "C1":
            assert float(charge[4]) == pytest.approx(-0.01, abs=0.01)
        elif charge[6] == "H1":
            assert float(charge[4]) == pytest.approx(0.01, abs=0.01)

    assert float(lj_cutoff) == pytest.approx(12.0, abs=0.1)
    assert float(q_cutoff) == pytest.approx(15.0, abs=0.1)
    assert len(pair_coeff) == 3
    for pair in pair_coeff:
        if pair[6] == "C1":
            assert float(pair[3]) == pytest.approx(0.001, abs=0.001)
            assert float(pair[4]) == pytest.approx(3.5, abs=0.1)
        elif pair[6] == "H1":
            assert float(pair[3]) == pytest.approx(0.001, abs=0.001)
            assert float(pair[4]) == pytest.approx(2.5, abs=0.1)
        elif pair[7] == "H1-C1" or pair[7] == "C1-H1":
            assert float(pair[3]) == pytest.approx(0.001, abs=0.001)
            assert float(pair[4]) == pytest.approx(3.4, abs=0.1)
            assert float(pair[5]) == pytest.approx(11.0, abs=0.1)

    assert len(bond_coeff) == 2
    for bond in bond_coeff:
        if bond[5] == "C1-C1":
            assert float(bond[2]) == pytest.approx(10.0, abs=0.1)
            assert float(bond[3]) == pytest.approx(1.0, abs=0.1)
        elif bond[5] == "H1-C1" or bond[5] == "C1-H1":
            assert float(bond[2]) == pytest.approx(10.0, abs=0.1)
            assert float(bond[3]) == pytest.approx(1.0, abs=0.1)

    assert len(angle_coeff) == 1
    assert float(angle_coeff[0][2]) == pytest.approx(1.0, abs=0.1)
    assert float(angle_coeff[0][3]) == pytest.approx(100.0, abs=0.1)

    assert len(dihedral_coeff) == 1
    assert float(dihedral_coeff[0][2]) == pytest.approx(0.0, abs=0.1)
    assert float(dihedral_coeff[0][3]) == pytest.approx(0.0, abs=0.1)
    assert float(dihedral_coeff[0][4]) == pytest.approx(0.01, abs=0.01)
    assert float(dihedral_coeff[0][5]) == pytest.approx(0.0, abs=0.1)
