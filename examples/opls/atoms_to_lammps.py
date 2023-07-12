#!/usr/bin/env python3
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


# This script demonstrates how to create an atomic configuration in a
# python script and generate input files for LAMMPS from it.
# Alternatively, atomic configurations can be read from an extended xyz
# file. See example 'extxyz_to_lammps.py'.


import numpy as np
import ase
import matscipy.opls
import matscipy.io.opls


# Simple example: ethane molecules
# matscipy.opls.OPLSStructure is a subclass of ase.Atoms. OPLSStructure
# objects can therefore be constructed and manipulated in the same way
# as ase.Atoms objects
a1 = matscipy.opls.OPLSStructure(
    'C2H6',
    positions = [[1.,  0.,  0.],
                 [2.,  0.,  0.],
                 [0.,  0.,  0.],
                 [1.,  1.,  0.],
                 [1., -1.,  0.],
                 [2.,  0.,  1.],
                 [2.,  0., -1.],
                 [3.,  0.,  0.]],
    cell = [10., 10., 10.]
    )
a1.translate([0., 3., 0.])

# Alternative: construct an ase.Atoms object and convert it to a
# matscipy.opls.OPLSStructure object.
a2 = ase.Atoms(
    'C2H6',
    positions = [[1.,  0.,  0.],
                 [2.,  0.,  0.],
                 [0.,  0.,  0.],
                 [1.,  1.,  0.],
                 [1., -1.,  0.],
                 [2.,  0.,  1.],
                 [2.,  0., -1.],
                 [3.,  0.,  0.]],
    cell = [10., 10., 10.]
    )
a2 = matscipy.opls.OPLSStructure(a2)


a = matscipy.opls.OPLSStructure(cell=[10., 10., 10.])
a.extend(a1)
a.extend(a2)
a.center()

# Specify atomic types. Notice the difference between type and element.
# In this example we are using two different types of hydrogen atoms.
a.set_types(['C1', 'C1', 'H1', 'H1', 'H1', 'H2', 'H2', 'H2',
             'C1', 'C1', 'H1', 'H1', 'H1', 'H2', 'H2', 'H2'])

# To perform a non-reactive simulation, all types of pair, angle and
# dihedral interactions must be specified manually. Usually this means
# searching the literature for suitable parameters, which can be a
# tedious task. If it is known which interactions are present in a
# system, this can be much easier. Lists of all existing interactions
# can be generated based on the distance of the atoms from each other.
# The maximum distances up to which two atoms are considered to interact
# can be read from a file.
cutoffs = matscipy.io.opls.read_cutoffs('cutoffs.in')
a.set_cutoffs(cutoffs)

bond_types, _ = a.get_bonds()
print('Pairwise interactions:')
print(bond_types)

angle_types, _ = a.get_angles()
print('\nAngular interactions:')
print(angle_types)

dih_types, _ = a.get_dihedrals()
print('\nDihedral interactions:')
print(dih_types)



# Once the parameters of all interactions are known, they can be written
# to a file. This can be used to generate the lists of all interactions
# and to create input files for LAMMPS.
cutoffs, atom_data, bond_data, angle_data, dih_data = matscipy.io.opls.read_parameter_file('parameters.in')

a.set_cutoffs(cutoffs)
a.set_atom_data(atom_data)

a.get_bonds(bond_data)
a.get_angles(angle_data)
a.get_dihedrals(dih_data)

# Write the atomic structure, the potential definitions and a sample
# input script for LAMMPS (3 files in total). The input script contains
# a simple relaxation of the atomic position. Modify this file for more
# complex simulations.
matscipy.io.opls.write_lammps('example', a)

