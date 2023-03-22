#
# Copyright 2020-2021 Lars Pastewka (U. Freiburg)
#           2020 Jan Griesser (U. Freiburg)
#           2019-2020 Wolfram G. Nöhring (U. Freiburg)
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

import gzip
import numpy as np

import ase.io as io

import matscipytest
from matscipy.calculators.eam import EAM

from ase.phonons import Phonons

from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from scipy.sparse import bsr_matrix

###

class TestEAMForcesHessian(matscipytest.MatSciPyTestCase):

    # In the force calculation
    force_tolerance = 1e-3
    hessian_tolerance = 1e-6

    def test_forces_CuZr_glass(self):
        """Calculate interatomic forces in CuZr glass

        Reference: tabulated forces from a calculation 
        with Lammmps (git version patch_29Mar2019-2-g585403d65)

        The forces can be re-calculated using the following
        Lammps commands:
            units metal
            atom_style atomic
            boundary p p p
            read_data CuZr_glass_460_atoms.lammps.data.gz
            pair_style eam/alloy
            pair_coeff * * ZrCu.onecolumn.eam.alloy Zr Cu
            # The initial configuration is in equilibrium
            # and the remaining forces are small
            # Swap atom types to bring system out of
            # equilibrium and create nonzero forces
            group originally_Zr type 1
            group originally_Cu type 2
            set group originally_Zr type 2
            set group originally_Cu type 1
            run 0
            write_dump all custom &
                CuZr_glass_460_atoms_forces.lammps.dump.gz &
                id type x y z fx fy fz &
                modify sort id format float "%.14g"
        """
        format = "lammps-dump" if "lammps-dump" in io.formats.all_formats.keys() else "lammps-dump-text"
        atoms = io.read("CuZr_glass_460_atoms_forces.lammps.dump.gz", format=format)
        old_atomic_numbers = atoms.get_atomic_numbers()
        sel, = np.where(old_atomic_numbers == 1)
        new_atomic_numbers = np.zeros_like(old_atomic_numbers)
        new_atomic_numbers[sel] = 40 # Zr
        sel, = np.where(old_atomic_numbers == 2)
        new_atomic_numbers[sel] = 29 # Cu
        atoms.set_atomic_numbers(new_atomic_numbers)
        calculator = EAM('ZrCu.onecolumn.eam.alloy')
        atoms.set_calculator(calculator)
        atoms.pbc = [True, True, True]
        forces = atoms.get_forces()
        # Read tabulated forces and compare
        with gzip.open("CuZr_glass_460_atoms_forces.lammps.dump.gz") as file:
            for line in file:
                if line.startswith(b"ITEM: ATOMS "): # ignore header
                    break
            dump = np.loadtxt(file)
        forces_dump = dump[:, 5:8]
        self.assertArrayAlmostEqual(forces, forces_dump, tol=self.force_tolerance) 

    def test_hessian_monoatomic(self):
        """Calculate Hessian matrix of pure Cu

        Reference: finite difference approximation of 
        Hessian from ASE
        """
        def _test_for_size(size):
            atoms = FaceCenteredCubic('Cu', size=size)
            calculator = EAM('CuAg.eam.alloy')
            self._test_hessian(atoms, calculator)
        _test_for_size(size=[1, 1, 1])
        _test_for_size(size=[2, 2, 2])
        _test_for_size(size=[1, 4, 4])
        _test_for_size(size=[4, 1, 4])
        _test_for_size(size=[4, 4, 1])
        _test_for_size(size=[4, 4, 4])

    def test_hessian_monoatomic_with_duplicate_pairs(self):
        """Calculate Hessian matrix of pure Cu

        In a small system, the same pair (i,j) will
        appear multiple times in the neighbor list,
        with different pair distance.

        Reference: finite difference approximation of 
        Hessian from ASE
        """
        atoms = FaceCenteredCubic('Cu', size=[2, 2, 2])
        calculator = EAM('CuAg.eam.alloy')
        self._test_hessian(atoms, calculator)

    def test_hessian_crystalline_alloy(self):
        """Calculate Hessian matrix of crystalline alloy

        Reference: finite difference approximation of 
        Hessian from ASE
        """
        calculator = EAM('ZrCu.onecolumn.eam.alloy')
        lattice_size = [4, 4, 4]
        # The lattice parameters are not correct, but that should be irrelevant
        # CuZr3
        atoms = L1_2(['Cu', 'Zr'], size=lattice_size, latticeconstant=4.0)
        self._test_hessian(atoms, calculator)
        # Cu3Zr
        atoms = L1_2(['Zr', 'Cu'], size=lattice_size, latticeconstant=4.0)
        self._test_hessian(atoms, calculator)
        # CuZr
        atoms = B2(['Zr', 'Cu'], size=lattice_size, latticeconstant=3.3)
        self._test_hessian(atoms, calculator)

    def test_hessian_amorphous_alloy(self):
        """Calculate Hessian matrix of amorphous alloy

        Reference: finite difference approximation of 
        Hessian from ASE
        """
        atoms = io.read('CuZr_glass_460_atoms.gz')
        atoms.pbc = [True, True, True]
        calculator = EAM('ZrCu.onecolumn.eam.alloy')
        self._test_hessian(atoms, calculator)

    def test_dynamical_matrix(self):
        """Test dynamical matrix construction

        To obtain the dynamical matrix, one could either divide by
        masses immediately when constructing the matrix, or one could
        first form the complete Hessian and then divide by masses.
        The former method is implemented.
        """
        atoms = io.read('CuZr_glass_460_atoms.gz')
        atoms.pbc = [True, True, True]
        calculator = EAM('ZrCu.onecolumn.eam.alloy')
        dynamical_matrix = calculator.calculate_hessian_matrix(
            atoms, divide_by_masses=True
        )
        # The second method requires a copy of Hessian, since
        # sparse matrix does not properly support *= operator
        hessian = calculator.calculate_hessian_matrix(atoms)
        masses = atoms.get_masses()
        mass_row = np.repeat(masses, np.diff(hessian.indptr))
        mass_col = masses[hessian.indices]
        inverse_mass = np.sqrt(mass_row * mass_col)**-1.0
        blocks = (inverse_mass * np.ones((inverse_mass.size, 3, 3), dtype=inverse_mass.dtype).T).T
        nat = len(atoms)
        dynamical_matrix_ref = hessian.multiply(
            bsr_matrix((blocks, hessian.indices, hessian.indptr), shape=(3*nat, 3*nat))
        )
        dynamical_matrix = dynamical_matrix.todense()
        dynamical_matrix_ref = dynamical_matrix_ref.todense()
        self.assertArrayAlmostEqual(
            dynamical_matrix, dynamical_matrix.T, tol=self.hessian_tolerance
        ) 
        self.assertArrayAlmostEqual(
            dynamical_matrix_ref, dynamical_matrix_ref.T, tol=self.hessian_tolerance
        ) 
        self.assertArrayAlmostEqual(
            dynamical_matrix, dynamical_matrix_ref, tol=self.hessian_tolerance
        ) 

    def _test_hessian(self, atoms, calculator):
        H_analytical = calculator.calculate_hessian_matrix(atoms)
        H_analytical = H_analytical.todense()
        # Hessian is symmetric:
        self.assertArrayAlmostEqual(H_analytical, H_analytical.T, tol=self.hessian_tolerance) 
        H_numerical = self._calculate_finite_difference_hessian(atoms, calculator)
        self.assertArrayAlmostEqual(H_numerical, H_numerical.T, tol=self.hessian_tolerance) 
        self.assertArrayAlmostEqual(H_analytical, H_numerical, tol=self.hessian_tolerance) 

    def _calculate_finite_difference_hessian(self, atoms, calculator):
        """Calcualte the Hessian matrix using finite differences."""
        ph = Phonons(atoms, calculator, supercell=(1, 1, 1), delta=1e-6)
        ph.clean()
        ph.run()
        ph.read(acoustic=False)
        ph.clean()
        H_numerical = ph.get_force_constant()[0, :, :]
        return H_numerical

if __name__ == '__main__':
    unittest.main()
