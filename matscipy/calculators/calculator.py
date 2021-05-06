# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014-2017) James Kermode, Warwick University
#                       Lars Pastewka, Karlsruhe Institute of Technology
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
import numpy as np
from ase.calculators.calculator import Calculator

from ..elasticity import Voigt_6_to_full_3x3_stress


class MatscipyCalculator(Calculator):
    def get_hessian(self, atoms, format='dense', limits=None, divide_by_masses=False):
        """
        Calculate the Hessian matrix for a pair potential. For an atomic
        configuration with N atoms in d dimensions the hessian matrix is a
        symmetric, hermitian matrix with a shape of (d*N,d*N). The matrix is
        in general a sparse matrix, which consists of dense blocks of shape
        (d,d), which are the mixed second derivatives.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.
        format: str, optional
            Output format of the Hessian matrix, either 'dense', 'sparse' or
            'neighbour-list'. The format 'sparse' returns a sparse matrix
            representations of scipy. The format 'neighbor-list' returns
            a representation within matscipy's and ASE's neighbor list
            format, i.e. the Hessian is returned per neighbor.
            (Default: 'dense')
        limits: list [atomID_low, atomID_up], optional
            Calculate the Hessian matrix only for the given atom IDs.
            If limits=[5,10] the Hessian matrix is computed for atom IDs 5,6,7,8,9 only.
            The Hessian matrix will have the full shape dim(3*N,3*N) where N is the number of atoms.
            This ensures correct indexing of the data.
        divide_by_masses : bool, optional
            Divided each block entry n the Hessian matrix by sqrt(m_i m_j)
            where m_i and m_j are the masses of the two atoms for the Hessian
            matrix.

        Returns
        -------
        If format=='sparse':
        hessian : scipy.sparse.bsr_matrix
            Hessian matrix in sparse matrix representation
        If format=='neighbor-list'
        hessian_ncc : np.ndarray
            Array containing the Hessian blocks per atom pair
        distances_nc : np.ndarray
            Distance vectors between atom pairs
        """
        raise NotImplementedError

    def get_born_elastic_constants(self, atoms):
        H_ncc, i_n, j_n, dr_nc, abs_dr_n = self.get_hessian(atoms, 'neighbour-list')

        # Second derivative
        C_ncccc = H_ncc.reshape(-1, 3, 1, 3, 1) * dr_nc.reshape(-1, 1, 3, 1, 1) * dr_nc.reshape(-1, 1, 1, 1, 3)
        C_cccc = -C_ncccc.sum(axis=0) / (2*atoms.get_volume())

        # Add stress term - not clear where this comes from
        stress_cc = Voigt_6_to_full_3x3_stress(self.get_stress())
        delta_cc = np.identity(3)
        C_cccc += delta_cc.reshape(3, 1, 3, 1) * stress_cc.reshape(1, 3, 1, 3)

        # If this term is included, the elastic constants are the derivative of the stress
        #C_cccc -= delta_cc.reshape(3, 3, 1, 1) * stress_cc.reshape(1, 1, 3, 3)

        # Symmetrize elastic constant tensor
        C_cccc = (C_cccc + C_cccc.swapaxes(0, 1) + C_cccc.swapaxes(2, 3) + C_cccc.swapaxes(0, 1).swapaxes(2, 3))/4

        # Add stress term
        #stress_cc = Voigt_6_to_full_3x3_stress(self.get_stress())
        #delta_cc = np.identity(3)
        #C_cccc += (delta_cc.reshape(3, 1, 3, 1) * stress_cc.reshape(1, 3, 1, 3)
        #           + delta_cc.reshape(1, 3, 3, 1) * stress_cc.reshape(3, 1, 1, 3)
        #           + delta_cc.reshape(3, 1, 1, 3) * stress_cc.reshape(1, 3, 3, 1)
        #           + delta_cc.reshape(1, 3, 1, 3) * stress_cc.reshape(3, 1, 3, 1)
        #           - 2 * delta_cc.reshape(1, 1, 3, 3) * stress_cc.reshape(3, 3, 1, 1))/2

        return C_cccc

    def get_nonaffine_forces(self, atoms):
        # Jan, implement me here
        raise NotImplementedError

    def get_non_affine_contribution_to_elastic_constants(self, atoms):
        """
        Compute the correction of non-affine displacements to the elasticity tensor.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        if self.atoms is None:
            self.atoms = atoms

        try:
            from scipy import linalg
        except ImportError:
            raise ImportError(
                "Import error: Can not compute non-affine elastic constants! Scipy is needed!")

        calc = atoms.get_calculator()

        # Non-affine forces
        forces_natccc = calc.non_affine_forces(atoms)

        # Inverse of Hessian matrix
        Hinv_nn = linalg.inv(calc.calculate_hessian_matrix(atoms))
        Hinv_nncc = Hinv_nn.reshape(nat, 3, nat, 3).swapaxes(1, 2)

        # Perform the contraction along gamma
        first_sum = np.einsum("igab, ijgk -> jkab", forces_natccc, Hinv_nncc)
        second_sum = np.einsum("jkab, jknm -> abnm", first_sum, forces_natccc)

        return second_sum

    def get_numerical_non_affine_forces(self, atoms, d=1e-6):
        """

        Calculate numerical non-affine forces using central finite differences.
        This is done by deforming the box, rescaling atoms and measure the force.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        nat = len(atoms)
        cell = atoms.cell.copy()
        V = atoms.get_volume()
        fna_ncc = np.zeros((nat, 3, 3, 3))

        for i in range(3):
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fplus = atoms.get_forces()

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fminus = atoms.get_forces()

            naForces_ncc = (fminus - fplus) / (2 * d)
            fna_ncc[:, 0, i, i] = naForces_ncc[:, 0]
            fna_ncc[:, 1, i, i] = naForces_ncc[:, 1]
            fna_ncc[:, 2, i, i] = naForces_ncc[:, 2]

            # Off diagonal
            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fplus = atoms.get_forces()

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fminus = atoms.get_forces()

            naForces_ncc = (fminus - fplus) / (4 * d)
            fna_ncc[:, 0, i, j] = naForces_ncc[:, 0]
            fna_ncc[:, 0, j, i] = naForces_ncc[:, 0]
            fna_ncc[:, 1, i, j] = naForces_ncc[:, 1]
            fna_ncc[:, 1, j, i] = naForces_ncc[:, 1]
            fna_ncc[:, 2, i, j] = naForces_ncc[:, 2]
            fna_ncc[:, 2, j, i] = naForces_ncc[:, 2]

        return fna_ncc
