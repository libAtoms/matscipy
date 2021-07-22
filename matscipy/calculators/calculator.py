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

from scipy import linalg

from ase.calculators.calculator import Calculator

from ..elasticity import Voigt_6_to_full_3x3_stress


class MatscipyCalculator(Calculator):
    def get_hessian(self, atoms, format='sparse', divide_by_masses=False):
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

        # Add stress term that comes from working with the Cauchy stress
        #stress_cc = Voigt_6_to_full_3x3_stress(self.get_stress())
        #delta_cc = np.identity(3)
        #C_cccc += delta_cc.reshape(3, 1, 3, 1) * stress_cc.reshape(1, 3, 1, 3) - \
        #          (delta_cc.reshape(3, 3, 1, 1) * stress_cc.reshape(1, 1, 3, 3) + \
        #           delta_cc.reshape(1, 1, 3, 3) * stress_cc.reshape(3, 3, 1, 1)) / 2

        # Symmetrize elastic constant tensor
        C_cccc = (C_cccc + C_cccc.swapaxes(0, 1) + C_cccc.swapaxes(2, 3) + C_cccc.swapaxes(0, 1).swapaxes(2, 3)) / 4

        # Add stress term
        #stress_cc = Voigt_6_to_full_3x3_stress(self.get_stress())
        #delta_cc = np.identity(3)
        #C_cccc += (delta_cc.reshape(3, 1, 3, 1) * stress_cc.reshape(1, 3, 1, 3)
        #           + delta_cc.reshape(1, 3, 3, 1) * stress_cc.reshape(3, 1, 1, 3)
        #           + delta_cc.reshape(3, 1, 1, 3) * stress_cc.reshape(1, 3, 3, 1)
        #           + delta_cc.reshape(1, 3, 1, 3) * stress_cc.reshape(3, 1, 3, 1)
        #           - 2 * delta_cc.reshape(1, 1, 3, 3) * stress_cc.reshape(3, 3, 1, 1))/2

        return C_cccc

    def get_stress_contribution_to_elastic_constants(self, atoms):
        # Add stress term that comes from working with the Cauchy stress
        stress_cc = Voigt_6_to_full_3x3_stress(self.get_stress())
        delta_cc = np.identity(3)
        C_cccc = delta_cc.reshape(3, 1, 3, 1) * stress_cc.reshape(1, 3, 1, 3) - \
                  (delta_cc.reshape(3, 3, 1, 1) * stress_cc.reshape(1, 1, 3, 3) + \
                   delta_cc.reshape(1, 1, 3, 3) * stress_cc.reshape(3, 3, 1, 1)) / 2

        # Symmetrize elastic constant tensor
        C_cccc = (C_cccc + C_cccc.swapaxes(0, 1) + C_cccc.swapaxes(2, 3) + C_cccc.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return C_cccc

    def get_birch_coefficients(self, atoms):
        if self.atoms is None:
            self.atoms = atoms

        # Born (affine) elastic constants
        calculator = atoms.get_calculator()
        bornC_cccc = calculator.get_born_elastic_constants(atoms)

        # Stress contribution to elastic constants
        stressC_cccc = calculator.get_stress_contribution_to_elastic_constants(atoms)

        return bornC_cccc + stressC_cccc
        

    def get_nonaffine_forces(self, atoms):
        """
        Compute the non-affine forces which result from an affine deformation of atoms.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        nat = len(atoms)
        H_ncc, i_n, j_n, dr_nc, abs_dr_n = self.get_hessian(atoms, 'neighbour-list')
        naForces_nccc = -0.5 * H_ncc.reshape(-1, 3, 3, 1) * dr_nc.reshape(-1, 1, 1, 3)
        naForces_natccc = np.empty((nat, 3, 3, 3))
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    naForces_natccc[:, i, j, k] = np.bincount(i_n, weights=naForces_nccc[:, i, j, k], minlength=nat) - \
                        np.bincount(j_n, weights=naForces_nccc[:, i, j, k], minlength=nat) 

        return naForces_natccc

    def get_non_affine_contribution_to_elastic_constants(self, atoms):
        """
        Compute the correction of non-affine displacements to the elasticity tensor.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """

        nat = len(atoms)
        calc = atoms.get_calculator()

        # Non-affine forces
        forces_natccc = calc.get_nonaffine_forces(atoms)

        # Diagonalize 
        H_nn = calc.get_hessian(atoms, "sparse").todense()
        eigvalues_n, eigvecs_nn = linalg.eigh(H_nn, b=None, subset_by_index=[3, 3*nat-1])
         
        #B_ncc = np.sum(eigvecs_nn.reshape(3*nat-3, 3*nat, 1, 1) * forces_natccc.reshape(1, 3*nat, 3, 3), axis=1)
        #print(B_ncc.shape)
        #B_ncc /= np.sqrt(eigvalues_n.reshape(3*nat-3, 1, 1))
        #B_ncc = B_ncc.reshape(3*nat-3, 3, 3, 1, 1) * B_ncc.reshape(3*nat-3, 1, 1, 3, 3)
        #C_cccc2 = np.sum(B_ncc, axis=0)

        # Compute non-affine contribution 
        C_cccc = np.empty((3,3,3,3))
        for index in range(0, 3*nat -3):
            first_con = np.sum((eigvecs_nn[:,index]).reshape(3*nat, 1, 1) * forces_natccc.reshape(3*nat, 3, 3), axis=0)
            C_cccc += (first_con.reshape(3,3,1,1) * first_con.reshape(1,1,3,3))/eigvalues_n[index]

        return - C_cccc/atoms.get_volume()

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
        fna_ncc = np.zeros((nat, 3, 3, 3))

        for i in range(3):
            # Diagonal 
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fplus = atoms.get_forces()

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            fminus = atoms.get_forces()

            naForces_ncc = (fplus - fminus) / (2 * d)
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

            naForces_ncc = (fplus - fminus) / (4 * d)
            fna_ncc[:, 0, i, j] = naForces_ncc[:, 0]
            fna_ncc[:, 0, j, i] = naForces_ncc[:, 0]
            fna_ncc[:, 1, i, j] = naForces_ncc[:, 1]
            fna_ncc[:, 1, j, i] = naForces_ncc[:, 1]
            fna_ncc[:, 2, i, j] = naForces_ncc[:, 2]
            fna_ncc[:, 2, j, i] = naForces_ncc[:, 2]

        return fna_ncc
