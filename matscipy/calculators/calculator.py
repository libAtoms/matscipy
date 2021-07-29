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

from scipy.sparse.linalg import cg

from ase.calculators.calculator import Calculator

from ..elasticity import Voigt_6_to_full_3x3_stress

from ..numpy_tricks import mabincount


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
        """
        Compute the Born elastic constants. 

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        H_pcc, i_p, j_p, dr_pc, abs_dr_p = self.get_hessian(atoms, 'neighbour-list')

        # Second derivative
        C_pabab = H_pcc.reshape(-1, 3, 1, 3, 1) * dr_pc.reshape(-1, 1, 3, 1, 1) * dr_pc.reshape(-1, 1, 1, 1, 3)
        C_abab = -C_pabab.sum(axis=0) / (2*atoms.get_volume())

        # Symmetrize elastic constant tensor
        C_abab = (C_abab + C_abab.swapaxes(0, 1) + C_abab.swapaxes(2, 3) + C_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return C_abab

    def get_stress_contribution_to_elastic_constants(self, atoms):
        """
        Compute the correction to the elastic constants due to non-zero stress in the configuration.
        Stress term  results from working with the Cauchy stress.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        stress_cc = Voigt_6_to_full_3x3_stress(self.get_stress())
        delta_cc = np.identity(3)
        C_abab = delta_cc.reshape(3, 1, 3, 1) * stress_cc.reshape(1, 3, 1, 3) - \
                  (delta_cc.reshape(3, 3, 1, 1) * stress_cc.reshape(1, 1, 3, 3) + \
                   delta_cc.reshape(1, 1, 3, 3) * stress_cc.reshape(3, 3, 1, 1)) / 2

        # Symmetrize elastic constant tensor
        C_abab = (C_abab + C_abab.swapaxes(0, 1) + C_abab.swapaxes(2, 3) + C_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return C_abab

    def get_birch_coefficients(self, atoms):
        """
        Compute the Birch coefficients (Effective elastic constants at non-zero stress). 
        
        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        if self.atoms is None:
            self.atoms = atoms

        # Born (affine) elastic constants
        calculator = atoms.get_calculator()
        bornC_abab = calculator.get_born_elastic_constants(atoms)

        # Stress contribution to elastic constants
        stressC_abab = calculator.get_stress_contribution_to_elastic_constants(atoms)

        return bornC_abab + stressC_abab
        

    def get_nonaffine_forces(self, atoms):
        """
        Compute the non-affine forces which result from an affine deformation of atoms.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        nat = len(atoms)

        H_pcc, i_p, j_p, dr_pc, abs_dr_p = self.get_hessian(atoms, 'neighbour-list')

        naF_pcab = -0.5 * H_pcc.reshape(-1, 3, 3, 1) * dr_pc.reshape(-1, 1, 1, 3)

        naforces_icab = mabincount(i_p, naF_pcab, nat) - mabincount(j_p, naF_pcab, nat)

        return naforces_icab

    def get_non_affine_contribution_to_elastic_constants(self, atoms, eigenvalues=None, eigenvectors=None, tol=1e-5):
        """
        Compute the correction of non-affine displacements to the elasticity tensor.
        The computation of the occuring inverse of the Hessian matrix is bypassed by using a cg solver.

        If eigenvalues and and eigenvectors are given the inverse of the Hessian can be easily computed.


        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        eigenvalues: array
            Eigenvalues in ascending order obtained by diagonalization of Hessian matrix.
            If given 

        eigenvectors: array
            Eigenvectors corresponding to eigenvalues.

        tol: float
            Tolerance for the conjugate-gradient solver. 

        """

        nat = len(atoms)

        calc = atoms.get_calculator()

        C_abab = np.zeros((3,3,3,3))        

        if (eigenvalues is not None) and (eigenvectors is not None):
            naforces_icab = calc.get_nonaffine_forces(atoms)

            G_incc = (eigenvectors.T).reshape(-1, 3*nat, 1, 1) * naforces_icab.reshape(1, 3*nat, 3, 3)
            G_incc = (G_incc.T/np.sqrt(eigenvalues)).T
            G_icc  = np.sum(G_incc, axis=1)
            C_abab = np.sum(G_icc.reshape(-1,3,3,1,1) * G_icc.reshape(-1,1,1,3,3), axis=0)

        else:
            H_nn = calc.get_hessian(atoms, "sparse")
            naforces_icab = calc.get_nonaffine_forces(atoms)

            D_iab = np.zeros((3*nat, 3, 3))
            for i in range(3):
                for j in range(3):
                    x, info = cg(H_nn, naforces_icab[:, :, i, j].flatten(), atol=tol)
                    if info != 0:
                        raise RuntimeError(" info > 0: CG tolerance not achieved, info < 0: Exceeded number of iterations.")
                    D_iab[:,i,j] = x

            C_abab = np.sum(naforces_icab.reshape(3*nat, 3, 3, 1, 1) * D_iab.reshape(3*nat, 1, 1, 3, 3), axis=0)
        
        # Symmetrize 
        C_abab = (C_abab + C_abab.swapaxes(0, 1) + C_abab.swapaxes(2, 3) + C_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4             

        return -C_abab/atoms.get_volume()

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
