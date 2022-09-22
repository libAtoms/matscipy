#
# Copyright 2021 Jan Griesser (U. Freiburg)
#           2021 Lars Pastewka (U. Freiburg)
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
import numpy as np

from scipy.sparse.linalg import cg
from ase.calculators.calculator import Calculator
from numpy import deprecate

from ..elasticity import (
    Voigt_6_to_full_3x3_stress,
    nonaffine_elastic_contribution,
)

from ..numerical import numerical_nonaffine_forces

from ..numpy_tricks import mabincount


class MatscipyCalculator(Calculator):
    def set_atoms(self, atoms):
        """Set inner Atoms object."""
        self.atoms = atoms.copy()

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)

        # Dispatching calls to special properties
        properties_map = {
            'hessian': self.get_hessian,
            'dynamical_matrix': self.get_dynamical_matrix,
            'nonaffine_forces': self.get_nonaffine_forces,
            'born_constants': self.get_born_elastic_constants,
            'stress_elastic_contribution':
            self.get_stress_contribution_to_elastic_constants,
            'birch_coefficients': self.get_birch_coefficients,
            'nonaffine_elastic_contribution':
            self.get_non_affine_contribution_to_elastic_constants,
            'elastic_constants':
            self.get_elastic_constants
        }

        for prop in filter(lambda p: p in properties, properties_map):
            self.results[prop] = properties_map[prop](atoms)

    @staticmethod
    def _virial(pair_distance_vectors, pair_forces):
        r_pc = pair_distance_vectors
        f_pc = pair_forces

        return np.concatenate([
            # diagonal components (xx, yy, zz)
            np.einsum('pi,pi->i', r_pc, f_pc, optimize=True),

            # off-diagonal (yz, xz, xy)
            np.einsum('pi,pi->i', r_pc[:, (1, 0, 0)], f_pc[:, (2, 2, 1)],
                      optimize=True)
        ])

    def get_dynamical_matrix(self, atoms):
        """
        Compute dynamical matrix (=mass weighted Hessian).
        """
        return self.get_hessian(atoms, format="sparse", divide_by_masses=True)

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

        # Second derivative with respect to displacement gradient
        C_pabab = H_pcc.reshape(-1, 3, 1, 3, 1) * dr_pc.reshape(-1, 1, 3, 1, 1) * dr_pc.reshape(-1, 1, 1, 1, 3)
        C_abab = -C_pabab.sum(axis=0) / (2 * atoms.get_volume())

        # This contribution is necessary in order to obtain second derivative with respect to Green-Lagrange
        stress_ab = self.get_property('stress', atoms)
        delta_ab = np.identity(3)

        if stress_ab.shape != (3, 3):
            stress_ab = Voigt_6_to_full_3x3_stress(stress_ab)

        C_abab -= stress_ab.reshape(1, 3, 1, 3) * delta_ab.reshape(3, 1, 3, 1)

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

        stress_ab = self.get_property('stress', atoms)

        if stress_ab.shape != (3, 3):
            stress_ab = Voigt_6_to_full_3x3_stress(stress_ab)

        delta_ab = np.identity(3)

        stress_contribution = 0.5 * sum(
            np.einsum(einsum, stress_ab, delta_ab)
            for einsum in (
                    'am,bn',
                    'an,bm',
                    'bm,an',
                    'bn,am',
            )
        )

        stress_contribution -= np.einsum('ab,mn', stress_ab, delta_ab)

        return stress_contribution

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
        calculator = self
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

    def get_elastic_constants(self,
                              atoms,
                              cg_parameters={
                                  "x0": None,
                                  "tol": 1e-5,
                                  "maxiter": None,
                                  "M": None,
                                  "callback": None,
                                  "atol": 1e-5}):
        """
        Compute the elastic constants at zero temperature.
        These are sum of the born, the non-affine and the stress contribution.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        cg_parameters: dict
            Dictonary for the conjugate-gradient solver.

            x0: {array, matrix}
                Starting guess for the solution.

            tol/atol: float, optional
                Tolerances for convergence, norm(residual) <= max(tol*norm(b), atol).

            maxiter: int
                Maximum number of iterations. Iteration will stop after maxiter steps even if the specified tolerance has not been achieved.

            M: {sparse matrix, dense matrix, LinearOperator}
                Preconditioner for A.

            callback: function
                User-supplied function to call after each iteration.
        """
        if self.atoms is None:
            self.atoms = atoms

        # Born (affine) elastic constants
        calculator = self
        C = calculator.get_born_elastic_constants(atoms)

        # Stress contribution to elastic constants
        C += calculator.get_stress_contribution_to_elastic_constants(atoms)

        # Non-affine contribution
        C += nonaffine_elastic_contribution(atoms, cg_parameters=cg_parameters)

        return C

    @deprecate(new_name="elasticity.nonaffine_elastic_contribution")
    def get_non_affine_contribution_to_elastic_constants(self, atoms, eigenvalues=None, eigenvectors=None, pc_parameters=None, cg_parameters={"x0": None, "tol": 1e-5, "maxiter": None, "M": None, "callback": None, "atol": 1e-5}):
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
            If given, use eigenvalues and eigenvectors to compute non-affine contribution.

        eigenvectors: array
            Eigenvectors corresponding to eigenvalues.

        cg_parameters: dict
            Dictonary for the conjugate-gradient solver.

            x0: {array, matrix}
                Starting guess for the solution.

            tol/atol: float, optional
                Tolerances for convergence, norm(residual) <= max(tol*norm(b), atol).

            maxiter: int
                Maximum number of iterations. Iteration will stop after maxiter steps even if the specified tolerance has not been achieved.

            M: {sparse matrix, dense matrix, LinearOperator}
                Preconditioner for A.

            callback: function
                User-supplied function to call after each iteration.

        pc_parameters: dict
            Dictonary for the incomplete LU decomposition of the Hessian.

            A: array_like
                Sparse matrix to factorize.

            drop_tol: float
                Drop tolerance for an incomplete LU decomposition.

            fill_factor: float
                Specifies the fill ratio upper bound.

            drop_rule: str
                Comma-separated string of drop rules to use.

            permc_spec: str
                How to permute the columns of the matrix for sparsity.

            diag_pivot_thresh: float
                Threshold used for a diagonal entry to be an acceptable pivot.

            relax: int
                Expert option for customizing the degree of relaxing supernodes.

            panel_size: int
                Expert option for customizing the panel size.

            options: dict
                Dictionary containing additional expert options to SuperLU.
        """

        nat = len(atoms)

        calc = self

        if (eigenvalues is not None) and (eigenvectors is not None):
            naforces_icab = calc.get_nonaffine_forces(atoms)

            G_incc = (eigenvectors.T).reshape(-1, 3*nat, 1, 1) * naforces_icab.reshape(1, 3*nat, 3, 3)
            G_incc = (G_incc.T/np.sqrt(eigenvalues)).T
            G_icc  = np.sum(G_incc, axis=1)
            C_abab = np.sum(G_icc.reshape(-1,3,3,1,1) * G_icc.reshape(-1,1,1,3,3), axis=0)

        else:
            H_nn = calc.get_hessian(atoms)
            naforces_icab = calc.get_nonaffine_forces(atoms)

            if pc_parameters != None:
                # Transform H to csc
                H_nn = H_nn.tocsc()

                # Compute incomplete LU
                approx_Hinv = spilu(H_nn, **pc_parameters)
                operator_Hinv = LinearOperator(H_nn.shape, approx_Hinv.solve)
                cg_parameters["M"] = operator_Hinv

            D_iab = np.zeros((3*nat, 3, 3))
            for i in range(3):
                for j in range(3):
                    x, info = cg(H_nn, naforces_icab[:, :, i, j].flatten(), **cg_parameters)
                    if info != 0:
                        print("info: ", info)
                        raise RuntimeError(" info > 0: CG tolerance not achieved, info < 0: Exceeded number of iterations.")
                    D_iab[:,i,j] = x

            C_abab = np.sum(naforces_icab.reshape(3*nat, 3, 3, 1, 1) * D_iab.reshape(3*nat, 1, 1, 3, 3), axis=0)

        # Symmetrize
        C_abab = (C_abab + C_abab.swapaxes(0, 1) + C_abab.swapaxes(2, 3) + C_abab.swapaxes(0, 1).swapaxes(2, 3)) / 4

        return -C_abab/atoms.get_volume()

    @deprecate(new_name='numerical.numerical_nonaffine_forces')
    def get_numerical_non_affine_forces(self, atoms, d=1e-6):
        """

        Calculate numerical non-affine forces using central finite differences.
        This is done by deforming the box, rescaling atoms and measure the force.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        """
        return numerical_nonaffine_forces(atoms, d=d)
