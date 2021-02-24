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
import time
import warnings

from math import sqrt
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import eigsh


from ase.optimize.precon import SparsePrecon, Exp

class Hessian_EAM(SparsePrecon):
    """ Creates matrix with the hessian matrix of the matscipy EAM calculator.
    """

    def __init__(self, calculator = None, nbr_of_evals = 5, 
                 terms=["pair", "T1", "T2", "T3", "T4", "T5", "T6", "T7"], 
                 r_cut=None, mu=None, mu_c=None, dim=3, c_stab=0.1,
                 force_stab=False,
                 reinitialize=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True, logfile=None):
        """Initialise an Hessian_EAM preconditioner with given parameters.

        Args:
            calculator: matscipy eam calculator
                The calculator that calculates the hessian of the system.
            nbr_of_evals: int
                The number of evaluation after the matrix is calculated.
            terms: list of strings
                The list which terms of the analytic eam hessian should be
                calculated.
            r_cut, mu, c_stab, dim, recalc_mu, array_convention: see
                precon.__init__()
        """
        super().__init__(r_cut=r_cut, mu=mu, mu_c=mu_c,
                         dim=dim, c_stab=c_stab,
                         force_stab=force_stab,
                         reinitialize=reinitialize,
                         array_convention=array_convention,
                         solver=solver, solve_tol=solve_tol,
                         apply_positions=apply_positions,
                         apply_cell=apply_cell,
                         logfile=logfile)
        
        self.calculator = calculator
        self.terms = terms
        self.nbr_of_evals = nbr_of_evals
        self.ctr_of_evals = 0

    def make_precon(self, atoms, reinitialize=None):    
        start_time = time.time()
        # Create the preconditioner:
        if (self.P is None or self.ctr_of_evals >= self.nbr_of_evals):
            self._make_sparse_precon(atoms, force_stab=self.force_stab)
            self.ctr_of_evals = 0
        self.ctr_of_evals += 1
        self.logfile.write('--- Precon created in %s seconds ---\n'
                           % (time.time() - start_time))


    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        self.logfile.write('creating eam hessian: terms={}'.format(self.terms))
        start_time = time.time()
        self.P  = self.calculator.calculate_hessian_matrix(atoms, terms = self.terms)
        val, vec = eigsh(self.P, k = 1, which= "SA")
        a_s = 0.00001
        a = a_s
        for k in range(100):
            self.P += a * sparse.eye(self.P.shape[0])
            val, vec = eigsh(self.P, k = 1, which= "SA")
            if (val > 0):
                break
            a = 1.5 * a
        self.logfile.write('--- Hessian created in %s seconds ---\n'
                           % (time.time() - start_time))
        self.create_solver()
        
class Hessian_EAM_EXP(SparsePrecon):
    """ Creates matrix with the hessian matrix of the matscipy EAM calculator.
    """

    def __init__(self, calculator = None, nbr_of_evals = 5, 
                 terms=["pair", "T1", "T2", "T3", "T4", "T5", "T6", "T7"], 
                 r_cut=None, mu=None, mu_c=None, dim=3, c_stab=0.1,
                 force_stab=False,
                 reinitialize=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True, logfile=None):
        """Initialise an Hessian_EAM preconditioner with given parameters.

        Args:
            calculator: matscipy eam calculator
                The calculator that calculates the hessian of the system.
            nbr_of_evals: int
                The number of evaluation after the matrix is calculated.
            terms: list of strings
                The list which terms of the analytic eam hessian should be
                calculated.
            r_cut, mu, c_stab, dim, recalc_mu, array_convention: see
                precon.__init__()
        """
        super().__init__(r_cut=r_cut, mu=mu, mu_c=mu_c,
                         dim=dim, c_stab=c_stab,
                         force_stab=force_stab,
                         reinitialize=reinitialize,
                         array_convention=array_convention,
                         solver=solver, solve_tol=solve_tol,
                         apply_positions=apply_positions,
                         apply_cell=apply_cell,
                         logfile=logfile)
        
        self.calculator = calculator
        self.terms = terms
        self.nbr_of_evals = nbr_of_evals
        self.ctr_of_evals = 0
        self.exp_precon = Exp()
        self.fmax = float('inf')

    def make_precon(self, atoms, reinitialize=None):    
        start_time = time.time()
        forces = atoms.get_forces()
        last_fmax = self.fmax
        self.fmax = sqrt((forces**2).sum(axis=1).max())
        # Create the preconditioner:
        if (self.P is None or self.ctr_of_evals >= self.nbr_of_evals or (self.fmax <= 0.1 and last_fmax > 0.1)):
            self._make_sparse_precon(atoms, force_stab=self.force_stab)
            self.ctr_of_evals = 0
        if self.fmax <= 0.1:
            self.ctr_of_evals += 1
        self.logfile.write('--- Precon created in %s seconds ---\n'
                           % (time.time() - start_time))


    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        self.logfile.write('creating eam hessian: terms={}'.format(self.terms))
        start_time = time.time()
        if self.fmax > 0.1:
            self.exp_precon.make_precon(atoms)
            self.P = self.exp_precon.P
            print("Fmax > 1")
        else:
            self.P  = self.calculator.calculate_hessian_matrix(atoms, terms = self.terms)
            val, vec = eigsh(self.P, k = 1, which= "SA")
            a_s = 0.00001
            a = a_s
            for k in range(100):
                self.P += a * sparse.eye(self.P.shape[0])
                val, vec = eigsh(self.P, k = 1, which= "SA")
                if (val > 0):
                    break
                a = 1.5 * a
        self.logfile.write('--- Hessian created in %s seconds ---\n'
                           % (time.time() - start_time))
        self.create_solver()