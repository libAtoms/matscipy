"""Hessian preconditioner."""

import numpy as np

from ase.optimize.precon import Precon
from ase.calculators.calculator import equal


class HessianPrecon(Precon):
    """Preconditioner for dense Hessian."""

    def __init__(self,
                 c_stab=0.01,
                 move_tol=0.1,
                 P=None,
                 old_positions=None):
        self.P = P
        self.c_stab = c_stab
        self.move_tol = move_tol
        self.old_positions = old_positions

    def make_precon(self, atoms):
        has_moved = not equal(atoms.positions,
                              self.old_positions,
                              atol=self.move_tol)
        initialized = self.P is not None and self.old_positions is not None

        if not initialized or has_moved:
            P = atoms.calc.get_property("hessian", atoms).todense()
            di = np.diag_indices_from(P)
            P[di] += self.c_stab
            D, Q = np.linalg.eigh(P)
            if np.any(D < 0):
                self.P = np.array(Q @ np.diag(np.abs(D)) @ Q.T)
            else:
                self.P = np.array(P)
            self.old_positions = atoms.positions.copy()

    def Pdot(self, x):
        return self.P.dot(x)

    def solve(self, x):
        return np.linalg.solve(self.P, x)

    def copy(self):
        return HessianPrecon(self.c_stab, self.move_tol, None, None)

    def asarray(self):
        return self.P.copy()
