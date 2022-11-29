"""Hessian preconditioner."""

import numpy as np

from ase.optimize.precon import Precon


class HessianPrecon(Precon):
    """Preconditioner for dense Hessian."""

    def __init__(self,
                 calc,
                 c_stab=0.01,
                 move_tol=0.1,
                 P=None,
                 old_positions=None):
        self.calc = calc
        self.P = P
        self.c_stab = c_stab
        self.move_tol = move_tol
        self.old_positions = old_positions

    def make_precon(self, atoms):
        if self.P is None or self.old_positions is None:
            max_move = np.inf
        else:
            max_move = np.abs(atoms.positions - self.old_positions).max()
        if self.P is None or max_move > self.move_tol:
            print('Recomputing Hessian...')
            P = self.calc.get_hessian(atoms, format='dense')
            P += np.diag([self.c_stab] * 3 * len(atoms))
            D, Q = np.linalg.eigh(P)
            print(D[:10])
            if np.any(D < 0):
                print(f'Flipping {np.sum(D < 0)} negative eigenvalues')
                self.P = np.array(Q @ np.diag(np.abs(D)) @ Q.T)
            else:
                self.P = np.array(P)
            self.old_positions = atoms.positions.copy()

    def Pdot(self, x):
        return self.P.dot(x)

    def solve(self, x):
        return np.linalg.solve(self.P, x)

    def copy(self):
        return HessianPrecon(self.calc, self.c_stab, self.move_tol, None, None)

    def asarray(self):
        return self.P.copy()
