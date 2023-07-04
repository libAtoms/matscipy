import unittest

import numpy as np
import matscipytest
import scipy.sparse.linalg as sla

from matscipy.calculators.eam import EAM
from matscipy.precon import HessianPrecon
from numpy.linalg import norm
from numpy.testing import assert_allclose

from ase.build import bulk
from ase.optimize.precon import PreconLBFGS
from ase.optimize import ODE12r, LBFGS
from ase.neb import NEB, NEBOptimizer
from ase.geometry.geometry import get_distances


class TestHessianPrecon(matscipytest.MatSciPyTestCase):

    def setUp(self):
        self.atoms = bulk("Ni", cubic=True) * 3
        del self.atoms[0]
        np.random.seed(0)
        self.atoms.rattle(1e-2)
        self.eam = EAM('Mishin-Ni-Al-2009.eam.alloy')
        self.tol = 1e-6

    def test_newton_opt(self):
        atoms, eam = self.atoms, self.eam

        f = eam.get_forces(atoms)
        norm_f = norm(f, np.inf)
        while norm_f > self.tol:
            H = eam.get_hessian(atoms).tocsc()
            D, P = sla.eigs(H, which='SM')
            print(D[D > 1e-6])
            print(f'|F| = {norm_f:12.6f}')
            step = sla.spsolve(H, f.reshape(-1)).reshape(-1, 3)
            atoms.positions += step
            f = eam.get_forces(atoms)
            norm_f = norm(f, np.inf)

    def test_precon_opt(self):
        atoms, eam = self.atoms, self.eam
        noprecon_atoms = atoms.copy()
        atoms.calc = eam
        noprecon_atoms.calc = eam
        opt = PreconLBFGS(atoms, precon=HessianPrecon())
        opt.run(fmax=self.tol)

        opt = LBFGS(noprecon_atoms)
        opt.run(fmax=self.tol)

        assert_allclose(
            atoms.positions, noprecon_atoms.positions, atol=self.tol)

    def test_precon_neb(self):
        N_cell = 4
        N_intermediate = 5

        initial = bulk('Ni', cubic=True)
        initial *= N_cell

        # place vacancy near centre of cell
        D, D_len = get_distances(
            np.diag(initial.cell) / 2, initial.positions, initial.cell,
            initial.pbc)
        vac_index = D_len.argmin()
        vac_pos = initial.positions[vac_index]
        del initial[vac_index]

        # identify two opposing nearest neighbours of the vacancy
        D, D_len = get_distances(vac_pos, initial.positions, initial.cell,
                                 initial.pbc)
        D = D[0, :]
        D_len = D_len[0, :]

        nn_mask = np.abs(D_len - D_len.min()) < 1e-8
        i1 = nn_mask.nonzero()[0][0]
        i2 = ((D + D[i1])**2).sum(axis=1).argmin()

        print(f'vac_index={vac_index} i1={i1} i2={i2} '
              f'distance={initial.get_distance(i1, i2, mic=True)}')

        final = initial.copy()
        final.positions[i1] = vac_pos

        initial.calc = self.eam
        final.calc = self.eam

        precon = HessianPrecon(c_stab=0.1)
        qn = ODE12r(initial, precon=precon)
        qn.run(fmax=1e-3)
        qn = ODE12r(final, precon=precon)
        qn.run(fmax=1e-3)

        images = [initial]
        for image in range(N_intermediate):
            image = initial.copy()
            image.calc = self.eam
            images.append(image)
        images.append(final)

        dummy_neb = NEB(images)
        dummy_neb.interpolate()

        neb = NEB(
            dummy_neb.images,
            allow_shared_calculator=True,
            precon=precon,
            method='spline')
        # neb.interpolate()
        opt = NEBOptimizer(neb)
        opt.run(fmax=0.1)  # large tolerance for test speed


if __name__ == '__main__':
    unittest.main()
