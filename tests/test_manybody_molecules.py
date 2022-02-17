#
# Copyright 2022 Lucas Frérot (U. Freiburg)
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
import numpy.testing as nt

from ase import Atoms
from matscipy.calculators.manybody.calculator import NiceManybody
from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood


class HarmonicBond(NiceManybody.F):
    def __init__(self, r0, k):
        self.r0 = r0
        self.k = k

    def __call__(self, r, xi, atype, ptype):
        e = 0.5 * self.k * (r - self.r0)**2
        e[ptype < 0] = 0  # ignore bonds from angles
        return e + xi

    def gradient(self, r, xi, atype, ptype):
        g = self.k * (r - self.r0)
        g[ptype < 0] = 0
        return [g, np.ones_like(xi)]

    def hessian(self, r, xi, atype, ptype):
        h = np.full_like(r, self.k)
        h[ptype < 0] = 0
        return [h, np.zeros_like(r), np.zeros_like(r)]


class ZeroAngle(NiceManybody.G):
    def __call__(self, *args):
        return np.zeros(args[0].shape[0])

    def gradient(self, *args):
        return np.zeros([2] + list(args[0].shape))

    def hessian(self, *args):
        return np.zeros([3] + list(args[0].shape))


class HarmonicAngle(NiceManybody.G):
    def __init__(self, a0, k, atoms: Atoms):
        self.a0 = a0
        self.k = k
        self.atoms = atoms

    def __call__(self, r_ij_c, r_ik_c, *args):
        _, (r_ij, r_ik, r_jk) = self._distance_triplet(r_ij_c, r_ik_c,
                                                       self.atoms.cell,
                                                       self.atoms.pbc)

        a = np.arccos(-(r_ij**2 + r_jk**2 - r_ik**2) / (2 * r_ij * r_jk))
        return 0.5 * self.k * (a - self.a0)**2

    def gradient(self, r_ij, r_ik, *args):
        D, d = self._distance_triplet(r_ij, r_ik,
                                      self.atoms.cell,
                                      self.atoms.pbc)
        # Normal vectors
        n_ij_c = D[0] / d[0][:, np.newaxis]
        n_ik_c = D[1] / d[1][:, np.newaxis]
        n_jk_c = D[2] / d[2][:, np.newaxis]

        # cos of angle
        f = -(d[0]**2 + d[2]**2 - d[1]**2) / (2 * d[0] * d[2])
        # derivatives with respect to triangle lengths
        df_rij = -(d[0]**2 - d[2]**2 + d[1]**2) / (2 * d[0]**2 * d[2])
        df_rjk = -(d[2]**2 - d[0]**2 + d[1]**2) / (2 * d[2]**2 * d[2])
        df_rik = d[1] / (d[0] * d[2])

        # Scalar derivatives
        def E_(a):
            return self.k * (a - self.a0)  # noqa

        def h_(f):
            with np.errstate(divide='raise'):
                d_arccos = -1 / np.sqrt(1 - f**2)
            return E_(np.arccos(f)) * d_arccos

        # Broadcast slices
        _c = np.s_[:, np.newaxis]

        # Derivatives with respect to vectors rij and rik
        dG = np.zeros([2] + list(r_ij.shape))
        # dG_rij
        dG[0] = df_rij[_c] * n_ij_c - df_rjk[_c] * n_jk_c
        # dG_rik
        dG[1] = df_rik[_c] * n_ik_c + df_rjk[_c] * n_jk_c

        dG *= h_(f)[_c]
        return dG

    def hessian(self, r_ij, r_ik, *args):
        return np.zeros([3] + list(r_ij.shape))


def test_harmonic_bond():
    r0 = 0.5
    atoms = Atoms("CO2",
                  positions=[[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
                  cell=[5, 5, 5])
    molecules = Molecules(bonds_connectivity=[[0, 1], [1, 2]])
    neigh = MolecularNeighbourhood(molecules)
    calc = NiceManybody(HarmonicBond(r0, 1), ZeroAngle(), neigh)
    atoms.calc = calc

    # Testing potential energy
    epot = atoms.get_potential_energy()
    epot_ref = 2 * (0.5 * r0**2)
    nt.assert_allclose(epot, epot_ref, rtol=1e-15)

    # Testing forces
    f = atoms.get_forces()
    f_ref = np.zeros_like(f)
    f_ref[:, 0] = [0.5, 0, -0.5]
    nt.assert_allclose(f, f_ref, rtol=1e-15)


def test_harmonic_angle():
    k, r0, kt, theta0 = 0, 1, 1, np.pi / 3
    atoms = Atoms("CO2",
                  positions=[[-1, 0, 0],
                             [0, 0, 0],
                             [np.cos(theta0), np.sin(theta0), 0]],
                  cell=[5, 5, 5])
    molecules = Molecules(bonds_connectivity=[[0, 1], [1, 2]],
                          angles_connectivity=[[0, 1, 2]])
    neigh = MolecularNeighbourhood(molecules)
    calc = NiceManybody(HarmonicBond(r0, k),
                        HarmonicAngle(0, kt, atoms),
                        neigh)
    atoms.calc = calc

    # Testing potential energy
    epot = atoms.get_potential_energy()
    epot_ref = 2 * (0.5 * k * r0**2) + 0.5 * kt * theta0**2
    nt.assert_allclose(epot, epot_ref, rtol=1e-15)

    # Testing forces
    f = atoms.get_forces()

    # Finite differences forces
    f_ref = calc.calculate_numerical_forces(atoms, d=1e-6)
    nt.assert_allclose(f, f_ref, rtol=1e-6, atol=1e-9)

    # Symmetric frame of reference
    theta0 = -theta0 / 2
    rot = np.array([[np.cos(theta0), -np.sin(theta0), 0],
                    [np.sin(theta0),  np.cos(theta0), 0],
                    [0,               0,              1]])
    f = np.einsum('ij,aj', rot, f)

    # Checking symmetries
    nt.assert_allclose(f[0, 0], -f[2, 0], rtol=1e-15)
    nt.assert_allclose(f[0, 1],  f[2, 1], rtol=1e-15)

    # Checking zeros
    assert np.sum(np.abs(f[1, (0, 2)])) < 1e-15
    assert np.abs(f.sum()) < 1e-15

