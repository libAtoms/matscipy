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
from matscipy.numerical import numerical_forces, numerical_hessian

import pytest


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
        return np.zeros([3] + list(args[0].shape) + [args[0].shape[1]])


class ZeroBond(NiceManybody.G):
    def __call__(self, r, xi, *args):
        return xi

    def gradient(self, r, xi, *args):
        return [np.zeros_like(xi), np.ones_like(xi)]

    def hessian(self, r, xi, *args):
        return [np.zeros_like(r)] * 3


class HarmonicAngle(NiceManybody.G):
    def __init__(self, a0, k, atoms: Atoms):
        self.a0 = a0
        self.k = k
        self.atoms = atoms

    def __call__(self, r_ij_c, r_ik_c, *args):
        _, (r_ij, r_ik, r_jk) = self._distance_triplet(
            r_ij_c, r_ik_c, self.atoms.cell, self.atoms.pbc
        )

        a = np.arccos(-(r_ij**2 + r_jk**2 - r_ik**2) / (2 * r_ij * r_jk))
        return 0.5 * self.k * (a - self.a0)**2

    def gradient(self, r_ij_c, r_ik_c, *args):
        D, d = self._distance_triplet(
            r_ij_c, r_ik_c, self.atoms.cell, self.atoms.pbc
        )

        # Broadcast slices
        _c = np.s_[:, np.newaxis]

        # Mapping: u <- r_ij, v <- r_ik, w <- r_jk = |r_ik_c - r_ij_c|
        u, v, w = d

        # Normal vectors
        nu, nv, nw = (D[i] / d[i][_c] for i in range(3))

        # cos of angle
        f = -(u**2 + w**2 - v**2) / (2 * u * w)
        # derivatives with respect to triangle lengths
        df_u = -(u**2 - w**2 + v**2) / (2 * u**2 * w)
        df_w = -(w**2 - u**2 + v**2) / (2 * w**2 * u)
        df_v = v / (u * w)

        # Scalar derivatives
        def E_(a):
            return self.k * (a - self.a0)  # noqa

        def h_(f):
            with np.errstate(divide="raise"):
                d_arccos = -1 / np.sqrt(1 - f**2)
            return E_(np.arccos(f)) * d_arccos

        # Derivatives with respect to vectors rij and rik
        dG = np.zeros([2] + list(r_ij_c.shape))
        # dG_rij
        dG[0] = df_u[_c] * nu + df_w[_c] * (-nw)
        # dG_rik
        dG[1] = df_v[_c] * nv + df_w[_c] * (+nw)

        dG *= h_(f)[_c]
        return dG

    def hessian(self, r_ij_c, r_ik_c, *args):
        D, d = self._distance_triplet(
            r_ij_c, r_ik_c, self.atoms.cell, self.atoms.pbc
        )

        # Utilities
        _c = np.s_[:, np.newaxis]
        _cc = np.s_[:, np.newaxis, np.newaxis]
        _o = lambda u, v: np.einsum('...i,...j', u, v, optimize=True) # noqa

        # Scalar functions
        dE = lambda a: self.k * (a - self.a0)  # Force
        ddE = lambda a: self.k                 # Stiffness
        arccos = np.arccos
        darccos = lambda x: -1 / np.sqrt(1 - x**2)
        ddarccos = lambda x: -x / (1 - x**2)**(3/2)

        dh = lambda f: dE(arccos(f)) * darccos(f)
        ddh = lambda f: (
            ddE(arccos(f)) * darccos(f) * darccos(f)
            + dE(arccos(f)) * ddarccos(f)
        )

        # Mapping: u <- r_ij, v <- r_ik, w <- r_jk = |r_ik_c - r_ij_c|
        u, v, w = d

        # Normal vectors
        nu, nv, nw = (D[i] / d[i][_c] for i in range(3))

        # Outer products
        nunu, nvnv, nwnw = (_o(n, n) for n in (nu, nv, nw))

        # Normal tensors
        Id = np.eye(3)[np.newaxis, :]
        nnu, nnv, nnw = ((Id - o) / d[i][_cc]
                         for i, o in enumerate((nunu, nvnv, nwnw)))

        # cos of angle
        f = -(u**2 + w**2 - v**2) / (2 * u * w)
        # derivatives with respect to triangle lengths
        df_u = -(u**2 - w**2 + v**2) / (2 * u**2 * w)
        df_w = -(w**2 - u**2 + v**2) / (2 * w**2 * u)
        df_v = v / (u * w)
        # second derivatives
        ddf_uu = (v**2 - w**2) / (u**3 * w)
        ddf_ww = (v**2 - u**2) / (w**3 * u)
        ddf_vv = 1 / (u * w)
        ddf_uv = -v / (u**2 * w)
        ddf_uw = (u**2 + w**2 + v**2) / (2 * u**2 * w**2)
        ddf_vw = -v / (w**2 * u)

        # Compond derivatives w/r to vectors
        U = df_u[_c] * nu + df_w[_c] * (-nw)
        V = df_v[_c] * nv + df_w[_c] * (+nw)

        # Second derivatives w/r to vectors
        dU_u = (
            _o(nu, ddf_uu[_c] * nu + ddf_uw[_c] * (-nw))
            + df_u[_cc] * nnu
            + _o(-nw, ddf_uw[_c] * nu + ddf_ww[_c] * (-nw))
            + df_w[_cc] * nnw
        )
        dV_v = (
            _o(nv, ddf_vv[_c] * nv + ddf_vw[_c] * nw)
            + df_v[_cc] * nnv
            + _o(nw, ddf_vw[_c] * nv + ddf_ww[_c] * nw)
            + df_w[_cc] * nnw
        )
        dU_v = (
            _o(nu, ddf_uv[_c] * nv + ddf_uw[_c] * nw)
            + _o(-nw, ddf_vw[_c] * nv + ddf_ww[_c] * nw)
            + df_w[_cc] * (-nnw)
        )

        # Scalar parts
        dh = dh(f)
        ddh = ddh(f)

        # Defining full derivatives
        ddG = np.zeros([3, r_ij_c.shape[0], r_ij_c.shape[1], r_ij_c.shape[1]])
        ddG[0] = ddh[_cc] * _o(U, U) + dh[_cc] * dU_u
        ddG[1] = ddh[_cc] * _o(V, V) + dh[_cc] * dV_v
        ddG[2] = ddh[_cc] * _o(U, V) + dh[_cc] * dU_v
        return ddG


@pytest.fixture(params=[1, 1.5, 2])
def length(request):
    return request.param


@pytest.fixture(params=[np.pi / 6, np.pi / 3, np.pi / 2])
def angle(request):
    return request.param


@pytest.fixture
def co2(length, angle):
    s = 1.1
    atoms = Atoms(
        "CO2",
        # Not symmetric on purpose
        positions=[
            [-1, 0, 0],
            [0, 0, 0],
            [s * np.cos(angle), s * np.sin(angle), 0],
        ],
        cell=[5, 5, 5],
    )

    atoms.positions[:] *= length
    return atoms


@pytest.fixture
def molecule():
    return MolecularNeighbourhood(
        Molecules(
            bonds_connectivity=[[0, 1], [1, 2]],
            angles_connectivity=[[0, 1, 2]],
        )
    )


def test_harmonic_bond(co2, molecule):
    k, r0 = 1, 0.5

    calc = NiceManybody(HarmonicBond(r0, k), ZeroAngle(), molecule)
    co2.calc = calc

    pair_vectors = np.array([
        co2.get_distance(0, 1, vector=True),
        co2.get_distance(1, 2, vector=True),
    ])

    pair_distances = np.linalg.norm(pair_vectors, axis=-1)

    # Testing potential energy
    epot = co2.get_potential_energy()
    epot_ref = np.sum(0.5 * k * (pair_distances - r0)**2)
    nt.assert_allclose(epot, epot_ref, rtol=1e-15)

    # Testing force on first atom
    f = co2.get_forces()
    f_ref = k * (pair_distances[0] - r0) * pair_vectors[0] / pair_distances[0]
    nt.assert_allclose(f[0], f_ref, rtol=1e-15)

    # Testing all forces with finite differences
    f_ref = numerical_forces(co2, d=1e-6)
    nt.assert_allclose(f, f_ref, rtol=1e-9, atol=1e-7)

    # Testing hessian
    h = co2.calc.get_property('hessian').todense()
    h_ref = numerical_hessian(co2, dx=1e-6).todense()
    nt.assert_allclose(h, h_ref, atol=1e-4)


def test_harmonic_angle(co2, molecule):
    kt, theta0 = 1, np.pi / 4
    calc = NiceManybody(ZeroBond(), HarmonicAngle(theta0, kt, co2), molecule)
    co2.calc = calc

    angle = np.pi - np.radians(co2.get_angle(0, 1, 2))

    # Testing potential energy
    epot = co2.get_potential_energy()
    epot_ref = 0.5 * kt * (angle - theta0)**2
    nt.assert_allclose(epot, epot_ref, rtol=1e-14)

    # Testing forces
    f = co2.get_forces()
    f_ref = numerical_forces(co2, d=1e-6)
    nt.assert_allclose(f, f_ref, rtol=1e-6, atol=1e-9)

    # Checking zeros
    nt.assert_allclose(np.abs(f.sum()), 0, atol=1e-13)

    # Testing hessian
    h = co2.calc.get_property('hessian')
    h_ref = numerical_hessian(co2, dx=1e-4)
    # print(f"{h}\n\n{h_ref}")
    nt.assert_allclose(h.todense(), h_ref.todense(), atol=1e-5)
