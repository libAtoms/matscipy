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

"""Harmonic potentials for bonds and triplets."""

import numpy as np

from ase import Atoms
from ..calculator import NiceManybody


class ZeroPair(NiceManybody.G):
    """Defines a non-interacting pair potential."""

    def __call__(self, r, xi, *args):
        """Return triplet energy only."""
        return xi

    def gradient(self, r, xi, *args):
        """Return triplet interaction only."""
        return [np.zeros_like(xi), np.ones_like(xi)]

    def hessian(self, r, xi, *args):
        """Zero hessian."""
        return [np.zeros_like(r)] * 3


class ZeroTriplet(NiceManybody.G):
    """Defines a non-interacting triplet potential."""

    def __call__(self, *args):
        """Zero triplet energy."""
        return np.zeros(args[0].shape[0])

    def gradient(self, *args):
        """Zero triplet force."""
        return np.zeros([2] + list(args[0].shape))

    def hessian(self, *args):
        """Zero triplet hessian."""
        return np.zeros([3] + list(args[0].shape) + [args[0].shape[1]])


class HarmonicBond(NiceManybody.F):
    """Defines a harmonic bond."""

    def __init__(self, r0, k):
        """Initialize with equilibrium distance and stiffness."""
        self.r0 = r0
        self.k = k

    def __call__(self, r, xi, atype, ptype):
        r"""Compute spring potential energy.

        .. math:: E(r) = \frac{1}{2} k(r - r_0)^2 + \xi

        """
        e = 0.5 * self.k * (r - self.r0)**2
        e[ptype < 0] = 0  # ignore bonds from angles
        return e + xi

    def gradient(self, r, xi, atype, ptype):
        """Compute spring force."""
        g = self.k * (r - self.r0)
        g[ptype < 0] = 0
        return [g, np.ones_like(xi)]

    def hessian(self, r, xi, atype, ptype):
        """Compute spring stiffness."""
        h = np.full_like(r, self.k)
        h[ptype < 0] = 0
        return [h, np.zeros_like(r), np.zeros_like(r)]


class HarmonicAngle(NiceManybody.G):
    """Defines a harmonic angle potential."""

    def __init__(self, a0, k, atoms: Atoms):
        """Initialize with equilibrium angle and stiffness.

        Note: atoms are needed because mics are calculated for triplet
        distances. This will be removed once G is redefined to take triplet
        distances instead of vectors.
        """
        self.a0 = a0
        self.k = k
        self.atoms = atoms

    def __call__(self, r_ij_c, r_ik_c, *args):
        r"""Angle harmonic energy.

        Define the following functional form for :math:`G`:

        .. math::
            E(a) & = \frac{1}{2} k(a - a_0)^2 \\
            \vec{u} & = \vec{r_{ij}} \\
            \vec{v} & = \vec{r_{ik}} \\
            \vec{w}(\vec{u}, \vec{v}) & = \vec{r_{jk}} = \vec{v} - \vec{u} \\
            f(u, v, w) & = -\frac{u^2 + w^2 - v^2}{2uw} \\
            F(\vec{u}, \vec{v}) & = \frac{\vec{u}\cdot\vec{w}(\vec{u}, \vec{v})}{uw} \\
                                & = f(u, v, |\vec{w}(\vec{u}, \vec{v})|) \\
            h(x) & = E(\arccos(x)) \\
            G(\vec{u}, \vec{v}) & = h(F(\vec{u}, \vec{v})))
        """
        _, (r_ij, r_ik, r_jk) = self._distance_triplet(
            r_ij_c, r_ik_c, self.atoms.cell, self.atoms.pbc
        )

        a = np.arccos(-(r_ij**2 + r_jk**2 - r_ik**2) / (2 * r_ij * r_jk))
        return 0.5 * self.k * (a - self.a0)**2

    def gradient(self, r_ij_c, r_ik_c, *args):
        r"""Compute derivatives of :math:`G` w/r to :math:`r_{ij}` and :math:`r_{ik}`.

        We have the following partial derivatives:

        .. math::
            \frac{\partial G}{\partial u_i}(\vec{u}, \vec{v}) & = h'(F(\vec{u}, \vec{v})) \frac{\partial F}{\partial u_i}(\vec{u}, \vec{v}) \\
            \frac{\partial G}{\partial v_i}(\vec{u}, \vec{v}) & = h'(F(\vec{u}, \vec{v})) \frac{\partial F}{\partial v_i}(\vec{u}, \vec{v}) \\

        The partial derivatives of :math:`F` are expressed as:

        .. math::
            \frac{\partial F}{\partial u_i} = U_i & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial u_i} + \frac{\partial f}{\partial w}\frac{\partial w}{\partial u_i}\\
            \frac{\partial F}{\partial v_i} = V_i & = \frac{\partial f}{\partial v}\frac{\partial v}{\partial v_i} + \frac{\partial f}{\partial w}\frac{\partial w}{\partial v_i}

        We note the normal vectors as:

        .. math::
            \bar{u}_i & = \frac{u_i}{u}\\
            \bar{v}_i & = \frac{v_i}{v}\\
            \bar{w}_i & = \frac{w_i}{w}

        So that we can write the following partial derivatives:

        .. math::
            \frac{\partial u}{\partial u_i} & = \bar{u}_i\\
            \frac{\partial v}{\partial v_i} & = \bar{v}_i\\
            \frac{\partial w}{\partial u_i} & = -\bar{w}_i\\
            \frac{\partial w}{\partial v_i} & = \bar{w}_i

        Which gives the final expressions for :math:`U_i` and :math:`V_i`:

        .. math::
            U_i &= \frac{\partial f}{\partial u} \bar{u}_i + \frac{\partial f}{\partial w} (-\bar{w}_i)\\
            V_i &= \frac{\partial f}{\partial v} \bar{v}_i + \frac{\partial f}{\partial w} \bar{w}_i

        The remaining scalar partial derivatives are simple to derive and left
        to the reader :P .

        """
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
        r"""Compute derivatives of :math:`G` w/r to :math:`r_{ij}` and :math:`r_{ik}`.

        We have the following partial derivatives:

        .. math::
            \frac{\partial^2 G}{\partial u_i\partial u_j}(\vec{u}, \vec{v}) & = h''(F) U_i U_j + h'(F)\frac{\partial U_i}{\partial u_j}\\
            \frac{\partial^2 G}{\partial v_i\partial v_j}(\vec{u}, \vec{v}) & = h''(F) V_i V_j + h'(F)\frac{\partial V_i}{\partial v_j}\\
            \frac{\partial^2 G}{\partial u_i\partial v_j}(\vec{u}, \vec{v}) & = h''(F) U_i V_j + h'(F)\frac{\partial U_i}{\partial v_j}
        
        The derivatives of :math:`U_i` and :math:`V_i` need careful treatment:

        .. math::
            \frac{\partial U_i}{\partial u_j} = \frac{\partial}{\partial u_j}\left(\frac{\partial f}{\partial u}(u, v, w(\vec{u}, \vec{v}))\right) \frac{\partial u}{\partial u_i} + \frac{\partial f}{\partial u}\frac{\partial^2 u}{\partial u_i\partial u_j} + \frac{\partial}{\partial u_j}\left(\frac{\partial f}{\partial w}(u, v, w(\vec{u}, \vec{v}))\right) \frac{\partial w}{\partial u_i} + \frac{\partial f}{\partial w} \frac{\partial^2 w}{\partial u_i\partial u_j}\\
            \frac{\partial V_i}{\partial v_j} = \frac{\partial}{\partial v_j}\left(\frac{\partial f}{\partial v}(u, v, w(\vec{u}, \vec{v}))\right) \frac{\partial v}{\partial v_i} + \frac{\partial f}{\partial v}\frac{\partial^2 v}{\partial v_i\partial v_j} + \frac{\partial}{\partial v_j}\left(\frac{\partial f}{\partial w}(u, v, w(\vec{u}, \vec{v}))\right) \frac{\partial w}{\partial v_i} + \frac{\partial f}{\partial w} \frac{\partial^2 w}{\partial v_i\partial v_j}\\
            \frac{\partial U_i}{\partial v_j} = \frac{\partial}{\partial v_j}\left(\frac{\partial f}{\partial u}(u, v, w(\vec{u}, \vec{v}))\right) \frac{\partial u}{\partial u_i} + \frac{\partial f}{\partial u}\frac{\partial^2 u}{\partial u_i\partial v_j} + \frac{\partial}{\partial v_j}\left(\frac{\partial f}{\partial w}(u, v, w(\vec{u}, \vec{v}))\right) \frac{\partial w}{\partial u_i} + \frac{\partial f}{\partial w} \frac{\partial^2 w}{\partial u_i\partial v_j}
        
        For the simple partial derivatives in the above section, we have:

        .. math::
            \frac{\partial^2 u}{\partial u_i\partial u_j} & = \bar{\bar{u}}_{ij} = \frac{\delta_{ij} - \bar{u}_i \bar{u}_j}{u}\\
            \frac{\partial^2 v}{\partial v_i\partial v_j} & = \bar{\bar{u}}_{ij} = \frac{\delta_{ij} - \bar{v}_i \bar{v}_j}{v}\\
            \frac{\partial^2 u}{\partial u_i\partial v_j} & = 0\\
            \frac{\partial^2 w}{\partial u_i\partial u_j} & = \bar{\bar{w}}_{ij} = \frac{\delta_{ij} - \bar{w}_i \bar{w}_j}{w}\\
            \frac{\partial^2 w}{\partial v_i\partial v_j} & = \bar{\bar{w}}_{ij}\\
            \frac{\partial^2 w}{\partial u_i\partial v_j} & = -\bar{\bar{w}}_{ij}
        
        For the more complex partial derivatives:

        .. math::
            \frac{\partial}{\partial u_j}\left(\frac{\partial f}{\partial u}(u, v, w(\vec{u}, \vec{v}))\right) & = \frac{\partial^2 f}{\partial u^2} \frac{\partial u}{\partial u_j} + \frac{\partial^2 f}{\partial u\partial w}\frac{\partial w}{\partial u_j}\\
            \frac{\partial}{\partial u_j}\left(\frac{\partial f}{\partial w}(u, v, w(\vec{u}, \vec{v}))\right) & = \frac{\partial^2 f}{\partial w\partial u} \frac{\partial u}{\partial u_j} + \frac{\partial^2 f}{\partial w^2}\frac{\partial w}{\partial u_j}\\
            \frac{\partial}{\partial v_j}\left(\frac{\partial f}{\partial v}(u, v, w(\vec{u}, \vec{v}))\right) & = \frac{\partial^2 f}{\partial v^2} \frac{\partial v}{\partial v_j} + \frac{\partial^2 f}{\partial v\partial w}\frac{\partial w}{\partial v_j}\\
            \frac{\partial}{\partial v_j}\left(\frac{\partial f}{\partial w}(u, v, w(\vec{u}, \vec{v}))\right) & = \frac{\partial^2 f}{\partial w\partial v} \frac{\partial v}{\partial v_j} + \frac{\partial^2 f}{\partial w^2}\frac{\partial w}{\partial v_j}\\
            \frac{\partial}{\partial v_j}\left(\frac{\partial f}{\partial u}(u, v, w(\vec{u}, \vec{v}))\right) & = \frac{\partial^2 f}{\partial u\partial v} \frac{\partial v}{\partial v_j} + \frac{\partial^2 f}{\partial u\partial w}\frac{\partial w}{\partial v_j}\\

        The remaining scalar derivatives are left to the reader.
        """
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
