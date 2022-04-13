"""Manybody potential definitions."""

import numpy as np

from functools import wraps
from typing import Iterable
from types import SimpleNamespace
from itertools import combinations_with_replacement
from .newmb import Manybody


def distance_defined(cls):
    """
    Decorate class to help potential definition from distance.

    Transforms a potential defined with the distance to
    one defined by the distance squared.
    """
    old = SimpleNamespace()
    old.__call__ = cls.__call__
    old.gradient = cls.gradient
    old.hessian = cls.hessian

    @wraps(cls.__call__)
    def call(self, rsq_p, xi_p):
        return old.__call__(self, np.sqrt(rsq_p), xi_p)

    @wraps(cls.gradient)
    def gradient(self, rsq_p, xi_p):
        r_p = np.sqrt(rsq_p)
        res = old.gradient(self, r_p, xi_p)
        res[0] *= 1 / (2 * r_p)
        return res

    @wraps(cls.hessian)
    def hessian(self, rsq_p, xi_p):
        r_p = np.sqrt(rsq_p)
        e_p = 1 / (2 * r_p)
        grad = old.gradient(self, r_p, xi_p)
        hess = old.hessian(self, r_p, xi_p)

        # Correcting double R derivative
        hess[0] *= e_p**2
        hess[0] += grad[0] * (-1 / 4) * rsq_p**(-3 / 2)

        # Correcting mixed derivative
        hess[2] *= e_p

        return hess

    cls.__call__ = call
    cls.gradient = gradient
    cls.hessian = hessian
    return cls

def angle_distance_defined(cls):
    """
    Decorate class to help potential definition from distance.

    Transforms a potential defined with the distance to
    one defined by the distance squared.
    """
    old = SimpleNamespace()
    old.__call__ = cls.__call__
    old.gradient = cls.gradient
    old.hessian = cls.hessian

    @wraps(cls.__call__)
    def call(self, rsq_ij, rsq_ik, rsq_jk):
        return old.__call__(self, np.sqrt(rsq_ij), np.sqrt(rsq_ik), np.sqrt(rsq_jk))

    @wraps(cls.gradient)
    def gradient(self, rsq_ij, rsq_ik, rsq_jk):
        rij = np.sqrt(rsq_ij)
        rik = np.sqrt(rsq_ik)
        rjk = np.sqrt(rsq_jk)

        grad = old.gradient(self, rij, rik, rjk)
        grad[0] *= 1 / (2 * rij)
        grad[1] *= 1 / (2 * rik)
        grad[2] *= 1 / (2 * rjk)

        return grad

    @wraps(cls.hessian)
    def hessian(self, rsq_ij, rsq_ik, rsq_jk):
        rij = np.sqrt(rsq_ij)
        rik = np.sqrt(rsq_ik)
        rjk = np.sqrt(rsq_jk)
        grad = old.gradient(self, rij, rik, rjk)
        hess = old.hessian(self, rij, rik, rjk)

        # Correction due to derivatives with respect to rsq
        hess[0] = hess[0] * (1 / (4 * rsq_ij)) - grad[0] * (1 / (4 * rij**3))
        hess[1] = hess[1] * (1 / (4 * rsq_ik)) - grad[1] * (1 / (4 * rik**3))
        hess[2] = hess[2] * (1 / (4 * rsq_jk)) - grad[2] * (1 / (4 * rjk**3))
        hess[3] = hess[3] * (1 / (4 * rik * rjk))
        hess[4] = hess[4] * (1 / (4 * rij * rjk))
        hess[5] = hess[5] * (1 / (4 * rij * rik))

        return hess

    cls.__call__ = call
    cls.gradient = gradient
    cls.hessian = hessian
    return cls


@distance_defined
class HarmonicPair(Manybody.Phi):
    """
    Implementation of a harmonic pair interaction.
    """

    def __init__(self, K=1, r0=0):
        self.K = K
        self.r0 = r0

    def __call__(self, r_p, xi_p):
        return 0.5 * self.K * (r_p - self.r0)**2 + xi_p

    def gradient(self, r_p, xi_p):
        return np.stack([
            self.K * (r_p - self.r0),
            np.ones_like(xi_p),
        ])

    def hessian(self, r_p, xi_p):
        return np.stack([
            np.full_like(r_p, self.K),
            np.zeros_like(xi_p),
            np.zeros_like(xi_p),
        ])

@angle_distance_defined
class HarmonicAngle(Manybody.Theta):
    """
    Implementation of a harmonic angle interaction.
    """

    def __init__(self, k0=1, theta0=np.pi/2):
        self.k0 = k0
        self.theta0 = theta0

    def __call__(self, rij, rik, rjk):
        r"""
        Angle harmonic energy.
        """
        f = np.arccos((rij**2 + rik**2 - rjk**2) / (2 * rij * rik))

        return 0.5 * self.k0 * (f - self.theta0)**2

    def gradient(self, rij, rik, rjk):
        r"""
        First order derivatives of :math:`\Theta` w/r to :math:`r_{ij}, r_{ik}, r_{jk}`
        """
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        f = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # derivatives with respect to r
        df_drij = (rsq_ij - rsq_ik + rsq_jk) / (2 * rsq_ij * rik)
        df_drik = (rsq_ik - rsq_ij + rsq_jk) / (2 * rsq_ik * rij)
        df_drjk = - rjk / (rij * rik)

        # Scalar derivatives
        def E(a):
            return self.k0 * (a - self.theta0)

        def h(f):
            with np.errstate(divide="raise"):
                d_arccos = -1 / np.sqrt(1 - f**2)
            return E(np.arccos(f)) * d_arccos

        return h(f) * np.stack([df_drij, df_drik, df_drjk])

    def hessian(self, rij, rik, rjk):
        r"""
        Second order derivatives of :math:`\Theta` w/r to :math:`r_{ij}, r_{ik}, r_{jk}`
        """
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        f = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # first derivatives with respect to r
        df_drij = (rsq_ij - rsq_ik + rsq_jk) / (2 * rsq_ij * rik)
        df_drik = (rsq_ik - rsq_ij + rsq_jk) / (2 * rsq_ik * rij)
        df_drjk = - rjk / (rij * rik)

        # second derivatives with respect to r
        ddf_drijdrij = (rsq_ik - rsq_jk) / (rij**3 * rik)
        ddf_drikdrik = (rsq_ij - rsq_jk) / (rik**3 * rij)
        ddf_drjkdrjk = - 1 / (rij * rik)
        ddf_drijdrik = - (rsq_ij + rsq_ik + rsq_jk) / (2 * rsq_ij * rsq_ik)
        ddf_drijdrjk = rjk / (rik * rsq_ij)
        ddf_drikdrjk = rjk / (rij * rsq_ik)

        # Scalar functions
        dE = lambda a: self.k0 * (a - self.theta0)
        ddE = lambda a: self.k0

        darcos = lambda x: -1 / np.sqrt(1 - x**2)
        ddarcos = lambda x: -x / (1 - x**2)**(3/2)

        # Scalar derivative of theta
        dtheta_dx = dE(np.arccos(f)) * darcos(f)
        ddtheta_dxdx = (
            ddE(np.arccos(f)) * darcos(f)**2 + dE(np.arccos(f)) * ddarcos(f)
        )

        return np.stack([
            ddtheta_dxdx * df_drij * df_drij + dtheta_dx * ddf_drijdrij,
            ddtheta_dxdx * df_drik * df_drik + dtheta_dx * ddf_drikdrik,
            ddtheta_dxdx * df_drjk * df_drjk + dtheta_dx * ddf_drjkdrjk,
            ddtheta_dxdx * df_drjk * df_drik + dtheta_dx * ddf_drikdrjk,
            ddtheta_dxdx * df_drjk * df_drij + dtheta_dx * ddf_drijdrjk,
            ddtheta_dxdx * df_drik * df_drij + dtheta_dx * ddf_drijdrik,
        ])


try:
    from sympy import lambdify, Expr, Symbol

    def _l(*args):
        return lambdify(*args, 'numpy')

    def _extend(symfunc, sample_array):
        """Extend array in case sympy returns litteral."""
        def f(*args):
            res = symfunc(*args)
            if not isinstance(res, np.ndarray):
                return np.full_like(sample_array, res, dtype=sample_array.dtype)
            return res
        return f

    class SymPhi(Manybody.Phi):
        """Pair potential from Sympy symbolic expression."""

        def __init__(self, energy_expression: Expr, symbols: Iterable[Symbol]):
            assert len(symbols) == 2, "Expression should only have 2 symbols"

            self.e = energy_expression

            # Lambdifying expression for energy and gradient
            self.phi = _l(symbols, self.e)
            self.dphi = [_l(symbols, self.e.diff(v)) for v in symbols]

            # Pairs of symbols for 2nd-order derivatives
            dvars = list(combinations_with_replacement(symbols, 2))
            dvars = [dvars[i] for i in (0, 2, 1)]  # arrange order

            # Lambdifying hessian
            self.ddphi = [_l(symbols, self.e.diff(*v)) for v in dvars]

        def __call__(self, rsq_p, xi_p):
            return _extend(self.phi, rsq_p)(rsq_p, xi_p)

        def gradient(self, rsq_p, xi_p):
            return np.stack([
                _extend(f, rsq_p)(rsq_p, xi_p)
                for f in self.dphi
            ])

        def hessian(self, rsq_p, xi_p):
            return np.stack([
                _extend(f, rsq_p)(rsq_p, xi_p)
                for f in self.ddphi
            ])

    class SymTheta(Manybody.Theta):
        """Three-body potential from Sympy symbolic expression."""

        def __init__(self, energy_expression: Expr, symbols: Iterable[Symbol]):
            assert len(symbols) == 3, "Expression should only have 3 symbols"

            self.e = energy_expression

            # Lambdifying expression for energy and gradient
            self.theta = _l(symbols, self.e)
            self.dtheta = [_l(symbols, self.e.diff(v)) for v in symbols]

            # Pairs of symbols for 2nd-order derivatives
            dvars = list(combinations_with_replacement(symbols, 2))
            dvars = [dvars[i] for i in (0, 3, 5, 4, 2, 1)]  # arrange order

            self.ddtheta = [_l(symbols, self.e.diff(*v)) for v in dvars]

        def __call__(self, R1_t, R2_t, R3_t):
            return _extend(self.theta, R1_t)(R1_t, R2_t, R3_t)

        def gradient(self, R1_t, R2_t, R3_t):
            return np.stack([
                _extend(f, R1_t)(R1_t, R2_t, R3_t)
                for f in self.dtheta
            ])

        def hessian(self, R1_t, R2_t, R3_t):
            return np.stack([
                _extend(f, R1_t)(R1_t, R2_t, R3_t)
                for f in self.ddtheta
            ])

except ImportError:
    pass
