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


class ZeroPair(Manybody.Phi):
    """Implementation of zero pair interaction."""

    def __call__(self, r_p, xi_p):
        return xi_p

    def gradient(self, r_p, xi_p):
        return np.stack([
            np.zeros_like(r_p),
            np.ones_like(xi_p),
        ])

    def hessian(self, r_p, xi_p):
        return np.zeros([3] + list(r_p.shape))


class ZeroAngle(Manybody.Theta):
    """Implementation of a zero three-body interaction."""

    def __call__(self, R1, R2, R3):
        return np.zeros_like(R1)

    def gradient(self, R1, R2, R3):
        return np.zeros([3] + list(R1.shape))

    def hessian(self, R1, R2, R3):
        return np.zeros([6] + list(R1.shape))


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


@distance_defined
class StillingerWeberPair(Manybody.Phi):
    """
    Implementation of the Stillinger-Weber Potential
    """

    def __init__(self, parameters):
        # Maybe set only parameters needed for \Phi
        self.el = parameters['el']
        self.epsilon = parameters['epsilon']
        self.sigma = parameters['sigma']
        self.costheta0 = parameters['costheta0']
        self.A = parameters['A']
        self.B = parameters['B']
        self.p = parameters['p']
        self.q = parameters['q']
        self.a = parameters['a']
        self.lambda1 = parameters['lambda1']
        self.gamma = parameters['gamma']

    def __call__(self, r_p, xi_p):
        U2 = self.B * np.power(self.sigma / r_p, self.p) - np.power(self.sigma / r_p, self.q)
        U2 *= self.A * self.epsilon * np.exp(self.sigma / (r_p - self.a * self.sigma))

        return U2 + self.lambda1 * xi_p

    def gradient(self, r_p, xi_p):
        sigma_r_p = np.power(self.sigma / r_p, self.p)
        sigma_r_q = np.power(self.sigma / r_p, self.q)

        h = np.exp(self.sigma / (r_p - self.a * self.sigma))
        dh = -self.sigma / np.power(r_p - self.a * self.sigma, 2) * m

        s = self.B * sigma_r_p - sigma_r_q
        ds = -self.p * self.B * sigma_r_p / r_p + self.q * sigma_r_q / r_p

        return np.stack([
            self.A * self.epsilon * (ds * h + dh * s),
            self.lambda1 * np.ones_like(xi_p)
        ])

    def hessian(self, r_p, xi_p):
        sigma_r_p = np.power(self.sigma / r_p, self.p)
        sigma_r_q = np.power(self.sigma / r_p, self.q)

        m = np.exp(self.sigma / (r_p - self.a * self.sigma))
        dm = -self.sigma / np.power(r_p - self.a * self.sigma, 2) * m
        ddm = m * self.sigma**2 / np.power(r_p - self.a * self.sigma, 4)
        ddm += m * 2 * self.sigma / np.power(r_p - self.a * self.sigma, 3)

        h = self.B * sigma_r_p - sigma_r_q
        dh = -self.p * self.B * sigma_r_p / r_p + self.q * sigma_r_q / r_p
        ddh = self.p * self.B * sigma_r_p / r_p**2 * (1 + self.p)
        ddh -= self.q * sigma_r_q / r_p**2 * (1 + self.q)

        return np.stack([
            self.A * self.epsilon * (m * ddh + 2 * dh * dm + h * ddm),
            np.zeros_like(xi_p),
            np.zeros_like(xi_p)
        ])

@angle_distance_defined
class StillingerWeberAngle(Manybody.Theta):
    """
    Implementation of the Stillinger-Weber Potential
    """

    def __init__(self, parameters):
        # Maybe set only parameters needed for \Phi
        self.el = parameters['el']
        self.epsilon = parameters['epsilon']
        self.sigma = parameters['sigma']
        self.costheta0 = parameters['costheta0']
        self.A = parameters['A']
        self.B = parameters['B']
        self.p = parameters['p']
        self.q = parameters['q']
        self.a = parameters['a']
        self.lambda1 = parameters['lambda1']
        self.gamma = parameters['gamma']

    def __call__(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # Functions
        m = np.exp(self.gamma * self.sigma / (rij - self.a * self.sigma))
        n = np.exp(self.gamma * self.sigma / (rik - self.a * self.sigma))
        g = np.power(cos - self.costheta0, 2)

        return self.epsilon * g * m * n

    def gradient(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # Functions
        m = np.exp(self.gamma * self.sigma / (rij - self.a * self.sigma))
        n = np.exp(self.gamma * self.sigma / (rik - self.a * self.sigma))
        g = np.power(cos - self.costheta0, 2)

        # Derivative of scalar functions
        dg_dcos = 2 * (cos - self.costheta0)
        dm_drij = - self.gamma * self.sigma / np.power(rij - self.a * self.sigma, 2) * m
        dn_drik = - self.gamma * self.sigma / np.power(rik - self.a * self.sigma, 2) * n

        # Derivative of cosine 
        dg_drij = dg_dcos * (rsq_ij - rsq_ik + rsq_jk) / (2 * rsq_ij * rik)
        dg_drik = dg_dcos * (rsq_ik - rsq_ij + rsq_jk) / (2 * rsq_ik * rij)
        dg_drjk = - dg_dcos * rjk / (rij * rik)

        return self.epsilon * np.stack([
            dg_drij * m * n + dm_drij * g * n,
            dg_drik * m * n + dn_drik * g * m,
            dg_drjk * m * n
        ])

    def hessian(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # Functions
        m = np.exp(self.gamma * self.sigma / (rij - self.a * self.sigma))
        n = np.exp(self.gamma * self.sigma / (rik - self.a * self.sigma))
        g = np.power(cos - self.costheta0, 2)

        # Derivative of scalar functions
        dg_dcos = 2 * (cos - self.costheta0)
        ddg_ddcos =  2 * np.ones_like(rij)

        dm_drij = - self.gamma * self.sigma / np.power(rij - self.a * self.sigma, 2) * m
        ddm_ddrij = 2 * self.gamma * self.sigma / np.power(rij - self.a * self.sigma, 3)
        ddm_ddrij += np.power(self.gamma * self.sigma, 2) / np.power(rij - self.a * self.sigma, 4)
        ddm_ddrij *= m 

        dn_drik = - self.gamma * self.sigma / np.power(rik - self.a * self.sigma, 2) * n
        ddn_ddrik = 2 * self.gamma * self.sigma / np.power(rik - self.a * self.sigma, 3)
        ddn_ddrik += np.power(self.gamma * self.sigma, 2) / np.power(rik - self.a * self.sigma, 4)
        ddn_ddrik *= n 

        # First derivative of cos with respect to r
        dcos_drij = (rsq_ij - rsq_ik + rsq_jk) / (2 * rsq_ij * rik)
        dcos_drik = (rsq_ik - rsq_ij + rsq_jk) / (2 * rsq_ik * rij)
        dcos_drjk = - rjk / (rij * rik)

        # Second derivatives with respect to r
        ddcos_drijdrij = (rsq_ik - rsq_jk) / (rij**3 * rik)
        ddcos_drikdrik = (rsq_ij - rsq_jk) / (rik**3 * rij)
        ddcos_drjkdrjk = - 1 / (rij * rik)
        ddcos_drijdrik = - (rsq_ij + rsq_ik + rsq_jk) / (2 * rsq_ij * rsq_ik)
        ddcos_drijdrjk = rjk / (rik * rsq_ij)
        ddcos_drikdrjk = rjk / (rij * rsq_ik) 

        # First and second order derivatives of g 
        dg_drij = dg_dcos * dcos_drij
        dg_drik = dg_dcos * dcos_drik
        dg_drjk = dg_dcos * dcos_drjk
        ddg_ddrij = ddg_ddcos * dcos_drij * dcos_drij + dg_dcos * ddcos_drijdrij
        ddg_ddrik = ddg_ddcos * dcos_drik * dcos_drik + dg_dcos * ddcos_drikdrik
        ddg_ddrjk = ddg_ddcos * dcos_drjk * dcos_drjk + dg_dcos * ddcos_drjkdrjk
        ddg_drjkdrik = dcos_drik * ddg_ddcos * dcos_drjk + dg_dcos * ddcos_drikdrjk
        ddg_drjkdrij = dcos_drij * ddg_ddcos * dcos_drjk + dg_dcos * ddcos_drijdrjk
        ddg_drikdrij = dcos_drij * ddg_ddcos * dcos_drik + dg_dcos * ddcos_drijdrik

        return self.epsilon * np.stack([
            n * (ddg_ddrij * m + dg_drij * dm_drij + ddm_ddrij * g + dm_drij * dg_drij) ,
            m * (ddg_ddrik * n + dn_drik * dg_drik + ddn_ddrik * g + dn_drik * dg_drik),
            ddg_ddrjk * m * n,
            m * (ddg_drjkdrik * n + dn_drik * dg_drjk),
            n * (ddg_drjkdrij * m + dm_drij * dg_drjk),
            ddg_drikdrij * m * n + dg_drij * dn_drik * m + dm_drij * dg_drik * n + dm_drij * dn_drik * g
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
