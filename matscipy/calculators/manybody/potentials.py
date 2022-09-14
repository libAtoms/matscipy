"""Manybody potential definitions."""

import numpy as np

from functools import wraps
from typing import Iterable
from types import SimpleNamespace
from itertools import combinations_with_replacement
from .newmb import Manybody
from ..pair_potential import LennardJonesCut


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
class SimplePairNoMix(Manybody.Phi):
    """
    Implementation of a harmonic pair interaction.
    """

    def __call__(self, r_p, xi_p):
        return 0.5 * r_p**2 + 0.5 * xi_p**2

    def gradient(self, r_p, xi_p):
        return np.stack([
            r_p,
            xi_p,
        ])

    def hessian(self, r_p, xi_p):
        return np.stack([
            np.ones_like(r_p),
            np.ones_like(xi_p),
            np.zeros_like(r_p),
        ])

@distance_defined
class SimplePairNoMixNoSecond(Manybody.Phi):
    """
    Implementation of a harmonic pair interaction.
    """

    def __call__(self, r_p, xi_p):
        return 0.5 * r_p**2 + xi_p

    def gradient(self, r_p, xi_p):
        return np.stack([
            r_p,
            np.ones_like(xi_p),
        ])

    def hessian(self, r_p, xi_p):
        return np.stack([
            np.ones_like(r_p),
            np.zeros_like(xi_p),
            np.zeros_like(r_p),
        ])



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
class LennardJones(Manybody.Phi):
    """Implementation of LennardJones potential."""

    def __init__(self, epsilon=1, sigma=1, cutoff=np.inf):
        self.lj = LennardJonesCut(epsilon, sigma, cutoff)

    def __call__(self, r, xi):
        return self.lj(r) + xi

    def gradient(self, r, xi):
        return np.stack([
            self.lj.first_derivative(r),
            np.ones_like(xi),
        ])

    def hessian(self, r, xi):
        return np.stack([
            self.lj.second_derivative(r),
            np.zeros_like(xi),
            np.zeros_like(xi),
        ])


@distance_defined
class BornMayerCut(Manybody.Phi):
    """
    Implementation of the Born-Mayer potential.
    Energy is shifted to zero at the cutoff
    """

    def __init__(self, A=1, C=1, D=1, sigma=1, rho=1, cutoff=np.inf):
        self.A, self.C, self.D = A, C, D
        self.sigma = sigma
        self.rho = rho
        self.cutoff = cutoff
        self.offset = (
            self.A * np.exp((self.sigma - cutoff) / self.rho)
            - self.C / cutoff**6
            + self.D / cutoff**8
        )

    def __call__(self, r, xi):
        return (
            self.A * np.exp((self.sigma - r) / self.rho)
            - self.C / r**6
            + self.D / r**8
            - self.offset + xi
        )

    def gradient(self, r, xi):
        return np.stack([
            (-self.A / self.rho) * np.exp((self.sigma - r) / self.rho)
            + 6 * self.C / r**7
            - 8 * self.D / r**9,
            np.ones_like(xi),
        ])

    def hessian(self, r, xi):
        return np.stack([
            (self.A / self.rho**2) * np.exp((self.sigma - r) / self.rho)
            - 42 * self.C / r**8
            + 72 * self.D / r**10,
            np.zeros_like(xi),
            np.zeros_like(xi),
        ])


@distance_defined
class StillingerWeberPair(Manybody.Phi):
    """
    Implementation of the Stillinger-Weber Potential
    """

    def __init__(self, parameters, cutoff=None):
        # Maybe set only parameters needed for \Phi
        self.ref = parameters['__ref__']
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
        self.cutoff = (
            parameters['a'] * parameters['sigma']
            if cutoff is None else cutoff
        )

    def __call__(self, r_p, xi_p):
        s = (
            self.B * np.power(self.sigma / r_p, self.p)
            - np.power(self.sigma / r_p, self.q)
        )

        h = np.exp(self.sigma / (r_p - self.a * self.sigma))

        return np.where(
            r_p <= self.cutoff,
            self.A * self.epsilon * s * h + self.lambda1 * xi_p,
            0.0,
        )

    def gradient(self, r_p, xi_p):
        sigma_r_p = np.power(self.sigma / r_p, self.p)
        sigma_r_q = np.power(self.sigma / r_p, self.q)

        h = np.exp(self.sigma / (r_p - self.a * self.sigma))
        dh = -self.sigma / np.power(r_p - self.a * self.sigma, 2) * h

        s = self.B * sigma_r_p - sigma_r_q
        ds = -self.p * self.B * sigma_r_p / r_p + self.q * sigma_r_q / r_p

        return np.where(
            r_p <= self.cutoff,
            np.stack([
                self.A * self.epsilon * (ds * h + dh * s),
                self.lambda1 * np.ones_like(xi_p)
            ]),
            0.0,
        )

    def hessian(self, r_p, xi_p):
        sigma_r_p = np.power(self.sigma / r_p, self.p)
        sigma_r_q = np.power(self.sigma / r_p, self.q)

        h = np.exp(self.sigma / (r_p - self.a * self.sigma))
        dh = -self.sigma / np.power(r_p - self.a * self.sigma, 2) * h
        ddh = h * self.sigma**2 / np.power(r_p - self.a * self.sigma, 4)
        ddh += h * 2 * self.sigma / np.power(r_p - self.a * self.sigma, 3)

        s = self.B * sigma_r_p - sigma_r_q
        ds = -self.p * self.B * sigma_r_p / r_p + self.q * sigma_r_q / r_p
        dds = self.p * self.B * sigma_r_p / r_p**2 * (1 + self.p)
        dds -= self.q * sigma_r_q / r_p**2 * (1 + self.q)

        return np.where(
            r_p <= self.cutoff,
            np.stack([
                self.A * self.epsilon * (h * dds + 2 * ds * dh + s * ddh),
                np.zeros_like(xi_p),
                np.zeros_like(xi_p)
            ]),
            0.0,
        )


@angle_distance_defined
class StillingerWeberAngle(Manybody.Theta):
    """
    Implementation of the Stillinger-Weber Potential
    """

    def __init__(self, parameters):
        # Maybe set only parameters needed for \Phi
        self.ref = parameters['__ref__']
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
        self.cutoff = parameters['a'] * parameters['sigma']

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
        g = np.power(cos + self.costheta0, 2)

        return np.where(rik <= self.cutoff, self.epsilon * g * m * n, 0.0)

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
        g = np.power(cos + self.costheta0, 2)

        # Derivative of scalar functions
        dg_dcos = 2 * (cos + self.costheta0)
        dm_drij = - self.gamma * self.sigma / np.power(rij - self.a * self.sigma, 2) * m
        dn_drik = - self.gamma * self.sigma / np.power(rik - self.a * self.sigma, 2) * n

        # Derivative of cosine
        dg_drij = dg_dcos * (rsq_ij - rsq_ik + rsq_jk) / (2 * rsq_ij * rik)
        dg_drik = dg_dcos * (rsq_ik - rsq_ij + rsq_jk) / (2 * rsq_ik * rij)
        dg_drjk = - dg_dcos * rjk / (rij * rik)

        return self.epsilon * np.where(rik <= self.cutoff,
            np.stack([
                dg_drij * m * n + dm_drij * g * n,
                dg_drik * m * n + dn_drik * g * m,
                dg_drjk * m * n
            ]), 0.0)

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
        g = np.power(cos + self.costheta0, 2)

        # Derivative of scalar functions
        dg_dcos = 2 * (cos + self.costheta0)
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

        return self.epsilon * np.where(rik <= self.cutoff,
            np.stack([
                n * (ddg_ddrij * m + dg_drij * dm_drij + ddm_ddrij * g + dm_drij * dg_drij) ,
                m * (ddg_ddrik * n + dn_drik * dg_drik + ddn_ddrik * g + dn_drik * dg_drik),
                ddg_ddrjk * m * n,
                m * (ddg_drjkdrik * n + dn_drik * dg_drjk),
                n * (ddg_drjkdrij * m + dm_drij * dg_drjk),
                ddg_drikdrij * m * n + dg_drij * dn_drik * m + dm_drij * dg_drik * n + dm_drij * dn_drik * g
        ]), 0.0)


@distance_defined
class KumagaiPair(Manybody.Phi):
    """
    Implementation of Phi for the Kumagai potential
    """

    def __init__(self, parameters):
        # Maybe set only parameters needed for \Phi
        self.ref = parameters['__ref__']
        self.el = parameters['el']
        self.A = parameters["A"]
        self.B = parameters["B"]
        self.lambda_1 = parameters["lambda_1"]
        self.lambda_2 = parameters["lambda_2"]
        self.eta = parameters["eta"]
        self.delta = parameters["delta"]
        self.alpha = parameters["alpha"]
        self.c_1 = parameters["c_1"]
        self.c_2 = parameters["c_2"]
        self.c_3 = parameters["c_3"]
        self.c_4 = parameters["c_4"]
        self.c_5 = parameters["c_5"]
        self.h = parameters["h"]
        self.R_1 = parameters["R_1"]
        self.R_2 = parameters["R_2"]

    def __call__(self, r_p, xi_p):
        # Cutoff
        fc = np.where(r_p <= self.R_1, 1.0,
                np.where(r_p >= self.R_2, 0.0,
                    1/2 + 9 / 16 * np.cos(np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                    - 1 / 16 * np.cos(3 * np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                )
            )

        fr = self.A * np.exp(-self.lambda_1 * r_p)

        fa = -self.B * np.exp(-self.lambda_2 * r_p)

        b = 1 / np.power(1 + xi_p**self.eta, self.delta)

        return fc * (fr + b * fa)

    def gradient(self, r_p, xi_p):
        # Cutoff function
        fc = np.where(r_p <= self.R_1, 1.0,
                np.where(r_p >= self.R_2, 0.0,
                    1/2 + 9 / 16 * np.cos(np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                    - 1 / 16 * np.cos(3 * np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                )
            )

        dfc = np.where(r_p <= self.R_1, 0.0,
                np.where(r_p >= self.R_2, 0.0,
                    3 * np.pi / (16 * (self.R_2 - self.R_1)) * (
                        np.sin(3 * np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                        - 3 * np.sin(np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                        )
                )
            )

        # Repulsive and attractive
        fr = self.A * np.exp(-self.lambda_1 * r_p)
        dfr = - self.lambda_1 * fr

        fa = -self.B * np.exp(-self.lambda_2 * r_p)
        dfa = - self.lambda_2 * fa

        # Bond-order expression
        b = 1 / np.power(1 + xi_p**self.eta, self.delta)
        db = - self.delta * self.eta * np.power(xi_p, self.eta - 1) * (1 + xi_p**self.eta)**(-self.delta - 1)

        return np.stack([
            dfc * (fr + b * fa) + fc * (dfr + b * dfa),
            fc * fa * db
            ])

    def hessian(self, r_p, xi_p):
        # Cutoff function
        fc = np.where(r_p <= self.R_1, 1.0,
                np.where(r_p >= self.R_2, 0.0,
                    1/2 + 9 / 16 * np.cos(np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                    - 1 / 16 * np.cos(3 * np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                )
            )

        dfc = np.where(r_p <= self.R_1, 0.0,
                np.where(r_p >= self.R_2, 0.0,
                    3 * np.pi / (16 * (self.R_2 - self.R_1)) * (
                        np.sin(3 * np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                        - 3 * np.sin(np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                        )
                )
            )

        ddfc = np.where(r_p <= self.R_1, 0.0,
                np.where(r_p >= self.R_2, 0.0,
                    9 * np.pi**2 / (16 * np.power(self.R_2 - self.R_1, 2)) * (
                        np.cos(3 * np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                        - np.cos(np.pi * (r_p - self.R_1) / (self.R_2 - self.R_1))
                        )
                )
            )

        # Repulsive and attractive
        fr = self.A * np.exp(-self.lambda_1 * r_p)
        dfr = - self.lambda_1 * fr
        ddfr = self.lambda_1**2 * fr

        fa = -self.B * np.exp(-self.lambda_2 * r_p)
        dfa = - self.lambda_2 * fa
        ddfa = self.lambda_2**2 * fa

        # Bond-order expression
        b = 1 / np.power(1 + xi_p**self.eta, self.delta)
        db = - self.delta * self.eta * np.power(xi_p, self.eta - 1) * (1 + xi_p**self.eta)**(-self.delta - 1)
        ddb = np.power(xi_p, 2 * self.eta - 2) * (self.eta * self.delta + 1)
        if self.eta != 1.0:
            ddb -= np.power(xi_p, self.eta - 2) * (self.eta -1)
        ddb *= self.delta * self.eta * np.power(1 + xi_p**self.eta, -self.delta - 2)

        return np.stack([
            ddfc * (fr + b * fa) + 2 * dfc * (dfr + b * dfa) + fc * (ddfr + b * ddfa),
            fc * fa * ddb,
            dfc * fa * db + fc * dfa * db
        ])


@angle_distance_defined
class KumagaiAngle(Manybody.Theta):
    """
    Implementation of Theta for the Kumagai potential
    """

    def __init__(self, parameters):
        # Maybe set only parameters needed for \Phi
        self.ref = parameters['__ref__']
        self.el = parameters['el']
        self.A = parameters["A"]
        self.B = parameters["B"]
        self.lambda_1 = parameters["lambda_1"]
        self.lambda_2 = parameters["lambda_2"]
        self.eta = parameters["eta"]
        self.delta = parameters["delta"]
        self.alpha = parameters["alpha"]
        self.beta = parameters["beta"]
        self.c_1 = parameters["c_1"]
        self.c_2 = parameters["c_2"]
        self.c_3 = parameters["c_3"]
        self.c_4 = parameters["c_4"]
        self.c_5 = parameters["c_5"]
        self.h = parameters["h"]
        self.R_1 = parameters["R_1"]
        self.R_2 = parameters["R_2"]

    def __call__(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # Cutoff
        fc = np.where(rik <= self.R_1, 1.0,
                np.where(rik >= self.R_2, 0.0,
                    (1/2 + (9 / 16) * np.cos(np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                    - (1 / 16) * np.cos(3 * np.pi * (rik - self.R_1) / (self.R_2 - self.R_1)))
                )
            )

        # Functions
        m = np.exp(self.alpha * np.power(rij - rik, self.beta))

        g0 = (self.c_2 * np.power(self.h - cos, 2)) / (self.c_3 + np.power(self.h - cos, 2))
        ga = 1 + self.c_4 * np.exp(-self.c_5 * np.power(self.h - cos, 2))
        g = self.c_1 + g0 * ga

        return fc * g * m

    def gradient(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # First derivative of cos with respect to r
        dcos_drij = (rsq_ij - rsq_ik + rsq_jk) / (2 * rsq_ij * rik)
        dcos_drik = (rsq_ik - rsq_ij + rsq_jk) / (2 * rsq_ik * rij)
        dcos_drjk = - rjk / (rij * rik)

        # Cutoff
        fc = np.where(rik <= self.R_1, 1.0,
                np.where(rik >= self.R_2, 0.0,
                    (1/2 + (9 / 16) * np.cos(np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                    - (1 / 16) * np.cos(3 * np.pi * (rik - self.R_1) / (self.R_2 - self.R_1)))
                )
            )

        dfc = np.where(rik <= self.R_1, 0.0,
                np.where(rik >= self.R_2, 0.0,
                    3 * np.pi / (16 * (self.R_2 - self.R_1)) * (
                        np.sin(3 * np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                        - 3 * np.sin(np.pi * (rik - self.R_1) / (self.R_2 - self.R_1)))
                )
            )

        # Functions
        m = np.exp(self.alpha * np.power(rij - rik, self.beta))
        dm_drij = self.alpha * self.beta * np.power(rij - rik, self.beta - 1) * m
        dm_drik = - dm_drij

        g0 = (self.c_2 * np.power(self.h - cos, 2)) / (self.c_3 + np.power(self.h - cos, 2))
        dg0_dcos =  (-2 * self.c_2 * self.c_3 * (self.h - cos)) / np.power(self.c_3 + np.power(self.h - cos, 2), 2)

        ga = 1 + self.c_4 * np.exp(-self.c_5 * np.power(self.h - cos, 2))
        dga_dcos = 2 * self.c_4 * self.c_5 * (self.h - cos) * np.exp(-self.c_5 * np.power(self.h - cos, 2))

        g = self.c_1 + g0 * ga
        dg_dcos = dg0_dcos * ga + g0 * dga_dcos

        dg_drij = dg_dcos * dcos_drij
        dg_drik = dg_dcos * dcos_drik
        dg_drjk = dg_dcos * dcos_drjk

        return np.stack([
            fc * dg_drij * m + fc * g * dm_drij,
            dfc * g * m + fc * dg_drik * m + fc * g * dm_drik,
            fc * dg_drjk * m
            ])

    def hessian(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

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

        # Cutoff
        fc = np.where(rik <= self.R_1, 1.0,
                np.where(rik > self.R_2, 0.0,
                    1/2 + 9 / 16 * np.cos(np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                    - 1 / 16 * np.cos(3 * np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                )
            )

        dfc = np.where(rik <= self.R_1, 0.0,
                np.where(rik >= self.R_2, 0.0,
                    3 * np.pi / (16 * (self.R_2 - self.R_1)) * (
                        np.sin(3 * np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                        - 3 * np.sin(np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                        )
                )
            )

        ddfc = np.where(rik <= self.R_1, 0.0,
                np.where(rik > self.R_2, 0.0,
                    9 * np.pi**2 / (16 * np.power(self.R_2 - self.R_1, 2)) * (
                        np.cos(3 * np.pi * (rik - self.R_1) / (self.R_2 - self.R_1)) -
                        np.cos(np.pi * (rik - self.R_1) / (self.R_2 - self.R_1))
                        )
                )
            )

        # Functions
        m = np.exp(self.alpha * np.power(rij - rik, self.beta))
        dm_drij = self.alpha * self.beta * np.power(rij - rik, self.beta - 1) * m
        dm_drik = - dm_drij
        ddm_ddrij = np.power(self.alpha * self.beta * np.power(rij - rik, self.beta - 1), 2)
        if self.beta != 1.0:
            ddm_ddrij += self.alpha * self.beta * (self.beta - 1) * np.power(rij - rik, self.beta - 2)
        ddm_ddrij *= m
        ddm_ddrik = ddm_ddrij
        ddm_drijdrik = - ddm_ddrij


        # New
        g0 = (self.c_2 * np.power(self.h - cos, 2)) / (self.c_3 + np.power(self.h - cos, 2))
        dg0_dcos =  (-2 * self.c_2 * self.c_3 * (self.h - cos)) / np.power(self.c_3 + np.power(self.h - cos, 2), 2)
        ddg0_ddcos = 2 * self.c_2 * self.c_3  * (self.c_3 - 3 * np.power(self.h - cos, 2)) / np.power(self.c_3 + np.power(self.h - cos, 2), 3)

        ga = 1 + self.c_4 * np.exp(-self.c_5 * np.power(self.h - cos, 2))
        dga_dcos = 2 * self.c_4 * self.c_5 * (self.h - cos) * np.exp(-self.c_5 * np.power(self.h - cos, 2))
        ddga_ddcos = 2 * self.c_5 * np.power(self.h - cos, 2) - 1
        ddga_ddcos *= 2 * self.c_4 * self.c_5 * np.exp(-self.c_5 * np.power(self.h - cos, 2))

        g = self.c_1 + g0 * ga
        dg_dcos = dg0_dcos * ga + g0 * dga_dcos
        ddg_ddcos = ddg0_ddcos * ga + 2 * dg0_dcos * dga_dcos + g0 * ddga_ddcos

        dg_drij = dg_dcos * dcos_drij
        dg_drik = dg_dcos * dcos_drik
        dg_drjk = dg_dcos * dcos_drjk

        ddg_drijdrij = ddg_ddcos * dcos_drij * dcos_drij + dg_dcos * ddcos_drijdrij
        ddg_drikdrik = ddg_ddcos * dcos_drik * dcos_drik + dg_dcos * ddcos_drikdrik
        ddg_drjkdrjk = ddg_ddcos * dcos_drjk * dcos_drjk + dg_dcos * ddcos_drjkdrjk
        ddg_drikdrjk = ddg_ddcos * dcos_drik * dcos_drjk + dg_dcos * ddcos_drikdrjk
        ddg_drijdrjk = ddg_ddcos * dcos_drij * dcos_drjk + dg_dcos * ddcos_drijdrjk
        ddg_drijdrik = ddg_ddcos * dcos_drij * dcos_drik + dg_dcos * ddcos_drijdrik

        return np.stack([
            fc * (ddg_drijdrij * m + dg_drij * dm_drij + dg_drij * dm_drij + g * ddm_ddrij),
            ddfc * g * m + dfc * dg_drik * m + dfc * g * dm_drik + \
            dfc * dg_drik * m + fc * ddg_drikdrik * m + fc * dg_drik * dm_drik + \
            dfc * g * dm_drik + fc * dg_drik * dm_drik + fc * g * ddm_ddrik,
            fc * ddg_drjkdrjk * m,
            dfc * dg_drjk * m + fc * ddg_drikdrjk * m + fc * dg_drjk * dm_drik,
            fc * ddg_drijdrjk * m + fc * dg_drjk * dm_drij,
            dfc * dg_drij * m + fc * ddg_drijdrik * m + fc * dg_drij * dm_drik + \
            dfc * g * dm_drij + fc * dg_drik * dm_drij + fc * g * ddm_drijdrik
            ])

@distance_defined
class TersoffBrennerPair(Manybody.Phi):
    """
    Implementation of Phi for Tersoff-Brenner potentials
    """

    def __init__(self, parameters):
        self.ref = parameters['__ref__']
        self.style = parameters['style'].lower()
        self.el = parameters['el']
        self.c = np.array(parameters['c'])
        self.d = np.array(parameters['d'])
        self.h = np.array(parameters['h'])
        self.R1 = np.array(parameters['R1'])
        self.R2 = np.array(parameters['R2'])

        if self.style == 'tersoff':
            # These are Tersoff-style parameters. The symbols follow the notation in
            # Tersoff J., Phys. Rev. B 39, 5566 (1989)
            #
            # In particular, pair terms are characterized by A, B, lam, mu and parameters for the three body terms ijk
            # depend only on the type of atom i
            self.A = np.array(parameters['A'])
            self.B = np.array(parameters['B'])
            self.lambda1 = np.array(parameters['lambda1'])
            self.mu = np.array(parameters['mu'])
            self.beta = np.array(parameters['beta'])
            self.lambda3 = np.array(parameters['lambda3'])
            self.chi = np.array(parameters['chi'])
            self.n = np.array(parameters['n'])

        elif self.style == 'brenner':
            # These are Brenner/Erhart-Albe-style parameters. The symbols follow the notation in
            # Brenner D., Phys. Rev. B 42, 9458 (1990) and
            # Erhart P., Albe K., Phys. Rev. B 71, 035211 (2005)
            #
            # In particular, pairs terms are characterized by D0, S, beta, r0, the parameters n, chi are always unity and
            # parameters for the three body terms ijk depend on the type of the bond ij
            _D0 = np.array(parameters['D0'])
            _S = np.array(parameters['S'])
            _r0 = np.array(parameters['r0'])
            _beta = np.array(parameters['beta'])
            _mu = np.array(parameters['mu'])
            gamma = np.array(parameters['gamma'])

            # Convert to Tersoff parameters
            self.lambda3 = 2 * _mu
            self.lam = _beta * np.sqrt(2 * _S)
            self.mu = _beta * np.sqrt(2 / _S)
            self.A = _D0 / (_S - 1) * np.exp(lam * _r0)
            self.B = _S * _D0 / (_S - 1) * np.exp(mu * _r0)

        else:
            raise ValueError(f'Unknown parameter style {self.style}')

    def __call__(self, r_p, xi_p):
        # Cutoff function
        fc = np.where(r_p <= self.R1, 1.0,
                np.where(r_p > self.R2, 0.0,
                    (1 + np.cos(np.pi * (r_p - self.R1) / (self.R2 - self.R1))) / 2
                )
            )

        # Attractive interaction
        fa = -self.B * np.exp(-self.mu * r_p)

        # Repulsive interaction
        fr = self.A * np.exp(-self.lambda1 * r_p)

        # Bond-order parameter
        if self.style == 'tersoff':
            b = self.chi / np.power(1 + np.power(self.beta * xi_p, self.n), 1 / (2 * self.n))

        else:
            raise ValueError(f'Brenner not implemented {self.style}')

        return fc * (fr + b * fa)

    def gradient(self, r_p, xi_p):
        # Cutoff function
        fc = np.where(r_p <= self.R1, 1.0,
                np.where(r_p > self.R2, 0.0,
                    (1 + np.cos(np.pi * (r_p - self.R1) / (self.R2 - self.R1))) / 2
                )
            )

        dfc = np.where(r_p <= self.R1, 0.0,
                np.where(r_p > self.R2, 0.0,
                    -np.pi / (2 * (self.R2 - self.R1)) * np.sin(np.pi * (r_p - self.R1) / (self.R2 - self.R1))
                )
            )

        # Attractive interaction
        fa = -self.B * np.exp(-self.mu * r_p)
        dfa = -self.mu * fa

        # Repulsive interaction
        fr = self.A * np.exp(-self.lambda1 * r_p)
        dfr = -self.lambda1 * fr

        # Bond-order parameter
        if self.style == 'tersoff':
            b = self.chi * np.power(1 + np.power(self.beta * xi_p, self.n), -1 / (2 * self.n))
            db = -0.5 * self.beta * self.chi * np.power(self.beta * xi_p, self.n - 1)
            db *= 1 / np.power(1 + np.power(self.beta * xi_p, self.n), 1 + 1 / (2 * self.n))

        else:
            raise ValueError(f'Brenner not implemented {self.style}')

        return np.stack([
            dfc * (fr + b * fa) + fc * (dfr + b * dfa),
            fc * fa * db
            ])

    def hessian(self, r_p, xi_p):
        # Cutoff function
        fc = np.where(r_p <= self.R1, 1.0,
                np.where(r_p > self.R2, 0.0,
                    (1 + np.cos(np.pi * (r_p - self.R1) / (self.R2 - self.R1))) / 2
                )
            )

        dfc = np.where(r_p <= self.R1, 0.0,
                np.where(r_p > self.R2, 0.0,
                    -np.pi / (2 * (self.R2 - self.R1)) * np.sin(np.pi * (r_p - self.R1) / (self.R2 - self.R1))
                )
            )

        ddfc = np.where(r_p <= self.R1, 0.0,
                np.where(r_p > self.R2, 0.0,
                    -np.pi**2 / (2 * np.power(self.R2 - self.R1, 2)) * np.cos(np.pi * (r_p - self.R1) / (self.R2 - self.R1))
                )
            )

        # Attractive interaction
        fa = -self.B * np.exp(-self.mu * r_p)
        dfa = -self.mu * fa
        ddfa = self.mu**2 * fa

        # Repulsive interaction
        fr = self.A * np.exp(-self.lambda1 * r_p)
        dfr = -self.lambda1 * fr
        ddfr = self.lambda1**2 * fr

        # Bond-order parameter
        if self.style == 'tersoff':
            b = self.chi * np.power(1 + np.power(self.beta * xi_p, self.n), -1 / (2 * self.n))
            db = -0.5 * self.beta * self.chi * np.power(self.beta * xi_p, self.n - 1)
            db *= 1 / np.power(1 + np.power(self.beta * xi_p, self.n), 1 + 1 / (2 * self.n))

            ddb = (self.n - 1) * np.power(self.beta * xi_p, self.n - 2) / np.power(1 + np.power(self.beta * xi_p, self.n), 1 + 1 / (2 * self.n))
            ddb += (-self.n - 0.5) * np.power(self.beta * xi_p, 2 * self.n - 2) / np.power(1 + np.power(self.beta * xi_p, self.n), 2 + 1 / (2 * self.n))
            ddb *= -0.5 * self.chi * self.beta**2

        else:
            raise ValueError(f'Brenner not implemented {self.style}')

        return np.stack([
            ddfc * (fr + b * fa) + 2 * dfc * (dfr + b * dfa) + fc * (ddfr + b * ddfa),
            fc * fa * ddb,
            dfc * fa * db + fc * dfa * db
        ])



@angle_distance_defined
class TersoffBrennerAngle(Manybody.Theta):
    """
    Implementation of Theta for Tersoff-Brenner potentials
    """

    def __init__(self, parameters):
        self.ref = parameters['__ref__']
        self.style = parameters['style'].lower()
        self.el = parameters['el']
        self.c = np.array(parameters['c'])
        self.d = np.array(parameters['d'])
        self.h = np.array(parameters['h'])
        self.R1 = np.array(parameters['R1'])
        self.R2 = np.array(parameters['R2'])

        if self.style == 'tersoff':
            # These are Tersoff-style parameters. The symbols follow the notation in
            # Tersoff J., Phys. Rev. B 39, 5566 (1989)
            #
            # In particular, pair terms are characterized by A, B, lam, mu and parameters for the three body terms ijk
            # depend only on the type of atom i
            self.A = np.array(parameters['A'])
            self.B = np.array(parameters['B'])
            self.lambda1 = np.array(parameters['lambda1'])
            self.mu = np.array(parameters['mu'])
            self.beta = np.array(parameters['beta'])
            self.lambda3 = np.array(parameters['lambda3'])
            self.chi = np.array(parameters['chi'])
            self.n = np.array(parameters['n'])

        elif self.style == 'brenner':
            # These are Brenner/Erhart-Albe-style parameters. The symbols follow the notation in
            # Brenner D., Phys. Rev. B 42, 9458 (1990) and
            # Erhart P., Albe K., Phys. Rev. B 71, 035211 (2005)
            #
            # In particular, pairs terms are characterized by D0, S, beta, r0, the parameters n, chi are always unity and
            # parameters for the three body terms ijk depend on the type of the bond ij
            _D0 = np.array(parameters['D0'])
            _S = np.array(parameters['S'])
            _r0 = np.array(parameters['r0'])
            _beta = np.array(parameters['beta'])
            _mu = np.array(parameters['mu'])
            gamma = np.array(parameters['gamma'])

            # Convert to Tersoff parameters
            self.lambda3 = 2 * _mu
            self.lam = _beta * np.sqrt(2 * _S)
            self.mu = _beta * np.sqrt(2 / _S)
            self.A = _D0 / (_S - 1) * np.exp(lam * _r0)
            self.B = _S * _D0 / (_S - 1) * np.exp(mu * _r0)

        else:
            raise ValueError(f'Unknown parameter style {self.style}')

    def __call__(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # Cutoff function
        fc = np.where(rik <= self.R1, 1.0,
                np.where(rik >= self.R2, 0.0,
                    (1 + np.cos(np.pi * (rik - self.R1) / (self.R2 - self.R1))) / 2
                )
            )

        if self.style == 'tersoff':
            g = 1 + np.power(self.c / self.d, 2) - self.c**2 / (self.d**2 + np.power(self.h - cos, 2))

        else:
            raise ValueError(f'Brenner not implemented {self.style}')

        return fc * g

    def gradient(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

        # First derivative of cos with respect to r
        dcos_drij = (rsq_ij - rsq_ik + rsq_jk) / (2 * rsq_ij * rik)
        dcos_drik = (rsq_ik - rsq_ij + rsq_jk) / (2 * rsq_ik * rij)
        dcos_drjk = - rjk / (rij * rik)

        # Cutoff function
        fc = np.where(rik <= self.R1, 1.0,
                np.where(rik > self.R2, 0.0,
                    (1 + np.cos(np.pi * (rik - self.R1) / (self.R2 - self.R1))) / 2
                )
            )

        dfc = np.where(rik <= self.R1, 0.0,
                np.where(rik > self.R2, 0.0,
                    -np.pi / (2 * (self.R2 - self.R1)) * np.sin(np.pi * (rik - self.R1) / (self.R2 - self.R1))
                )
            )

        if self.style == 'tersoff':
            g = 1 + np.power(self.c / self.d, 2) - self.c**2 / (self.d**2 + np.power(self.h - cos, 2))
            dg_dcos = -2 * self.c**2 * (self.h - cos) / np.power(self.d**2 + np.power(self.h - cos, 2) , 2)

            dg_drij = dg_dcos * dcos_drij
            dg_drik = dg_dcos * dcos_drik
            dg_drjk = dg_dcos * dcos_drjk

        else:
            raise ValueError(f'Brenner not implemented {self.style}')

        return np.stack([
            fc * dg_drij,
            dfc * g + fc * dg_drik,
            fc * dg_drjk
            ])

    def hessian(self, rij, rik, rjk):
        # Squared distances
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle
        cos = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)

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

        # Cutoff function
        fc = np.where(rik <= self.R1, 1.0,
                np.where(rik >= self.R2, 0.0,
                    (1 + np.cos(np.pi * (rik - self.R1) / (self.R2 - self.R1))) / 2
                )
            )

        dfc = np.where(rik <= self.R1, 0.0,
                np.where(rik >= self.R2, 0.0,
                    -np.pi / (2 * (self.R2 - self.R1)) * np.sin(np.pi * (rik - self.R1) / (self.R2 - self.R1))
                )
            )

        ddfc = np.where(rik <= self.R1, 0.0,
                np.where(rik > self.R2, 0.0,
                    -np.pi**2 / (2 * np.power(self.R2 - self.R1, 2)) * np.cos(np.pi * (rik - self.R1) / (self.R2 - self.R1))
                )
            )

        if self.style == 'tersoff':
            g = 1 + np.power(self.c / self.d, 2) - self.c**2 / (self.d**2 + np.power(self.h - cos, 2))
            dg_dcos = -2 * self.c**2 * (self.h - cos) / np.power(self.d**2 + np.power(self.h - cos, 2) , 2)

            ddg_ddcos = (2 * self.c**2 * (self.d**2 - 3 * np.power(self.h - cos, 2))) / np.power(self.d**2 + np.power(self.h - cos, 2) , 3)

            dg_drij = dg_dcos * dcos_drij
            dg_drik = dg_dcos * dcos_drik
            dg_drjk = dg_dcos * dcos_drjk

            ddg_drijdrij = ddg_ddcos * dcos_drij * dcos_drij + dg_dcos * ddcos_drijdrij
            ddg_drikdrik = ddg_ddcos * dcos_drik * dcos_drik + dg_dcos * ddcos_drikdrik
            ddg_drjkdrjk = ddg_ddcos * dcos_drjk * dcos_drjk + dg_dcos * ddcos_drjkdrjk
            ddg_drikdrjk = ddg_ddcos * dcos_drik * dcos_drjk + dg_dcos * ddcos_drikdrjk
            ddg_drijdrjk = ddg_ddcos * dcos_drij * dcos_drjk + dg_dcos * ddcos_drijdrjk
            ddg_drijdrik = ddg_ddcos * dcos_drij * dcos_drik + dg_dcos * ddcos_drijdrik

        else:
            raise ValueError(f'Brenner not implemented {self.style}')

        return np.stack([
            fc * ddg_drijdrij,
            ddfc * g + dfc * dg_drik + dfc * dg_drik + fc * ddg_drikdrik,
            fc * ddg_drjkdrjk,
            dfc * dg_drjk + fc * ddg_drikdrjk,
            fc * ddg_drijdrjk,
            dfc * dg_drij + fc * ddg_drijdrik
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
