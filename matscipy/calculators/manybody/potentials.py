"""Manybody potential definitions."""

import numpy as np

from functools import wraps
from types import SimpleNamespace
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
        # I would delete this one. The mixed derivative is zero anyways
        # We did not change the second derivative with \xi. 
        hess[2] *= e_p
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

class HarmonicAngle(Manybody.Theta):
    """
    Implementation of a harmonic angle interaction.
    """

    def __init__(self, k0=0.1, theta0=149.47):
        self.k0 = k0
        self.theta0 = theta0

    def __call__(self, rij, rik, rjk):
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        theta_c = np.arccos((rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rjk))
        return 0.5 * self.k0 * (theta_c - self.theta0)**2

    def gradient(self, rij, rik, rjk):
        rsq_ij = rij**2
        rsq_ik = rik**2
        rsq_jk = rjk**2

        # cos of angle 
        f = (rsq_ij + rsq_ik - rsq_jk) / (2 * rij * rik)
        # derivatives 
        df_drsq_ij = (rsq_ij - rsq_ik + rsq_jk) / (4 * rik * rij**3)
        df_drsq_ik = (rsq_ik - rsq_ij + rsq_jk) / (4 * rij * rik**3)
        df_drsq_jk = - 1 / (2 * rij * rik)

        # Scalar derivatives
        def E(a):
            return self.k0 * (a - self.theta0)  

        def h(f):
            with np.errstate(divide="raise"):
                d_arccos = -1 / np.sqrt(1 - f**2)
            return E(np.arccos(f)) * d_arccos

        # Derivatives with respect to squared distances
        dtheta = np.zeros((3, len(rij)))
        dtheta[0] = df_drsq_ij
        dtheta[1] = df_drsq_ik
        dtheta[2] = df_drsq_jk

        return dtheta * h(f)

    def hessian(self, rij, rik, rjk):
        pass

