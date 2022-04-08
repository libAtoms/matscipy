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

        res = old.gradient(self, rij, rik, rjk)
        res[0] *= 1 / (2 * rij)
        res[1] *= 1 / (2 * rik)
        res[2] *= 1 / (2 * rjk)
        return res

    @wraps(cls.hessian)
    def hessian(self, rij, rik, rjk):
        pass

    cls.__call__ = call
    cls.gradient = gradient
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
        f = np.arccos((rij**2 + rik**2 - rjk**2) / (2 * rij * rik))

        return 0.5 * self.k0 * (f - self.theta0)**2

    def gradient(self, rij, rik, rjk):
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
        pass

