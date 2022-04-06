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


@distance_defined
class HarmonicPair(Manybody.Phi):
    """Implementation of a harmonic pair interaction."""

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
