"""Fixtures for Manybody potentials."""
import inspect
import pytest
import matscipy.calculators.manybody.potentials as potentials
import numpy as np


_classes = inspect.getmembers(potentials, inspect.isclass)

_impl_potentials = {
    mother: [
        cls for _, cls in _classes
        if issubclass(cls, mother)
    ]

    for mother in (potentials.Manybody.Phi, potentials.Manybody.Theta)
}

# Filtering out sympy classes
if getattr(potentials, 'SymPhi', None) is not None:
    for m in _impl_potentials:
        for i, c in enumerate(_impl_potentials[m]):
            if issubclass(c, (potentials.SymPhi, potentials.SymTheta)):
                del _impl_potentials[m][i]


class FiniteDiff:
    """Helper class for finite difference tests."""

    hessian_ordering = {
        2: [(0, 0), (1, 1), (0, 1)],
        3: [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)],
    }

    def __init__(self, pot, coords):
        self.pot = pot
        self.coords = coords
    def __call__(self, *args):
        return self.pot(*args)
    def gradient(self, *args):
        E = self.pot(*args)
        return np.gradient(E, *self.coords, edge_order=2)
    def hessian(self, *args):
        G = self.pot.gradient(*args)
        return np.stack([
            np.gradient(G[i], self.coords[j], axis=j, edge_order=2)
            for i, j in self.hessian_ordering[len(args)]
        ])


@pytest.fixture(params=_impl_potentials[potentials.Manybody.Phi])
def pair_potential(request):
    """Fixture for pair potentials."""
    return request.param()


@pytest.fixture(params=_impl_potentials[potentials.Manybody.Theta])
def three_body_potential(request):
    """Fixture for three-body potentials."""
    return request.param()


try:
    from matscipy.calculators.manybody.potentials import (
        SymPhi,
        SymTheta,
        HarmonicPair,
        HarmonicAngle,
    )

    from sympy import symbols, acos, sqrt, pi
    from sympy.abc import R, xi

    has_sympy = True

    R1, R2, R3 = symbols("R_{1:4}")

    _analytical_pair_potentials = [
        (HarmonicPair(1, 1), SymPhi(0.5 * (sqrt(R) - 1)**2 + xi, (R, xi))),
    ]

    _analytical_triplet_potentials = [
        (HarmonicAngle(1, np.pi / 2),
         SymTheta(0.5 * (acos((R1 + R2 - R3)/(2 * sqrt(R1 * R2))) - pi / 2)**2,
                  (R1, R2, R3)))
    ]

    @pytest.fixture(params=_analytical_pair_potentials)
    def analytical_pair(request):
        return request.param

    @pytest.fixture(params=_analytical_triplet_potentials)
    def analytical_triplet(request):
        return request.param

except ImportError:
    has_sympy = False
