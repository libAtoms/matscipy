"""Fixtures for Manybody potentials."""
import inspect
import pytest
import matscipy.calculators.manybody.potentials as potentials
import numpy as np

_classes = inspect.getmembers(potentials, inspect.isclass)

_impl_potentials = {
    mother: [cls for _, cls in _classes if issubclass(cls, mother)]
    for mother in (potentials.Manybody.Phi, potentials.Manybody.Theta)
}

# Default arguments for classes where that do not define a default constructor
_default_arguments = {
    potentials.StillingerWeberPair: [{
        "__ref__": "",
        "el": 1,
        "epsilon": 1,
        "sigma": 0.9,
        "costheta0": 1,
        "A": 1,
        "B": 1,
        "p": 1,
        "q": 1,
        "a": 2,
        "lambda1": 1,
        "gamma": 1,
    }, np.inf],
    potentials.StillingerWeberAngle: [{
        "__ref__": "",
        "el": 1,
        "epsilon": 1,
        "sigma": 0.9,
        "costheta0": 1,
        "A": 1,
        "B": 1,
        "p": 1,
        "q": 1,
        "a": 2,
        "lambda1": 1,
        "gamma": 1,
    }],
    potentials.KumagaiPair: [{
        '__ref__': 'T. Kumagai et. al., Comp. Mat. Sci. 39 (2007)',
        'el': 1.0,
        'A': 1.0,
        'B': 1.0,
        'lambda_1': 1.0,
        'lambda_2': 1.0,
        'eta': 1.0,
        'delta': 1.0,
        'alpha': 1.0,
        'beta': 1.0,
        'c_1': 1.0,
        'c_2': 1.0,
        'c_3': 1.0,
        'c_4': 1.0,
        'c_5': 1.0,
        'h': 1.0,
        'R_1': 1.0,
        'R_2': 4.0
    }],
    potentials.KumagaiAngle: [{
        '__ref__': 'T. Kumagai et. al., Comp. Mat. Sci. 39 (2007)',
        'el': 1.0,
        'A': 1.0,
        'B': 1.0,
        'lambda_1': 1.0,
        'lambda_2': 1.0,
        'eta': 1.0,
        'delta': 1.0,
        'alpha': 1.0,
        'beta': 1.0,
        'c_1': 1.0,
        'c_2': 1.0,
        'c_3': 1.0,
        'c_4': 1.0,
        'c_5': 1.0,
        'h': 1.0,
        'R_1': 1.0,
        'R_2': 4.0
    }],
    potentials.TersoffBrennerPair: [{
        '__ref__': 'Tersoff J., Phys. Rev. B 39, 5566 (1989)',
        'style': 'tersoff',
        'el': 1.0,
        'c': 1.0,
        'd': 1.0,
        'h': 1.0,
        'R1': 2.7,
        'R2': 3.0,
        'A': 1.0,
        'B': 1.0,
        'lambda1': 1.0,
        'mu': 1.0,
        'beta': 1.0,
        'lambda3': 1.0,
        'chi': 1.0,
        'n': 1.0
    }],
    potentials.TersoffBrennerAngle: [{
        '__ref__': 'Tersoff J., Phys. Rev. B 39, 5566 (1989)',
        'style': 'tersoff',
        'el': 1.0,
        'c': 1.0,
        'd': 1.0,
        'h': 1.0,
        'R1': 2.7,
        'R2': 3.0,
        'A': 1.0,
        'B': 1.0,
        'lambda1': 1.0,
        'mu': 1.0,
        'beta': 1.0,
        'lambda3': 1.0,
        'chi': 1.0,
        'n': 1.0
    }],
    potentials.LennardJones: [
        1,
        1e-3,  # so that we test where FD is good
        np.inf,
    ],
}

# Filtering out sympy classes
if getattr(potentials, 'SymPhi', None) is not None:
    for m in _impl_potentials:
        for i, c in enumerate(_impl_potentials[m]):
            if issubclass(c, (potentials.SymPhi, potentials.SymTheta)):
                del _impl_potentials[m][i]

# Marking expected failures / TODO fix the following classes
xfails = [
    potentials.KumagaiPair, potentials.KumagaiAngle,
    potentials.TersoffBrennerPair, potentials.BornMayerCut,
]

for fail in xfails:
    for m in _impl_potentials:
        classes = _impl_potentials[m]
        if fail in classes:
            classes[classes.index(fail)] = \
                pytest.param(fail,
                             marks=pytest.mark.xfail(reason="Not implemented"))


class FiniteDiff:
    """Helper class for finite difference tests."""

    hessian_ordering = {
        2: [(0, 0), (1, 1), (0, 1)],  # R, xi
        3: [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)],  # R1, R2, R3
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
    if request.param in _default_arguments:
        return request.param(*_default_arguments[request.param])
    return request.param()


@pytest.fixture(params=_impl_potentials[potentials.Manybody.Theta])
def three_body_potential(request):
    """Fixture for three-body potentials."""
    if request.param in _default_arguments:
        return request.param(*_default_arguments[request.param])
    return request.param()


try:
    from matscipy.calculators.manybody.potentials import (
        SymPhi,
        SymTheta,
        HarmonicPair,
        HarmonicAngle,
        LennardJones,
        ZeroPair,
        BornMayerCut,
    )

    from sympy import symbols, acos, sqrt, pi, exp
    from sympy.abc import R, xi

    has_sympy = True

    R1, R2, R3 = symbols("R_{1:4}")

    _analytical_pair_potentials = [
        (HarmonicPair(1, 1), SymPhi(0.5 * (sqrt(R) - 1)**2 + xi, (R, xi))),
        (ZeroPair(), SymPhi(xi, (R, xi))),
        (LennardJones(1, 1, np.inf),
         SymPhi(4 * ((1 / sqrt(R))**12 - (1 / sqrt(R))**6) + xi, (R, xi))),
        (BornMayerCut(),
         SymPhi(exp((1-sqrt(R))) - 1/R**3 + 1/R**4 + xi, (R, xi))),
    ]

    _analytical_triplet_potentials = [
        (HarmonicAngle(1, np.pi / 2),
         SymTheta(
             0.5 * (acos((R1 + R2 - R3) / (2 * sqrt(R1 * R2))) - pi / 2)**2,
             (R1, R2, R3)))
    ]

    def _pot_names(potlist):
        return [type(pot).__name__ for pot, _ in potlist]

    @pytest.fixture(params=_analytical_pair_potentials,
                    ids=_pot_names(_analytical_pair_potentials))
    def analytical_pair(request):
        return request.param

    @pytest.fixture(params=_analytical_triplet_potentials,
                    ids=_pot_names(_analytical_triplet_potentials))
    def analytical_triplet(request):
        return request.param

except ImportError:
    has_sympy = False
