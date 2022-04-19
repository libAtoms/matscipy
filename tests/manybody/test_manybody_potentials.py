"""Tests for implementation of potentials and derivatives."""

import pytest
import numpy as np
import numpy.testing as nt

from types import SimpleNamespace
from manybody_fixtures import (
    pair_potential,
    three_body_potential,
    has_sympy,
    analytical_pair,
    analytical_triplet,
    FiniteDiff,
)


def evaluate(pot, *args):
    data = SimpleNamespace()
    data.E = pot(*args)
    data.gradient = pot.gradient(*args)
    data.hessian = pot.hessian(*args)
    return data


@pytest.fixture
def fd_evaluated_pair(pair_potential):
    """
    Compute derivatives and finite differences.
    """
    N = 10
    L1, L2 = 0.5, 5
    rsq, xi = np.linspace(L1, L2, N), np.linspace(L1, L2, N)

    fd_pot = FiniteDiff(pair_potential, (rsq, xi))
    args = np.meshgrid(rsq, xi, indexing='ij')
    return evaluate(pair_potential, *args), evaluate(fd_pot, *args)


@pytest.fixture
def fd_evaluated_three_body(three_body_potential):
    """
    Compute derivatives and finite differences.
    """
    N = 10
    L1, L2 = 1.0, 1.15

    rij, rik, rjk = (
        np.linspace(L1, L2, N),
        np.linspace(L1, L2, N),
        np.linspace(L1, L2, N),
    )

    fd_pot = FiniteDiff(three_body_potential, (rij, rik, rjk))
    args = np.meshgrid(rij, rik, rjk, indexing='ij')
    return evaluate(three_body_potential, *args), evaluate(fd_pot, *args)


def test_fd_pair(fd_evaluated_pair):
    pot, ref = fd_evaluated_pair
    nt.assert_allclose(pot.gradient, ref.gradient, rtol=1e-10, atol=1e-14)
    nt.assert_allclose(pot.hessian, ref.hessian, rtol=1e-10, atol=1e-14)


def test_fd_three_body(fd_evaluated_three_body):
    pot, ref = fd_evaluated_three_body
    nt.assert_allclose(pot.gradient, ref.gradient, rtol=1e-4, atol=1e-4)
    nt.assert_allclose(pot.hessian, ref.hessian, rtol=1e-4, atol=1e-4)


@pytest.fixture
def analytical_evaluated_pair(analytical_pair):
    N = 10
    L1, L2 = 0.5, 5
    rsq, xi = np.linspace(L1, L2, N), np.linspace(L1, L2, N)
    args = np.meshgrid(rsq, xi, indexing='ij')

    pot, analytical = analytical_pair
    return evaluate(pot, *args), evaluate(analytical, *args)


@pytest.fixture
def analytical_evaluated_three_body(analytical_triplet):
    """
    Compute derivatives and finite differences.
    """
    N = 10
    L1, L2 = 1.0, 1.15
    rij, rik, rjk = (
        np.linspace(L1, L2, N),
        np.linspace(L1, L2, N),
        np.linspace(L1, L2, N),
    )

    args = np.meshgrid(rij, rik, rjk, indexing='ij')
    pot, analytical = analytical_triplet
    return evaluate(pot, *args), evaluate(analytical, *args)


@pytest.mark.skipif(not has_sympy, reason="Sympy not installed")
def test_analytical_pairs(analytical_evaluated_pair):
    pot, ref = analytical_evaluated_pair

    # Checking all computed fields
    for k in pot.__dict__:
        f, f_ref = getattr(pot, k), getattr(ref, k)
        nt.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-14)


@pytest.mark.skipif(not has_sympy, reason="Sympy not installed")
def test_analytical_triplets(analytical_evaluated_three_body):
    pot, ref = analytical_evaluated_three_body

    # Checking all computed fields
    for k in pot.__dict__:
        f, f_ref = getattr(pot, k), getattr(ref, k)
        nt.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-14)
