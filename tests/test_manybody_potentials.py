"""Tests for implementation of potentials and derivatives."""

import pytest
import numpy as np
import numpy.testing as nt

from types import SimpleNamespace
from manybody_fixtures import pair_potential, three_body_potential


@pytest.fixture
def pair_derivatives(pair_potential):
    """Compute derivatives and finite differences."""
    N = 10
    L1, L2 = 0.5, 5
    dx = (L2 - L1) / (N - 1)
    rsq, xi = np.linspace(L1, L2, N), np.linspace(L1, L2, N)
    rsq, xi = np.meshgrid(rsq, xi, indexing='ij')

    pot_data, ref_data = SimpleNamespace(), SimpleNamespace()

    # Computing data from potential
    pot_data.E = pair_potential(rsq, xi)
    pot_data.gradient = pair_potential.gradient(rsq, xi)
    pot_data.hessian = pair_potential.hessian(rsq, xi)

    ref_data.gradient = np.gradient(pot_data.E, dx, edge_order=2)
    ref_data.hessian = np.stack([
        np.gradient(pot_data.gradient[0], dx, axis=0, edge_order=2),
        np.gradient(pot_data.gradient[1], dx, axis=1, edge_order=2),
        np.gradient(pot_data.gradient[0], dx, axis=1, edge_order=2),
    ])

    return pot_data, ref_data


def test_pair_potentials(pair_derivatives):
    pot, ref = pair_derivatives
    nt.assert_allclose(pot.gradient[0], ref.gradient[0], rtol=1e-10, atol=1e-14)
    nt.assert_allclose(pot.hessian, ref.hessian, rtol=1e-10, atol=1e-14)
