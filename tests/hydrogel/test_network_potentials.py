import numpy as np
from matscipy.calculators.hydrogel.network import LangevinChain

# ============== Langevin Chain Tests ==============


class TestLangevinChain:
    """Tests for Langevin chain conformational energy."""

    def test_energy_increases_with_extension(self):
        """Test that energy increases monotonically with extension."""
        chain = LangevinChain(kuhn_length=1.0, chain_monomers=50)
        L0 = chain.L0  # Contour length = 49

        # Energy should increase monotonically from 0 to L0
        r = np.linspace(0.1, 0.9 * L0, 50)
        E = chain(r)

        # Check that energy is monotonically increasing
        dE = np.diff(E)
        assert np.all(dE > 0), "Energy should increase with extension"

    def test_force_always_positive(self):
        """Test that force (dE/dr) is always positive (chain wants to contract)."""
        chain = LangevinChain(kuhn_length=1.0, chain_monomers=50)
        L0 = chain.L0

        # Force should be positive (restoring/contracting) at all extensions
        r = np.linspace(0.1, 0.9 * L0, 50)
        dE = chain.derivative(r)

        # dE/dr > 0 means force = -dE/dr < 0 (pulls back toward r=0)
        assert np.all(dE > 0), "Force should always be positive (contracting)"

    def test_force_diverges_near_contour_length(self):
        """Test that force diverges as chain approaches contour length."""
        chain = LangevinChain(kuhn_length=1.0, chain_monomers=50)
        L0 = chain.L0

        # Force should increase rapidly near L0
        r_far = np.array([0.5 * L0])
        r_near = np.array([0.95 * L0])

        f_far = chain.derivative(r_far)[0]
        f_near = chain.derivative(r_near)[0]

        assert f_near > 10 * f_far, "Force should be much larger near contour length"

    def test_derivative_numerical(self):
        """Test derivative against numerical differentiation."""
        chain = LangevinChain(kuhn_length=1.0, chain_monomers=50)
        L0 = chain.L0

        # Test in a stable region (not too close to L0)
        r = np.linspace(0.1 * L0, 0.7 * L0, 20)
        h = 1e-6

        # Numerical derivative
        dE_num = (chain(r + h) - chain(r - h)) / (2 * h)

        # Analytical derivative
        dE_ana = chain.derivative(r)

        np.testing.assert_allclose(dE_ana, dE_num, rtol=1e-4)

    def test_second_derivative_numerical(self):
        """Test second derivative against numerical differentiation."""
        chain = LangevinChain(kuhn_length=1.0, chain_monomers=50)
        L0 = chain.L0

        # Test in a stable region (not too close to L0)
        r = np.linspace(0.1 * L0, 0.6 * L0, 20)
        h = 1e-6

        # Numerical second derivative
        d2E_num = (chain.derivative(r + h) - chain.derivative(r - h)) / (2 * h)

        # Analytical second derivative
        d2E_ana = chain.second_derivative(r)

        np.testing.assert_allclose(d2E_ana, d2E_num, rtol=1e-3)
