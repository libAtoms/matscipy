import numpy as np
from matscipy.calculators.hydrogel.potentials import (
    ChainPotential,
    FloryHuggins,
    LangevinChain,
    LucyWeightFunction,
    LucyWeightFunction2D,
)


class TestLucyWeightFunction:
    """Tests for the Lucy weight function."""

    def test_normalization(self):
        """Test that Lucy function integrates to 1 over 3D space."""
        lucy = LucyWeightFunction(cutoff=5.0)

        # Numerical integration using spherical coordinates
        # ∫∫∫ W(r) r² sin(θ) dr dθ dφ = 4π ∫ W(r) r² dr
        r = np.linspace(0, lucy.cutoff, 1000)
        dr = r[1] - r[0]
        w = lucy(r)
        integral = 4 * np.pi * np.sum(w * r**2) * dr

        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_at_zero(self):
        """Test W(0) value."""
        lucy = LucyWeightFunction(cutoff=5.0)
        w0 = lucy.at_zero()
        w_near_zero = lucy(np.array([1e-10]))[0]

        np.testing.assert_allclose(w0, w_near_zero, rtol=1e-5)

    def test_continuity_at_cutoff(self):
        """Test that W(rc) = 0."""
        lucy = LucyWeightFunction(cutoff=5.0)

        # Should be zero at cutoff
        w_at_rc = lucy(np.array([lucy.cutoff]))[0]
        assert w_at_rc == 0.0

        # Should be zero beyond cutoff
        w_beyond = lucy(np.array([lucy.cutoff + 0.1]))[0]
        assert w_beyond == 0.0

    def test_derivative_continuity(self):
        """Test that dW/dr approaches 0 at cutoff."""
        lucy = LucyWeightFunction(cutoff=5.0)

        # Should be zero at cutoff
        dw_at_rc = lucy.derivative(np.array([lucy.cutoff - 1e-10]))[0]
        np.testing.assert_allclose(dw_at_rc, 0.0, atol=1e-5)

    def test_numerical_derivative(self):
        """Test derivative against numerical differentiation."""
        lucy = LucyWeightFunction(cutoff=5.0)
        r = np.linspace(0.1, lucy.cutoff - 0.1, 50)
        h = 1e-6

        # Numerical derivative
        dw_num = (lucy(r + h) - lucy(r - h)) / (2 * h)

        # Analytical derivative
        dw_ana = lucy.derivative(r)

        np.testing.assert_allclose(dw_ana, dw_num, rtol=1e-5)

    def test_numerical_second_derivative(self):
        """Test second derivative against numerical differentiation."""
        lucy = LucyWeightFunction(cutoff=5.0)
        r = np.linspace(0.1, lucy.cutoff - 0.1, 50)
        h = 1e-6

        # Numerical second derivative
        d2w_num = (lucy.derivative(r + h) - lucy.derivative(r - h)) / (2 * h)

        # Analytical second derivative
        d2w_ana = lucy.second_derivative(r)

        np.testing.assert_allclose(d2w_ana, d2w_num, rtol=1e-4)


class TestLucyWeightFunction2D:
    """Tests for the 2D Lucy weight function."""

    def test_normalization(self):
        """Test that Lucy function integrates to 1 over 2D space."""
        lucy = LucyWeightFunction2D(cutoff=5.0)

        # Numerical integration using polar coordinates
        # ∫∫ W(r) r dr dθ = 2π ∫ W(r) r dr
        r = np.linspace(0, lucy.cutoff, 1000)
        dr = r[1] - r[0]
        w = lucy(r)
        integral = 2 * np.pi * np.sum(w * r) * dr

        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_at_zero(self):
        """Test W(0) value."""
        lucy = LucyWeightFunction2D(cutoff=5.0)
        w0 = lucy.at_zero()
        w_near_zero = lucy(np.array([1e-10]))[0]

        np.testing.assert_allclose(w0, w_near_zero, rtol=1e-5)

    def test_continuity_at_cutoff(self):
        """Test that W(rc) = 0."""
        lucy = LucyWeightFunction2D(cutoff=5.0)

        # Should be zero at cutoff
        w_at_rc = lucy(np.array([lucy.cutoff]))[0]
        assert w_at_rc == 0.0

        # Should be zero beyond cutoff
        w_beyond = lucy(np.array([lucy.cutoff + 0.1]))[0]
        assert w_beyond == 0.0

    def test_derivative_continuity(self):
        """Test that dW/dr approaches 0 at cutoff."""
        lucy = LucyWeightFunction2D(cutoff=5.0)

        # Should be zero at cutoff
        dw_at_rc = lucy.derivative(np.array([lucy.cutoff - 1e-10]))[0]
        np.testing.assert_allclose(dw_at_rc, 0.0, atol=1e-5)

    def test_numerical_derivative(self):
        """Test derivative against numerical differentiation."""
        lucy = LucyWeightFunction2D(cutoff=5.0)
        r = np.linspace(0.1, lucy.cutoff - 0.1, 50)
        h = 1e-6

        # Numerical derivative
        dw_num = (lucy(r + h) - lucy(r - h)) / (2 * h)

        # Analytical derivative
        dw_ana = lucy.derivative(r)

        np.testing.assert_allclose(dw_ana, dw_num, rtol=1e-5)

    def test_numerical_second_derivative(self):
        """Test second derivative against numerical differentiation."""
        lucy = LucyWeightFunction2D(cutoff=5.0)
        r = np.linspace(0.1, lucy.cutoff - 0.1, 50)
        h = 1e-6

        # Numerical second derivative
        d2w_num = (lucy.derivative(r + h) - lucy.derivative(r - h)) / (2 * h)

        # Analytical second derivative
        d2w_ana = lucy.second_derivative(r)

        np.testing.assert_allclose(d2w_ana, d2w_num, rtol=1e-4)


# ============== Flory-Huggins Tests ==============


class TestFloryHuggins:
    """Tests for Flory-Huggins embedding energy."""

    def test_energy_at_low_density(self):
        """Test that energy is finite at low density."""
        fh = FloryHuggins(
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        w0 = 105.0 / (16.0 * np.pi * 20**3)

        # Low density (mostly solvent)
        rho = np.array([0.01])
        E = fh(rho)

        assert np.isfinite(E).all()

    def test_derivative_numerical(self):
        """Test derivative against numerical differentiation."""
        fh = FloryHuggins(
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        w0 = 105.0 / (16.0 * np.pi * 20**3)
        rho = np.linspace(0.01, 0.1, 10)
        h = 1e-8

        # Numerical derivative
        dF_num = (fh(rho + h) - fh(rho - h)) / (2 * h)

        # Analytical derivative
        dF_ana = fh.derivative(rho)

        np.testing.assert_allclose(dF_ana, dF_num, rtol=1e-4)

    def test_second_derivative_numerical(self):
        """Test second derivative against numerical differentiation."""
        fh = FloryHuggins(
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        rho = np.linspace(0.01, 0.1, 10)
        h = 1e-7

        # Numerical second derivative
        d2F_num = (fh.derivative(rho + h) - fh.derivative(rho - h)) / (2 * h)

        # Analytical second derivative
        d2F_ana = fh.second_derivative(rho)

        np.testing.assert_allclose(d2F_ana, d2F_num, rtol=1e-4)

    def test_energy_consistency(self):
        """Test that energy, derivative, and second derivative are consistent."""
        fh = FloryHuggins(
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )

        rho = np.linspace(0.01, 0.9, 10) * fh.max_crosslink_density

        # energy per crosslink
        E = fh(rho)

        # energy per volume
        E_vol = E * rho

        # Implementation of energy per volume
        E_vol_impl = fh.per_volume(rho)

        assert np.allclose(E_vol, E_vol_impl, rtol=1e-8)


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
