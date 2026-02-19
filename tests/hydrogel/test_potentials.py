import numpy as np
from matscipy.calculators.hydrogel.network import (
    ChainPotential,
    LangevinChain,
)
from matscipy.calculators.hydrogel.weight_functions import (
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

