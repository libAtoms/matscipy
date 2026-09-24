
from abc import ABC, abstractmethod
import numpy as np

from matscipy.calculators.manybody.newmb import Manybody
from matscipy.calculators.manybody.potentials import distance_defined


class ChainPotential(ABC):
    """Abstract base class for chain conformational free energy potentials.

    Provides the interface for computing chain conformational free energies
    and their derivatives as a function of end-to-end distance.

    Chain potentials are used in hydrogel simulations to describe the
    conformational free energy of polymer chains connecting crosslinkers.
    """
    @distance_defined
    class ManyBodyPhi(Manybody.Phi):
        """
        Implementation of a harmonic pair interaction.
        """

        def __init__(self, chainpotential):
            self.chainpotential = chainpotential

        def __call__(self, r_p, xi_p):
            return self.chainpotential(r_p) + xi_p

        def gradient(self, r_p, xi_p):
            return np.stack([
                self.chainpotential.derivative(r_p),
                np.ones_like(xi_p),
            ])

        def hessian(self, r_p, xi_p):
            return np.stack([
                self.chainpotential.second_derivative(r_p),
                np.zeros_like(xi_p),
                np.zeros_like(xi_p),
            ])
        
    @abstractmethod
    def __call__(self, r):
        """Compute chain conformational free energy A(r).

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        energy : array_like
            Conformational free energy
        """
        pass

    @abstractmethod
    def derivative(self, r):
        """Compute dA/dr.

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        force : array_like
            Derivative of conformational free energy (force magnitude)
        """
        pass

    @abstractmethod
    def second_derivative(self, r):
        """Compute d²A/dr².

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        stiffness : array_like
            Second derivative of conformational free energy
        """
        pass

    def to_manybody_phi(self):
        return ChainPotential.ManyBodyPhi(self)


class GaussianChain(ChainPotential):
    """Ideal Gaussian chain conformational free energy.

    The chain conformational free energy for an ideal chain is:

        A(r) = (3/2) * (kT / (N * b²)) * r²
    """

    def __init__(self, kuhn_length, chain_monomers, dim=3):
        self.b = kuhn_length
        self.N = chain_monomers
        self.Nm1 = chain_monomers - 1  # N - 1
        self.L0 = self.Nm1 * kuhn_length  # Contour length
        self.dim = dim

    @property
    def rms_end_to_end(self):
        """RMS end-to-end distance of the chain."""
        return np.sqrt(self.Nm1 * self.b**2)

    @property
    def stiffness(self):
        """Effective spring constant of the chain."""
        Re2 = self.rms_end_to_end**2
        return self.dim / Re2

    def __call__(self, r):
        """Compute chain conformational energy A(r).

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        energy : array_like
            Conformational free energy
        """
        r = np.asarray(r)
        return self.stiffness * 0.5 * r**2

    def derivative(self, r):
        """Compute dA/dr.

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        force : array_like
            Derivative of conformational free energy
        """
        r = np.asarray(r)
        return self.stiffness * r

    def second_derivative(self, r):
        """Compute d²A/dr².

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        stiffness : array_like
            Second derivative of conformational free energy
        """
        r = np.asarray(r)
        return np.full_like(r, self.stiffness, dtype=float)


class LangevinChain(ChainPotential):
    """Langevin chain conformational free energy.

    The chain conformational free energy for a freely-jointed chain is:

        A(λ) = (N-1) * kT * [-ln(4π / L⁻¹(λ) * sinh(L⁻¹(λ))) + λ * L⁻¹(λ)]

    where:
        - λ = r / L₀ is the stretch ratio (fractional extension)
        - L₀ = (N - 1) * b is the contour length
        - b is the Kuhn length
        - L⁻¹(x) is the inverse Langevin function

    The inverse Langevin function is approximated using the Cohen Padé
    approximation:
        L⁻¹(x) ≈ x * (3 - x²) / (1 - x²)

    The force is:
        f = dA/dr = L⁻¹(λ) / b

    Note: kT = 1 in reduced units.

    Parameters
    ----------
    kuhn_length : float
        Kuhn length b of the polymer chain
    chain_monomers : float
        Number of monomers N per chain
    """

    def __init__(self, kuhn_length, chain_monomers):
        self.b = kuhn_length
        self.N = chain_monomers
        self.Nm1 = chain_monomers - 1  # N - 1
        self.L0 = self.Nm1 * kuhn_length  # Contour length

    def _inverse_langevin(self, x):
        """Cohen Padé approximation for inverse Langevin function."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        # Handle near-zero values
        mask_small = np.abs(x) < 1e-10
        mask_large = np.abs(x) >= 1.0 - 1e-10
        mask_normal = ~mask_small & ~mask_large

        # For very small x, L^{-1}(x) ≈ 3x
        result[mask_small] = 3.0 * x[mask_small]
        # For x near 1, return large value
        result[mask_large] = 1e10
        # Normal range
        if np.any(mask_normal):
            xn = x[mask_normal]
            result[mask_normal] = xn * (3.0 - xn**2) / (1.0 - xn**2)
        return result

    def _inverse_langevin_derivative(self, x):
        """Derivative of inverse Langevin approximation."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask_small = np.abs(x) < 1e-10
        mask_large = np.abs(x) >= 1.0 - 1e-10
        mask_normal = ~mask_small & ~mask_large

        # For very small x, d/dx[3x] = 3
        result[mask_small] = 3.0
        # For x near 1, derivative is very large
        result[mask_large] = 1e10
        # Normal range: d/dx [x(3-x²)/(1-x²)]
        if np.any(mask_normal):
            xn = x[mask_normal]
            # Let u = x(3-x²), v = 1-x²
            # u' = 3 - 3x², v' = -2x
            # (u/v)' = (u'v - uv')/v² = [(3-3x²)(1-x²) + 2x²(3-x²)] / (1-x²)²
            num = (3.0 - 3.0 * xn**2) * (1.0 - xn**2) + 2.0 * xn**2 * (3.0 - xn**2)
            denom = (1.0 - xn**2) ** 2
            result[mask_normal] = num / denom
        return result

    def _inverse_langevin_second_derivative(self, x):
        """Second derivative of inverse Langevin approximation."""
        x = np.asarray(x)
        # Numerical second derivative for simplicity
        h = 1e-6
        return (
            self._inverse_langevin_derivative(x + h)
            - self._inverse_langevin_derivative(x - h)
        ) / (2.0 * h)

    def __call__(self, r):
        """Compute chain conformational energy A(r).

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        energy : array_like
            Conformational free energy
        """
        r = np.asarray(r)

        if self.Nm1 <= 0 or self.b <= 0:
            return np.zeros_like(r, dtype=float)

        lam = r / self.L0  # Stretch ratio λ = r / L₀

        # Clamp lambda to valid range (0, 1)
        lam = np.clip(lam, 1e-10, 1.0 - 1e-10)

        Linv = self._inverse_langevin(lam)

        # A = (N-1) * [-ln(4π / L⁻¹ * sinh(L⁻¹)) + λ * L⁻¹]
        # = (N-1) * [-ln(4π) + ln(L⁻¹) - ln(sinh(L⁻¹)) + λ * L⁻¹]
        # For numerical stability with large L⁻¹:
        # sinh(x) ~ exp(x)/2 for large x, so ln(sinh(x)) ~ x - ln(2)
        term1 = np.where(
            Linv < 20.0,
            -np.log(4.0 * np.pi / Linv * np.sinh(Linv)),
            -np.log(4.0 * np.pi) + np.log(Linv) - Linv + np.log(2.0),
        )
        term2 = lam * Linv

        return self.Nm1 * (term1 + term2)

    def derivative(self, r):
        """Compute dA/dr (force magnitude, positive = attractive).

        The full derivative of:
        A = (N-1) * [-ln(4π/L⁻¹ * sinh(L⁻¹)) + λ*L⁻¹]
          = (N-1) * [ln(L⁻¹) - ln(sinh(L⁻¹)) - ln(4π) + λ*L⁻¹]

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        deriv : array_like
            Derivative of energy with respect to r
        """
        r = np.asarray(r)

        if self.Nm1 <= 0 or self.b <= 0:
            return np.zeros_like(r, dtype=float)

        lam = r / self.L0
        lam = np.clip(lam, 1e-10, 1.0 - 1e-10)

        Linv = self._inverse_langevin(lam)
        dLinv_dlam = self._inverse_langevin_derivative(lam)

        # d/dλ[ln(L⁻¹)] = (1/L⁻¹) * dL⁻¹/dλ
        # d/dλ[ln(sinh(L⁻¹))] = coth(L⁻¹) * dL⁻¹/dλ
        # d/dλ[λ*L⁻¹] = L⁻¹ + λ*dL⁻¹/dλ
        #
        # dA/dλ = (N-1) * [(1/L⁻¹ - coth(L⁻¹)) * dL⁻¹/dλ + L⁻¹ + λ*dL⁻¹/dλ]
        # dA/dr = dA/dλ / L₀

        # Handle large Linv to avoid overflow in coth
        coth_Linv = np.where(
            Linv < 20.0, 1.0 / np.tanh(Linv + 1e-10), 1.0  # coth(x) → 1 for large x
        )

        d_term1 = (1.0 / Linv - coth_Linv) * dLinv_dlam
        d_term2 = Linv + lam * dLinv_dlam

        dA_dlam = self.Nm1 * (d_term1 + d_term2)
        return dA_dlam / self.L0

    def second_derivative(self, r):
        """Compute d²A/dr² using numerical differentiation.

        Parameters
        ----------
        r : array_like
            End-to-end distance of the chain

        Returns
        -------
        deriv2 : array_like
            Second derivative of energy
        """
        r = np.asarray(r)

        if self.Nm1 <= 0 or self.b <= 0:
            return np.zeros_like(r, dtype=float)

        # Use numerical differentiation for second derivative
        h = 1e-6
        return (self.derivative(r + h) - self.derivative(r - h)) / (2 * h)
