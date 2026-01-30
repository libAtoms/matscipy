#
# Copyright 2025 Antoine Sanner (ETH Zürich)
#           2025 Lars Pastewka (University of Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""Potential functions for hydrogel simulations.

This module provides:
- Weight functions for density estimation (Lucy)
- Flory-Huggins mixing free energy
- Langevin chain conformational free energy
"""

from abc import ABC, abstractmethod
from typing import Union
import warnings

import numpy as np

from matscipy.calculators.manybody.newmb import Manybody
from matscipy.calculators.manybody.potentials import distance_defined


class WeightFunction(ABC):
    """Abstract base class for weight functions used in density estimation."""

    @abstractmethod
    def __call__(self, r) -> float:
        """Evaluate weight function W(r) for cutoff rc."""
        pass

    @abstractmethod
    def derivative(self, r) -> float:
        """Evaluate dW/dr for cutoff rc."""
        pass

    @abstractmethod
    def second_derivative(self, r) -> float:
        """Evaluate d²W/dr² for cutoff rc."""
        pass

    @abstractmethod
    def at_zero(self) -> float:
        """Evaluate W(0) for cutoff rc (self-contribution)."""
        pass


class LucyWeightFunction(WeightFunction):
    """Lucy weight function for SPH-like density estimation.

    The Lucy function is normalized to integrate to 1 over 3D space:

        W(r) = (105 / 16π rc³) * (1 + 3r/rc) * (1 - r/rc)³  for r < rc
        W(r) = 0                                             for r >= rc

    This weight function is smooth (C¹ continuous at r=rc) and commonly
    used in smoothed particle hydrodynamics.
    """

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, r):
        """Evaluate Lucy weight function."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        # Normalization: 105 / (16 * pi * rc^3)
        norm = 105.0 / (16.0 * np.pi * self.cutoff**3)
        result[mask] = norm * (1.0 + 3.0 * x) * (1.0 - x) ** 3
        return result

    def derivative(self, r):
        """Evaluate dW/dr."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 105.0 / (16.0 * np.pi * self.cutoff**3)
        # d/dr[(1 + 3x)(1 - x)^3] = (1/rc) * [3(1-x)^3 - 3(1+3x)(1-x)^2]
        #                        = (1/rc) * 3(1-x)^2 * [(1-x) - (1+3x)]
        #                        = (1/rc) * 3(1-x)^2 * (-4x)
        #                        = -12x(1-x)^2 / rc
        result[mask] = norm * (-12.0 * x * (1.0 - x) ** 2) / self.cutoff
        return result

    def second_derivative(self, r):
        """Evaluate d²W/dr²."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 105.0 / (16.0 * np.pi * self.cutoff**3)
        # d²/dr²[(1 + 3x)(1 - x)^3]
        # First derivative: dW/dr = -12*norm*x*(1-x)^2 / rc
        # Second derivative: d/dr[-12*norm*x*(1-x)^2 / rc] / rc
        # = -12*norm/rc² * d/dx[x*(1-x)^2]
        # = -12*norm/rc² * [(1-x)^2 + x*2*(1-x)*(-1)]
        # = -12*norm/rc² * [(1-x)^2 - 2x*(1-x)]
        # = -12*norm/rc² * (1-x)*[(1-x) - 2x]
        # = -12*norm/rc² * (1-x)*(1-3x)
        result[mask] = norm * (-12.0 * (1.0 - x) * (1.0 - 3.0 * x)) / self.cutoff**2
        return result

    def at_zero(self):
        """Evaluate W(0)."""
        return 105.0 / (16.0 * np.pi * self.cutoff**3)


class LucyWeightFunction2D(WeightFunction):
    """Lucy weight function for SPH-like density estimation.

    The Lucy function is normalized to integrate to 1 over 2D space:

        W(r) = (5 / π rc²) * (1 + 3r/rc) * (1 - r/rc)³  for r < rc
        W(r) = 0                                         for r >= rc

    This weight function is smooth (C¹ continuous at r=rc) and commonly
    used in smoothed particle hydrodynamics.
    """

    dim = 2

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, r):
        """Evaluate Lucy weight function."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        # Normalization: 5 / (pi * rc^2)
        norm = 5.0 / (np.pi * self.cutoff**2)
        result[mask] = norm * (1.0 + 3.0 * x) * (1.0 - x) ** 3
        return result

    def derivative(self, r):
        """Evaluate dW/dr."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 5.0 / (np.pi * self.cutoff**2)
        # d/dr[(1 + 3x)(1 - x)^3] = (1/rc) * [3(1-x)^3 - 3(1+3x)(1-x)^2]
        #                        = (1/rc) * 3(1-x)^2 * [(1-x) - (1+3x)]
        #                        = (1/rc) * 3(1-x)^2 * (-4x)
        #                        = -12x(1-x)^2 / rc
        result[mask] = norm * (-12.0 * x * (1.0 - x) ** 2) / self.cutoff
        return result

    def second_derivative(self, r):
        """Evaluate d²W/dr²."""
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=float)
        mask = r < self.cutoff
        x = r[mask] / self.cutoff
        norm = 5.0 / (np.pi * self.cutoff**2)
        # d²/dr²[(1 + 3x)(1 - x)^3]
        # First derivative: dW/dr = -12*norm*x*(1-x)^2 / rc
        # Second derivative: d/dr[-12*norm*x*(1-x)^2 / rc] / rc
        # = -12*norm/rc² * d/dx[x*(1-x)^2]
        # = -12*norm/rc² * [(1-x)^2 + x*2*(1-x)*(-1)]
        # = -12*norm/rc² * [(1-x)^2 - 2x*(1-x)]
        # = -12*norm/rc² * (1-x)*[(1-x) - 2x]
        # = -12*norm/rc² * (1-x)*(1-3x)
        result[mask] = norm * (-12.0 * (1.0 - x) * (1.0 - 3.0 * x)) / self.cutoff**2
        return result

    def at_zero(self):
        """Evaluate W(0)."""
        return 5.0 / (np.pi * self.cutoff**2)


class EmbeddingPotential(ABC):

    @abstractmethod
    def __call__(self, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        rho: Denity of crosslinks (atoms) following including the self-contribution.
        """
        pass

    @abstractmethod
    def derivative(self, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        rho: Denity of crosslinks (atoms) following including the self-contribution.
        """
        pass

    @abstractmethod
    def second_derivative(self, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        rho: Denity of crosslinks (atoms) following including the self-contribution.
        """
        pass


class FloryHugginsPotential(EmbeddingPotential):
    """Flory-Huggins mixing free energy for polymer-solvent systems.

    The mixing free energy per volume is:

        a_mix/kT = (1/v) φ ln(φ) + (1/v₀)(1-φ)ln(1-φ) + (χ/v₀)φ(1-φ)

    where v = N*v₀ is the chain volume and φ is the polymer volume fraction.

    The embedding energy per crosslinker is F = a_mix * v_i where v_i = 1/n
    is the volume per crosslinker and n is the local crosslinker density.

    Parameters
    ----------
    chain_monomers : float
        Number of monomers N per chain
    monomer_volume : float
        Volume v₀ of a monomer
    flory_chi : float
        Flory-Huggins interaction parameter χ
    coordination : float
        Number of chains per crosslinker (typically 4 for diamond lattice)

    Note: kT = 1 in reduced units.
    """

    def __init__(self, chain_monomers, monomer_volume, flory_chi, coordination):
        self.N = chain_monomers
        self.v0 = monomer_volume
        self.chi = flory_chi
        self.coord = coordination
        self.vchain = monomer_volume * chain_monomers  # Chain volume

    @property
    def max_crosslink_density(self):
        """Maximum crosslink density where the monomoer volume fraction is 1"""
        return self.crosslink_density_from_polymer_volume_fraction(1)

    def polymer_volume_fraction_from_crosslink_density(self, crosslink_density):
        """Compute polymer volume fraction φ from crosslink density ρ."""
        ρ = np.asarray(crosslink_density)
        assert np.min(ρ) > 0, "Negative crosslink density encountered." 

        # Chain density (chains per unit volume)
        # Each crosslinker has coord chains, but each chain connects two
        # crosslinkers, so we divide by 2
        ν = ρ * self.coord / 2.0
        # Polymer volume fraction φ = ν * v_chain
        ϕ = ν * self.vchain
        return ϕ
    
    def crosslink_density_from_polymer_volume_fraction(self, polymer_volume_fraction):
        """Compute crosslink density ρ from polymer volume fraction φ."""
        ϕ = np.asarray(polymer_volume_fraction)
        # Chain density ν = φ / v_chain
        ν = ϕ / self.vchain
        # Crosslink density ρ = 2 * ν / coord
        ρ = 2 * ν / self.coord
        return ρ

    def per_volume(self, crosslink_density, der="0"):
        """
        Mixing free energy per unit volume. This is the standard flory huggins theory (except that it takes the crosslink density as input),
        we define it here for purpose of analytical computations

        Here crosslink_density is the real crosslink density, including the self contribution
        Derivatives with respect to the crosslink_density or the polymer volume fraction can be computed
        """

        ϕ = self.polymer_volume_fraction_from_crosslink_density(crosslink_density)

        # Clamp phi to avoid log(0) and ensure physical range
        ϕ_original = ϕ.copy() if hasattr(ϕ, 'copy') else ϕ
        # ϕ = np.clip(ϕ, 1e-13        , 1.0 - 1e-13)
        
        # # Warn if clipping occurred
        # if np.any(ϕ_original < 1e-13) or np.any(ϕ_original > 1.0 - 1e-13):
        #     warnings.warn(f"Volume fraction φ was clipped: original range [{np.min(ϕ_original):.2e}, {np.max(ϕ_original):.2e}] "
        #                  f"to valid range [1e-13, {1.0 - 1e-13}]. This may indicate unphysical crosslink densities.",
        #                  UserWarning)

        χ = self.chi
        v0 = self.v0

        # Volume of a chain
        vchain = self.vchain

        if der == "0":
            return (
                1 / (vchain) * ϕ * np.log(ϕ)
                + 1 / v0 * (1 - ϕ) * np.log(1 - ϕ)
                + 1 / v0 * χ * ϕ * (1 - ϕ)
            )
        elif der == "phi":
            return (
                1 / vchain * (np.log(ϕ) + 1)
                - 1 / v0 * (np.log(1 - ϕ) + 1)
                + 1 / v0 * χ * (1 - 2 * ϕ)
            )
        elif der == "phi2":
            return 1 / vchain * (1 / ϕ) + 1 / v0 * (1 / (1 - ϕ)) - 2 / v0 * χ
        elif der == "rho":
            return (
                (
                    1 / vchain * (np.log(ϕ) + 1)
                    - 1 / v0 * (np.log(1 - ϕ) + 1)
                    + 1 / v0 * χ * (1 - 2 * ϕ)
                )
                * (self.coord / 2)
                * vchain
            )
        elif der == "rho2":
            d2f_dphi2 = 1 / vchain * (1 / ϕ) + 1 / v0 * (1 / (1 - ϕ)) - 2 / v0 * χ
            return d2f_dphi2 * ((self.coord / 2) * vchain) ** 2
        else:
            raise ValueError(f"Unknown derivative option der={der}")

    def pressure(self, crosslink_density=None, polymer_volume_fraction=None):
        """Compute osmotic pressure"""
        if crosslink_density is None and polymer_volume_fraction is None:
            raise ValueError("Either crosslink_density or polymer_volume_fraction must be provided.")

        if polymer_volume_fraction is None:
            polymer_volume_fraction = self.polymer_volume_fraction_from_crosslink_density(crosslink_density)
        ϕ = np.asarray(polymer_volume_fraction)
        # Clamp phi to avoid log(0) and ensure physical range
        ϕ_original =ϕ.copy()
        ϕ = np.clip(ϕ, 1e-10, 1.0 - 1e-10)

        # Warn if clipping occurred
        if np.any(ϕ_original < 1e-10) or np.any(ϕ_original > 1.0 - 1e-10):
            warnings.warn(f"Volume fraction φ was clipped: original range [{np.min(ϕ_original):.2e}, {np.max(ϕ_original):.2e}] "
                         f"to valid range [1e-10, {1.0 - 1e-10}]. This may indicate unphysical crosslink densities.",
                         UserWarning)
            
        χ = self.chi
        v0 = self.v0
        vchain = self.vchain

        return (
                - ϕ * ( 1/ v0 - 1 / (vchain))  
                - 1 / v0 * np.log(1 - ϕ)
                - 1 / v0 * χ * ϕ**2
            )

    def __call__(self, rho):
        """Compute embedding energy F(ρ).

        Parameters
        ----------
        rho : array_like
            Local crosslinker density at each crosslinker (including self),
            i.e., rho = sum_j W(r_ij) where the sum includes i=j.
            Note that the rho as defined in EAM implementations usually excludes
            the self-contribution W(0).

        Returns
        -------
        energy : array_like
            Embedding energy at each crosslinker
        """
        rho = np.asarray(rho)

        # Volume per crosslinker
        vi = 1.0 / rho

        # Free energy per crosslinker
        return self.per_volume(rho) * vi

    def derivative(self, rho):
        """Compute dF/dρ using numerical differentiation.

        Parameters
        ----------
        rho : array_like
            Local crosslinker density at each crosslinker (including self)


        Returns
        -------
        deriv : array_like
            Derivative of embedding energy with respect to rho
        """
        rho = np.asarray(rho)
        vi = 1.0 / rho
        vip = - 1.0 / (rho ** 2)
        return self.per_volume(rho, der='rho') * vi + self.per_volume(rho) * vip


    def second_derivative(self, rho):
        """Compute d²F/dρ² using numerical differentiation.

        Parameters
        ----------
        rho : array_like
            Local crosslinker density at each crosslinker (including self)

        Returns
        -------
        deriv2 : array_like
            Second derivative of embedding energy
        """
        rho = np.asarray(rho)
        vi = 1.0 / rho
        vip = - 1.0 / (rho ** 2)
        vipp = 2.0 / (rho ** 3)
        return self.per_volume(rho, der='rho2') * vi + 2 * self.per_volume(rho, der='rho') * vip + self.per_volume(rho) * vipp


        


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
