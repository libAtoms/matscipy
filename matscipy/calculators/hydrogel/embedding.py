"""

Embedding mixing free energy for hydrogels (Flory Huggins theory) 
allowing for different coordination numbers of the crosslinks

"""



from abc import ABC, abstractmethod
import numpy as np
from typing import Union

class EmbeddingPotential(ABC):

    @abstractmethod
    def __call__(self, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        rho: Denity of crosslinks (atoms) including the self-contribution.
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
    coordination : float or array of floats
        Number of chains per crosslinker (typically 4 for diamond lattice)

    Note: kT = 1 in reduced units.
    """

    def __init__(self, chain_monomers, monomer_volume, flory_chi):
        self.N = chain_monomers
        self.v0 = monomer_volume
        self.chi = flory_chi
        self.vchain = monomer_volume * chain_monomers  # Chain volume

    @property
    def max_crosslink_density(self):
        """Maximum crosslink density where the monomoer volume fraction is 1"""
        return self.crosslink_density_from_polymer_volume_fraction(1)

    def polymer_volume_fraction_from_chain_density(self, chain_density):
        """Compute polymer volume fraction φ from crosslink density ρ."""
        ν = np.asarray(chain_density)
        assert np.min(ν) > 0, "Negative chain density encountered." 

        # Polymer volume fraction φ = ν * v_chain
        ϕ = ν * self.vchain
        return ϕ

    def polymer_volume_fraction_from_crosslink_density(self, crosslink_density,coordination):
        """Compute polymer volume fraction φ from crosslink density ρ."""
        ρ = np.asarray(crosslink_density)
        assert np.min(ρ) > 0, "Negative crosslink density encountered." 

            
        # Chain density (chains per unit volume)
        # Each crosslinker has coord chains, but each chain connects two
        # crosslinkers, so we divide by 2
        ν = ρ * coordination / 2.0
        # Polymer volume fraction φ = ν * v_chain
        ϕ = ν * self.vchain
        return ϕ
    
    def crosslink_density_from_polymer_volume_fraction(self, polymer_volume_fraction,coordination):
        """Compute crosslink density ρ from polymer volume fraction φ."""
        ϕ = np.asarray(polymer_volume_fraction)
        
        
        # Chain density ν = φ / v_chain
        ν = ϕ / self.vchain
        # Crosslink density ρ = 2 * ν / coord
        ρ = 2 * ν / coordination
        return ρ

    def per_volume(self, phi, der="0"):
        """
        Mixing free energy per unit volume. This is the standard flory huggins theory (except that it takes the crosslink density as input),
        we define it here for purpose of analytical computations

        Here chain_density is the real chain density, including the self contribution
        Derivatives with respect to the chain_density or the polymer volume fraction can be computed
        """

        # ϕ = self.polymer_volume_fraction_from_chain_density(chain_density)
        ϕ = phi
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
        else:
            raise ValueError(f"Invalid derivative option: {der}. Must be '0', 'phi', or 'phi2'.")


    def pressure(self, polymer_volume_fraction):
        """Compute osmotic pressure"""
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

    def __call__(self, rho, phi):
        """Compute embedding energy F(ρ).

        Parameters
        ----------
        rho : array_like
            Local crosslinker density at each crosslinker (including self),
            i.e., rho = sum_j W(r_ij) where the sum includes i=j.
            Note that the rho as defined in EAM implementations usually excludes
            the self-contribution W(0).
        phi : array_like
            Polymer volume fraction at each crosslinker . vchain * chain_density 

        Returns
        -------
        energy : array_like
            Embedding energy at each crosslinker
        """
        rho = np.asarray(rho)

        # Volume per crosslinker
        vi = 1.0 / rho

        # Free energy per crosslinker
        return self.per_volume(phi) * vi

    def derivative_rho(self, rho, phi):
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
        return  self.per_volume(phi) * vip

    
    def derivative_phi(self, rho, phi):
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
        return  self.per_volume(phi, der='phi') * vi
