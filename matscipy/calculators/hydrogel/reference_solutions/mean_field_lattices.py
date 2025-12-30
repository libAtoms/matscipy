"""
Analytical solutions for simple lattices using the mean-field Flory-Huggins theory  
and assuming affine elastic deformation of ideal Gaussian chains.
"""

from typing import Optional
from matscipy.calculators.hydrogel.potentials import GaussianChain, FloryHuggins
import numpy as np

CROSSLINK_VOLUMES={
"diamond":  (4 / np.sqrt(3)) ** 3 / 8,
"graphite":  3 * np.sqrt(3) / 2 /2,
"square": 1,
"cubic": 1 ,
}

COORDINATIONS= {
"diamond":4,
"graphie": 3, 
"square": 4, 
"cubic": 6, 
}



class MeanFieldHydrogelLattice():

    def __init__(self, chain_nb_monomers: int, coordination: int, 
                       vpcl_factor: float, 
                       flory_chi: float, kuhn: float = 1, v0: Optional[float] = None, dim: int = 3):
        """
        Initialize a mean-field hydrogel model.

        Parameters
        ----------
        chain_nb_monomers : int, optional
            Number of monomers per chain.
        coordination : float, optional
            Coordination number of the network.
        vpcl_factor: float
            volume per crosslink for a crosslink distance of 1
        flory_chi : float, optional
            Flory-Huggins interaction parameter (default is -1).
        kuhn : float, optional
            Kuhn length (default is 1).
        v0 : float, optional
            Volume per monomer. If None, calculated from dimension and Kuhn length, assume a monomer is a sphere with diameter the kuhn length (default is None).
        dim : int, optional
            Spatial dimension, 2 or 3 (default is 3).
        """
        self.chain_nb_monomers = chain_nb_monomers
        self.kuhn = kuhn
        self.dim = dim

        if v0 is None:
            if self.dim == 2:
                self.v0 = np.pi * (0.5 * kuhn)**2
            elif self.dim == 3:
                self.v0 = 4/3 * np.pi * (0.5 * kuhn)**3
        else:
            self.v0 = v0
        self.flory_chi = flory_chi
        self.coordination = coordination
        self.vpcl_factor = vpcl_factor
        self.chain = GaussianChain(kuhn_length=1, chain_monomers=chain_nb_monomers, dim=dim)

        self._flory_huggins = FloryHuggins(chain_nb_monomers, self.v0, flory_chi, coordination) 

    def vpcl(self, r,):
        """Volume per atom (crosslink) at crosslink distance r"""
        return self.vpcl_factor * r ** (self.dim)

    def crosslink_density(self, r):
        """Crosslink density at crosslink distance r"""
        return 1 / self.vpcl(r)
    
    def radius_from_density(self, density):
        """
        Compute the radius corresponding to a given crosslink density in the network lattice
        """
        return (self.vpcl(1) * density)**(-1/self.dim)

    def emixv(self, ρ,J: float=1.):
        """
        Mixing free energy per unit reference volume as function of the crosslink density ρ

        If J != 1, this corresponds to the mixing free energy per __reference volume__ (corresponding to the density ρ) 
        of the deformed state with density ρ / J 
        """
        pervolume= self._flory_huggins.per_volume(ρ / J) 
        perrefvolume = pervolume  * J
        return perrefvolume

    def demixv_dJ(self, rho: float, J: float=1.):
        """
        First derivative of mixing free energy per unit reference volume with respect to J.

        Args:
            ρ (float): Initial crosslink density
            J (float): Deformation gradient determinant (default: 1.0)
            
        Returns:
            float: First derivative of mixing free energy with respect to J
        """
        ρJ = rho / J
        
        # First term: ΔF_mix(φ(J))/V
        f_mix = self._flory_huggins.per_volume(ρJ, der="0")
        
        df_drho = self._flory_huggins.per_volume(ρJ, der="rho")
        return f_mix - ρJ * df_drho
        
    def ddemixv_ddJ(self, rho: float, J: float=1.):
        """
        Second derivative of mixing free energy per unit reference volume with respect to J.

        Args:
            rho (float): Initial crosslink density
            J (float): Deformation gradient determinant (default: 1.0)
            
        Returns:
            float: Second derivative of mixing free energy with respect to J
        """
        ρJ = rho / J
                                                       
        d2f_drho2 = self._flory_huggins.per_volume(ρJ, der="rho2")
        
        return  ρJ**2 /J * d2f_drho2

    def _parse_rho(self, rho, r):
        """
        allows to provide either rho or r as input
        """
        if rho is None and r is not None:
            rho = self.crosslink_density(r)
        else:
            assert rho is not None, "Either rho or r must be provided"
        return rho

    def C44(self, rho=None, r=None):
        """
        rho: crosslink density 
        r: intercrosslink distance 
        One of these two values must be provided
        """
        rho = self._parse_rho(rho, r)

        return - self.demixv_dJ(rho, 1.)


    def C11(self, rho=None, r=None):
        rho = self._parse_rho(rho, r)

        return self.ddemixv_ddJ(rho, 1.) - self.demixv_dJ(rho, 1.)

    def C12(self, rho=None, r=None):
        rho = self._parse_rho(rho, r)
        return  self.ddemixv_ddJ(rho, 1.) + self.demixv_dJ(rho, 1.)

    def bulk_modulus(self, rho=None, r=None) -> float:
        rho = self._parse_rho(rho, r)
        if self.dim == 2:
            return self.ddemixv_ddJ(rho, 1.)
        elif self.dim == 3:
            return (self.ddemixv_ddJ(rho, 1.) + self.demixv_dJ(rho, 1.) / 3  )
        else: 
            raise ValueError("Dimension must be 2 or 3") 

    def shear_modulus(self, rho=None, r=None) -> float:
        rho = self._parse_rho(rho, r)
        return self.C44(rho=rho)

    def youngs_modulus(self, rho=None, r=None) -> float:
        rho = self._parse_rho(rho, r)
        K = self.bulk_modulus(rho=rho)
        G = self.shear_modulus(rho=rho)
        return 9 * K * G / (3 * K + G)
    
    def poisson_ratio(self, rho=None, r=None):
        rho = self._parse_rho(rho, r)
        K = self.bulk_modulus(rho=rho)
        G = self.shear_modulus(rho=rho)
        return (3 * K - 2 * G )/ (2 * (3* K + G))

    @property
    def rms_end_to_end_distance(self):
        return np.sqrt(self.chain_nb_monomers) * self.kuhn

    def monomer_volume_fraction(self, r):
        """ Monomer volume fraction at distance r
        """
        ρ = self.crosslink_density(r)
        vchain = self.v0 * self.chain_nb_monomers
        ν = ρ * self.coordination / 2
        ϕ = ν * vchain 
        return ϕ
    

    @property
    def max_crosslink_density(self):
        """ Maximum allowed crosslink density where the monomoer volume fraction is 1
        """
        return 2 / (self.coordination * self.v0 * self.chain_nb_monomers)
    
    @property
    def min_radius(self):
        """ Minimum radius where the monomoer volume fraction is 1
        """
        return self.radius_from_density(self.max_crosslink_density)

    def total_energy(self, r):
        '''
        Total free energy per crosslink using mean field flory huggins theory
        and ideal chain elasticity
        '''
        return self.mixing_energy(r) + self.elastic_energy(r)
    
    def elastic_energy(self, r):
        """ Elastic free energy density at distance r, per crosslink, in units of kT, assuming affine deformation
        """
        return 0.5 * self.coordination * self.chain(r)

    def mixing_energy(self, r):
        '''
        Mixing free energy per crosslink using mean field flory huggins theory in units of kT
        '''
        return self.emixv(self.crosslink_density(r)) * self.vpcl(r)

    def compute_equilibrium_distance(self, tol=0.0001):
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(self.total_energy, bounds=(0.1 * self.min_radius, 0.9 * self.kuhn * self.chain_nb_monomers), method='bounded', tol=tol)
        assert res.success, f'minimize_scalar failed, {res.message}'
        return res.x




    
        