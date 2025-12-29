
from matscipy.calculators.hydrogel.reference_solutions.lattice_shell_structures import ShellStructure, GraphiteShellStructure
from matscipy.calculators.hydrogel.potentials import (ChainPotential, EmbeddingPotential, 
    WeightFunction)
from typing import Literal
import scipy
import numpy as np

class ShellHydrogelCalculator():
    """Hydrogel calculator using shell reference solutions."""
    dim: Literal[2, 3]

    def __init__(self, shell_structure: ShellStructure, 
                       chain_potential: ChainPotential, 
                       embedding_potential: EmbeddingPotential,
                       weight_function: WeightFunction
                       , ):
        """
        Initialize the shell hydrogel calculator.

        Parameters
        ----------
        shell_structure : ShellStructure
            The shell structure to use.
        chain_potential : ChainPotential
            The chain potential to use.
        flory_chi : float
            The Flory-Huggins interaction parameter.
        """
        self.shell_structure = shell_structure
        self.chain_potential = chain_potential
        self.embedding_potential = embedding_potential
        self.weight_function = weight_function
        self.dim = shell_structure.dim

    # some convenience functions to call weight function methods
    def f(self, r) -> float:
        return self.weight_function(r)
    
    def fp(self, r) -> float:
        return self.weight_function.derivative(r)
    def fpp(self, r) -> float:
        return self.weight_function.second_derivative(r)
    def ft(self, r2) -> float:
        return self.f(np.sqrt(r2))
    def ftp(self, r2) -> float:
        return self.fp(np.sqrt(r2)) / ( 2 * np.sqrt(r2))
    def ftpp(self, r2) -> float:
        r = np.sqrt(r2)
        return self.fpp(r) / (4 * r2) - self.fp(r) / (4 * r2 * r)

    def F(self, rho) -> float:
        return self.embedding_potential(rho)
    
    def Fp(self, rho) -> float:
        return self.embedding_potential.derivative(rho)
    
    def Fpp(self, rho) -> float:
        return self.embedding_potential.second_derivative(rho)
    
    def density_noself(self, r):
        """
        Electron density at distance r, excluding self-contribution
        """
        # First neighbors:
        rho = 0
        rn = self.shell_structure.a * r
        Zn = self.shell_structure.Z
        rho = np.sum( Zn * self.f(rn))
        return rho 
    
    def density(self, r):
        """
        Electron density at distance r, including self-contribution
        """
        # First neighbors:
        rho = 0
        rn = self.shell_structure.a * r
        Zn = self.shell_structure.Z
        rho = np.sum( Zn * self.f(rn))
        return rho  + self.f(0)
    
    def density_derivative(self, r) -> float:
        """
        Derivative of electron density at distance r, including self-contribution
        """
        # First neighbors: d/dr f(a*r) = f'(a*r) * a
        rho = 0
        rn = self.shell_structure.a * r
        Zn = self.shell_structure.Z
        an = self.shell_structure.a
        rho = np.sum( Zn * self.fp(rn) * an)
        return rho  # No self-contribution derivative since f'(0) * 0 = 0

    def density_second_derivative(self, r) -> float:
        """
        Second derivative of electron density at distance r, including self-contribution
        """
        # First neighbors: d²/dr² f(a*r) = f''(a*r) * a²
        rho = 0
        rn = self.shell_structure.a * r
        Zn = self.shell_structure.Z
        an = self.shell_structure.a
        rho = np.sum( Zn * self.fpp(rn) * an**2)
        return rho  # No self-contribution second derivative

    def embedding_energy(self, r) -> float:
        """ 
        EAM energy per atom at distance r
        
        Including contributions up to shell s

        """
        return self.F(self.density(r)) 

    def embedding_energy_derivative(self, r):
        """ 
        EAM energy per atom at distance r
        
        Including contributions up to shell s

        """
        return self.Fp(self.density(r)) * self.density_derivative(r)

    def embedding_energy_second_derivative(self, r):
        """ 
        EAM energy per atom at distance r
        
        Including contributions up to shell s

        """
        return self.Fpp(self.density(r)) * self.density_derivative(r)**2 + self.Fp(self.density(r)) * self.density_second_derivative(r)

    def bond_energy(self, r):
        """
        We assume that bonds exist only towards the first shell.
        """
        # Factor of 0.5 to account for double counting of bonds
        return self.chain_potential(r) * self.shell_structure.Z[0] * 0.5

    def bond_energy_derivative(self, r):
        """
        We assume that bonds exist only towards the first shell.
        """
        return self.chain_potential.derivative(r) * self.shell_structure.Z[0] * 0.5

    def bond_energy_second_derivative(self, r):
        """
        We assume that bonds exist only towards the first shell.
        """
        return self.chain_potential.second_derivative(r) * self.shell_structure.Z[0] * 0.5

    def energy(self, r):
        """
        Total energy per atom at distance r
        """
        return self.embedding_energy(r) + self.bond_energy(r)

    def energy_derivative(self, r):
        """
        Total energy derivative per atom at distance r
        """
        return self.embedding_energy_derivative(r) + self.bond_energy_derivative(r)
    
    def energy_second_derivative(self, r):
        """
        Total energy second derivative per atom at distance r
        """
        return self.embedding_energy_second_derivative(r) + self.bond_energy_second_derivative(r)

    def compute_equilibrium_distance(self, r0, tol=1e-6, maxiter=2000):
        """
        Compute the equilibrium distance by finding the root of the energy derivative.
        """
        res, success = scipy.optimize.newton(self.energy_derivative, fprime=self.energy_second_derivative, 
                                        x0=r0, tol=tol, maxiter=maxiter, full_output=True)
        assert success, f'Newton root finding failed: {res.message}'
        return res
        
    def stiffness_matrix(self, r):
        """
        Stiffness matrix from EAM potential
        """
        dim = self.dim
        ρ0 = self.density(r)

        a = self.shell_structure.a
        Z = self.shell_structure.Z
        term1 = 0
        term2 = 0

        # TODO: vectorize this ! 
        for n in range(0, len(a) ):
            # Note that here we define the shell tensor as nu / Z , where nu is the shell tensor as defined in Muser, Sukhomlinov, Pastewka
            term1 += self.ftpp(r ** 2  * a[n]**2) * r**4 * a[n]**4 * self.shell_structure.shellTensor4(n) * Z[n]
            for m in range(0, len(a)): 

                term2 += self.ftp(r**2 * a[n]**2) * self.ftp(r ** 2 * a[m]**2) * r**4 *  a[n]**2 * a[m]**2 * Z[n] * Z[m] \
                  * self.shell_structure.shellTensor2(n).reshape(dim, dim, 1, 1) * self.shell_structure.shellTensor2(m).reshape(1, 1, dim, dim)
        
        term1 *= self.Fp(ρ0)
        term2 *= self.Fpp(ρ0)

        return 4  * (
            term1
            + term2
            ) * self.density(r)
    

class Isotropic2DShellHydrogelCalculator(ShellHydrogelCalculator):
    """2D isotropic shell hydrogel calculator.
    
    This just implements some shortcuts for computing elastic constants in 2D isotropic case.
    """
    def __init__(self, shell_structure: ShellStructure, 
                       chain_potential: ChainPotential, 
                       embedding_potential: EmbeddingPotential,
                       weight_function: WeightFunction
                       , ):
        super().__init__(shell_structure, chain_potential, embedding_potential, weight_function)
        assert self.dim == 2, "Isotropic2DShellHydrogelCalculator only works for 2D shell structures."
    

    def C44(self, r):
        """
        Shear modulus C44 from EAM potential

        """
        ρ0 = self.density(r)

        a = self.shell_structure.a
        Z = self.shell_structure.Z
        
        # Sum runs over shells
        # My derivation yields an additional factor of 4. But here is the implementation of Muser Pastewka
        # TODO check this factor  4
        return 4 * self.Fp(ρ0) * np.sum(self.ftpp(r ** 2  * a**2) * r**4 * a**4  * Z / 8) * self.density(r)   
    
    def C11(self, r):
        """
        modulus C11 from EAM potential 
        

        """
        ρ0 = self.density(r)

        a = self.shell_structure.a
        Z = self.shell_structure.Z
        
        A = self.ftp(r**2 * a**2) * r**2 * a**2 * Z / 2
        return 4 * ( self.Fp(ρ0) * np.sum(self.ftpp(r ** 2  * a**2) * r**4 * a**4  * Z * 3 / 8 ) 
                    + self.Fpp(ρ0) * (np.sum(A.reshape(-1, 1) * A.reshape(1, -1)) )
                    ) * self.density(r)

    def C12(self, r):
        """
        modulus C12 from EAM potential 

        """
        ρ0 = self.density(r)

        a = self.shell_structure.a
        Z = self.shell_structure.Z
        A = self.ftp(r**2 * a**2) * r**2 * a**2 * Z / 2
        return 4 * ( self.Fp(ρ0) * np.sum(self.ftpp(r ** 2  * a**2) * r**4 * a**4  * Z * 1 / 8 ) 
                    + self.Fpp(ρ0) * (np.sum(A.reshape(-1, 1) * A.reshape(1, -1)) )
                    ) * self.density(r)
    
    def bulk_modulus(self, r):
        """
        Bulk modulus from EAM potential 

        """
        C11 = self.C11(r)
        C12 = self.C12(r)
        return (C11 + C12) / 2
    
    def shear_modulus(self, r):
        return self.C44(r)