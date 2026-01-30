"""Test consistency between stiffness matrix and individual elastic constants in shell calculator."""

import pytest
import numpy as np

from matscipy.calculators.hydrogel.reference_solutions.lattice_shell_structures import GraphiteShellStructure
from matscipy.calculators.hydrogel.reference_solutions.shell_calculator import (
    ShellHydrogelCalculator, 
    Isotropic2DShellHydrogelCalculator
)
from matscipy.calculators.hydrogel.potentials import FloryHugginsPotential  , GaussianChain, LucyWeightFunction2D
from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import CROSSLINK_VOLUMES, MeanFieldHydrogelLattice


@pytest.fixture(params=[{
        'chain_monomers': 100,
        'flory_chi': -1.0,
}, 
{
        'chain_monomers': 50,
        'flory_chi': -1.0,
},
{
        'chain_monomers': 100,
        'flory_chi': 0.0,
},
{
        'chain_monomers': 100,
        'flory_chi': 1.,
}
], scope='module')
def parameters(request):
    return request.param


@pytest.fixture(scope="module")
def meanfield(parameters):    
    return MeanFieldHydrogelLattice(parameters['chain_monomers'], 
        3, CROSSLINK_VOLUMES['graphite'], 
        flory_chi = parameters['flory_chi'], dim=2)


@pytest.fixture(scope='module')
def shellstructure():
    return GraphiteShellStructure(cutoff=10.0)


@pytest.fixture(params=[2.0, 4.0, 5.0], scope='function')
def shell_calculator(request, parameters, meanfield, shellstructure):
    """Create shell calculator with different cutoff factors."""
    rc_factor = request.param
    
    N = meanfield.chain_nb_monomers
    χ = meanfield.flory_chi
    kuhn = meanfield.kuhn
    req_mf = meanfield.compute_equilibrium_distance()
    
    cutoff = rc_factor * req_mf
    
    calc = ShellHydrogelCalculator(
        shellstructure, 
        chain_potential=GaussianChain(1, N, dim=2),
        embedding_potential=FloryHugginsPotential (
            N,
            monomer_volume=meanfield.v0,
            flory_chi=χ,
            coordination=meanfield.coordination,
        ),
        weight_function=LucyWeightFunction2D(cutoff=cutoff)
    )
    
    return calc, req_mf


@pytest.fixture(params=[2.0, 4.0, 5.0], scope='function') 
def isotropic_2d_calculator(request, parameters, meanfield, shellstructure):
    """Create isotropic 2D shell calculator with different cutoff factors."""
    rc_factor = request.param
    
    N = meanfield.chain_nb_monomers
    χ = meanfield.flory_chi
    kuhn = meanfield.kuhn
    req_mf = meanfield.compute_equilibrium_distance()
    
    cutoff = rc_factor * req_mf
    
    calc = Isotropic2DShellHydrogelCalculator(
        shellstructure, 
        chain_potential=GaussianChain(1, N, dim=2),
        embedding_potential=FloryHugginsPotential (
            N,
            monomer_volume=meanfield.v0,
            flory_chi=χ,
            coordination=meanfield.coordination,
        ),
        weight_function=LucyWeightFunction2D(cutoff=cutoff)
    )
    
    return calc, req_mf


class TestStiffnessConsistency:
    """Test consistency between stiffness matrix and individual elastic constants."""
    
    def test_stiffness_matrix_vs_individual_constants(self, isotropic_2d_calculator):
        """Test that individual C11, C12, C44 match corresponding components of stiffness matrix."""
        calc, req_mf = isotropic_2d_calculator
        
        # Compute equilibrium distance
        r_eq = calc.compute_equilibrium_distance(req_mf)
        
        # Get individual elastic constants from specialized methods
        C11_individual = calc.C11(r_eq)
        C12_individual = calc.C12(r_eq)
        C44_individual = calc.C44(r_eq)
        
        # Get full stiffness matrix
        stiffness_matrix = calc.stiffness_matrix(r_eq)
        
        # Extract corresponding components from stiffness matrix
        # For 2D isotropic case:
        # C11 corresponds to stiffness_matrix[0,0,0,0] 
        # C12 corresponds to stiffness_matrix[0,0,1,1]
        # C44 corresponds to stiffness_matrix[0,1,0,1] (shear component)
        C11_matrix = stiffness_matrix[0,0,0,0]
        C12_matrix = stiffness_matrix[0,0,1,1] 
        C44_matrix = stiffness_matrix[0,1,0,1]
        
        # Test consistency with reasonable tolerance
        rtol = 1e-10  # Very tight tolerance since these should be identical
        
        assert np.isclose(C11_individual, C11_matrix, rtol=rtol), \
            f"C11 mismatch: individual={C11_individual}, matrix={C11_matrix}"
        
        assert np.isclose(C12_individual, C12_matrix, rtol=rtol), \
            f"C12 mismatch: individual={C12_individual}, matrix={C12_matrix}"
        
        assert np.isclose(C44_individual, C44_matrix, rtol=rtol), \
            f"C44 mismatch: individual={C44_individual}, matrix={C44_matrix}"
    
    def test_bulk_shear_modulus_consistency(self, isotropic_2d_calculator):
        """Test that bulk and shear moduli computed from individual methods are consistent."""
        calc, req_mf = isotropic_2d_calculator
        
        # Compute equilibrium distance
        r_eq = calc.compute_equilibrium_distance(req_mf)
        
        # Get individual elastic constants
        C11 = calc.C11(r_eq)
        C12 = calc.C12(r_eq) 
        C44 = calc.C44(r_eq)
        
        # Get bulk and shear moduli from convenience methods
        bulk_modulus_method = calc.bulk_modulus(r_eq)
        shear_modulus_method = calc.shear_modulus(r_eq)
        
        # Compute bulk and shear moduli from individual constants
        bulk_modulus_computed = (C11 + C12) / 2
        shear_modulus_computed = C44
        
        rtol = 1e-12
        
        assert np.isclose(bulk_modulus_method, bulk_modulus_computed, rtol=rtol), \
            f"Bulk modulus mismatch: method={bulk_modulus_method}, computed={bulk_modulus_computed}"
        
        assert np.isclose(shear_modulus_method, shear_modulus_computed, rtol=rtol), \
            f"Shear modulus mismatch: method={shear_modulus_method}, computed={shear_modulus_computed}"
    
    def test_stiffness_matrix_symmetries(self, shell_calculator):
        """Test that the stiffness matrix has the expected symmetries."""
        calc, req_mf = shell_calculator
        
        # Compute equilibrium distance
        r_eq = calc.compute_equilibrium_distance(req_mf)
        
        # Get stiffness matrix
        C = calc.stiffness_matrix(r_eq)
        
        # Test major symmetries: C_ijkl = C_jikl = C_ijlk = C_jilk
        rtol = 1e-12
        
        # Test symmetry in first two indices: C_ijkl = C_jikl
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        assert np.isclose(C[i,j,k,l], C[j,i,k,l], rtol=rtol), \
                            f"Symmetry C[{i},{j},{k},{l}] != C[{j},{i},{k},{l}]"
        
        # Test symmetry in last two indices: C_ijkl = C_ijlk  
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        assert np.isclose(C[i,j,k,l], C[i,j,l,k], rtol=rtol), \
                            f"Symmetry C[{i},{j},{k},{l}] != C[{i},{j},{l},{k}]"
        
        # Test major symmetry: C_ijkl = C_klij
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        assert np.isclose(C[i,j,k,l], C[k,l,i,j], rtol=rtol), \
                            f"Major symmetry C[{i},{j},{k},{l}] != C[{k},{l},{i},{j}]"
    
    def test_isotropic_2d_relationships(self, isotropic_2d_calculator):
        """Test relationships specific to 2D isotropic materials."""
        calc, req_mf = isotropic_2d_calculator
        
        # Compute equilibrium distance
        r_eq = calc.compute_equilibrium_distance(req_mf)
        
        # Get stiffness matrix
        C = calc.stiffness_matrix(r_eq)
        
        rtol = 1e-10
        
        # For 2D isotropic materials, we should have:
        # C_xxxx = C_yyyy (C11 components equal)
        assert np.isclose(C[0,0,0,0], C[1,1,1,1], rtol=rtol), \
            "C_xxxx should equal C_yyyy for 2D isotropic material"
        
        # C_xxyy = C_yyxx (C12 components equal)  
        assert np.isclose(C[0,0,1,1], C[1,1,0,0], rtol=rtol), \
            "C_xxyy should equal C_yyxx for 2D isotropic material"
        
        # C_xyxy = C_yxyx = C_xyyx = C_yxxy (all shear components equal)
        shear_components = [C[0,1,0,1], C[1,0,1,0], C[0,1,1,0], C[1,0,0,1]]
        for i, comp in enumerate(shear_components[1:], 1):
            assert np.isclose(shear_components[0], comp, rtol=rtol), \
                f"Shear component {i} differs from component 0 in 2D isotropic material"
    
    def test_positive_definite_stiffness_matrix(self, shell_calculator):
        """Test that the stiffness matrix is positive definite (stable material)."""
        calc, req_mf = shell_calculator
        
        # Compute equilibrium distance
        r_eq = calc.compute_equilibrium_distance(req_mf)
        
        # Get stiffness matrix
        C = calc.stiffness_matrix(r_eq)
        
        # Reshape to 2D matrix form for eigenvalue analysis
        # For 2D case, we have a 2x2x2x2 tensor
        # We can check positive definiteness by looking at the Voigt form
        C_voigt = np.zeros((3, 3))  # 2D Voigt notation: xx, yy, xy
        C_voigt[0,0] = C[0,0,0,0]  # xx,xx
        C_voigt[1,1] = C[1,1,1,1]  # yy,yy  
        C_voigt[2,2] = C[0,1,0,1]  # xy,xy
        C_voigt[0,1] = C_voigt[1,0] = C[0,0,1,1]  # xx,yy
        C_voigt[0,2] = C_voigt[2,0] = 0  # xx,xy should be zero for isotropic
        C_voigt[1,2] = C_voigt[2,1] = 0  # yy,xy should be zero for isotropic
        
        # Check that all eigenvalues are positive
        eigenvals = np.linalg.eigvals(C_voigt)
        
        assert np.all(eigenvals > 0), \
            f"Stiffness matrix has non-positive eigenvalues: {eigenvals}"
        
        # Also check individual elastic constants are positive  
        if isinstance(calc, Isotropic2DShellHydrogelCalculator):
            C11 = calc.C11(r_eq)
            C12 = calc.C12(r_eq)
            C44 = calc.C44(r_eq)
            
            assert C11 > 0, f"C11 should be positive: {C11}"
            assert C44 > 0, f"C44 should be positive: {C44}"
            assert C11 > abs(C12), f"Stability requires C11 > |C12|: C11={C11}, C12={C12}"


if __name__ == "__main__":
    pytest.main([__file__])