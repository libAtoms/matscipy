"""
Tests for numerical verification of derivatives in ShellHydrogelCalculator.

This module tests that analytical derivatives match numerical derivatives
for all methods in the ShellHydrogelCalculator class.
"""

import pytest
import numpy as np

from matscipy.calculators.hydrogel.reference_solutions.shell_calculator import ShellHydrogelCalculator
from matscipy.calculators.hydrogel.reference_solutions.lattice_shell_structures import GraphiteShellStructure
from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import MeanFieldHydrogelLattice, CROSSLINK_VOLUMES
from matscipy.calculators.hydrogel import (
    GaussianChain, 
    LucyWeightFunction2D
)
from matscipy.calculators.hydrogel.embedding_constant_coordination import FloryHugginsPotential 

from hydrogel.reference_solutions.test_shell_calculator_stiffness_consistency import meanfield


class TestShellCalculatorDerivatives:
    """Test analytical derivatives against numerical derivatives."""

    @pytest.fixture(scope="class")
    def meanfield(self):
        """Create a mean field lattice for comparison."""
        return MeanFieldHydrogelLattice(
            chain_nb_monomers=50, 
            coordination=3,
            vpcl_factor=CROSSLINK_VOLUMES['graphite'], 
            flory_chi=0.5,
            dim=2
        )

    @pytest.fixture(scope="class")
    def shell_calculator(self, meanfield):
        """Create a shell calculator instance for testing."""
        # Set up shell structure
        shell_structure = GraphiteShellStructure(cutoff=4.0)
        
        # Set up chain potential
        chain_potential = GaussianChain(kuhn_length=1.0, chain_monomers=50, dim=2)
        
        req_mf = meanfield.compute_equilibrium_distance()

        # Set up embedding potential (Flory-Huggins)
        embedding_potential = FloryHugginsPotential (
            chain_monomers=50,
            monomer_volume=np.pi *(1/2)**2,  # Assuming kuhn length = 1
            flory_chi=0.5,
            coordination=3,
        )
        
        # Set up weight function
        weight_function = LucyWeightFunction2D(cutoff=4.0 * req_mf)
        
        return ShellHydrogelCalculator(
            shell_structure=shell_structure,
            chain_potential=chain_potential,
            embedding_potential=embedding_potential,
            weight_function=weight_function
        )

    def test_density_derivative(self, shell_calculator):
        """Test density derivative against numerical differentiation."""
        # Test at several distances
        r_values = np.linspace(0.5, 3.0, 10)
        h = 1e-6
        
        for r in r_values:
            # Numerical derivative
            density_plus = shell_calculator.density(r + h)
            density_minus = shell_calculator.density(r - h)
            numerical_derivative = (density_plus - density_minus) / (2 * h)
            
            # Analytical derivative
            analytical_derivative = shell_calculator.density_derivative(r)
            
            np.testing.assert_allclose(
                analytical_derivative, 
                numerical_derivative, 
                rtol=1e-5,
                err_msg=f"Density derivative mismatch at r={r}"
            )

    def test_density_second_derivative(self, shell_calculator):
        """Test density second derivative against numerical differentiation."""
        # Test at several distances
        r_values = np.linspace(0.5, 3.0, 10)
        h = 1e-6
        
        for r in r_values:
            # Numerical second derivative
            deriv_plus = shell_calculator.density_derivative(r + h)
            deriv_minus = shell_calculator.density_derivative(r - h)
            numerical_second_derivative = (deriv_plus - deriv_minus) / (2 * h)
            
            # Analytical second derivative
            analytical_second_derivative = shell_calculator.density_second_derivative(r)
            
            np.testing.assert_allclose(
                analytical_second_derivative, 
                numerical_second_derivative, 
                rtol=1e-5,
                err_msg=f"Density second derivative mismatch at r={r}"
            )

    def test_embedding_energy_derivative(self, shell_calculator):
        """Test embedding energy derivative against numerical differentiation."""
        # Test at several distances
        r_values = np.linspace(0.5, 3.0, 10)
        h = 1e-6
        
        for r in r_values:
            # Numerical derivative
            energy_plus = shell_calculator.embedding_energy(r + h)
            energy_minus = shell_calculator.embedding_energy(r - h)
            numerical_derivative = (energy_plus - energy_minus) / (2 * h)
            
            # Analytical derivative
            analytical_derivative = shell_calculator.embedding_energy_derivative(r)
            
            np.testing.assert_allclose(
                analytical_derivative, 
                numerical_derivative, 
                rtol=1e-5,
                err_msg=f"Embedding energy derivative mismatch at r={r}"
            )

    def test_embedding_energy_second_derivative(self, shell_calculator):
        """Test embedding energy second derivative against numerical differentiation."""
        # Test at several distances
        r_values = np.linspace(0.5, 3.0, 10)
        h = 1e-6
        
        for r in r_values:
            # Numerical second derivative
            deriv_plus = shell_calculator.embedding_energy_derivative(r + h)
            deriv_minus = shell_calculator.embedding_energy_derivative(r - h)
            numerical_second_derivative = (deriv_plus - deriv_minus) / (2 * h)
            
            # Analytical second derivative
            analytical_second_derivative = shell_calculator.embedding_energy_second_derivative(r)
            
            np.testing.assert_allclose(
                analytical_second_derivative, 
                numerical_second_derivative, 
                rtol=1e-4,  # Slightly more relaxed for second derivatives
                err_msg=f"Embedding energy second derivative mismatch at r={r}"
            )

    def test_bond_energy_derivative(self, shell_calculator):
        """Test bond energy derivative against numerical differentiation."""
        # Test at several distances, but avoid getting too close to contour length
        L0 = shell_calculator.chain_potential.L0
        r_values = np.linspace(0.5, 0.7 * L0, 10)  # Stay well below contour length
        h = 1e-7
        
        for r in r_values:
            # Numerical derivative
            energy_plus = shell_calculator.bond_energy(r + h)
            energy_minus = shell_calculator.bond_energy(r - h)
            numerical_derivative = (energy_plus - energy_minus) / (2 * h)
            
            # Analytical derivative
            analytical_derivative = shell_calculator.bond_energy_derivative(r)
            
            np.testing.assert_allclose(
                analytical_derivative, 
                numerical_derivative, 
                rtol=1e-5,
                err_msg=f"Bond energy derivative mismatch at r={r}"
            )

    def test_bond_energy_second_derivative(self, shell_calculator):
        """Test bond energy second derivative against numerical differentiation."""
        # Test at several distances, but avoid getting too close to contour length
        L0 = shell_calculator.chain_potential.L0
        r_values = np.linspace(0.5, 0.6 * L0, 10)  # Stay well below contour length
        h = 1e-6
        
        for r in r_values:
            # Numerical second derivative
            deriv_plus = shell_calculator.bond_energy_derivative(r + h)
            deriv_minus = shell_calculator.bond_energy_derivative(r - h)
            numerical_second_derivative = (deriv_plus - deriv_minus) / (2 * h)
            
            # Analytical second derivative
            analytical_second_derivative = shell_calculator.bond_energy_second_derivative(r)
            
            np.testing.assert_allclose(
                analytical_second_derivative, 
                numerical_second_derivative, 
                rtol=1e-4,  # Slightly more relaxed for second derivatives
                err_msg=f"Bond energy second derivative mismatch at r={r}"
            )

    def test_total_energy_derivative(self, shell_calculator):
        """Test total energy derivative against numerical differentiation."""
        # Test at several distances
        L0 = shell_calculator.chain_potential.L0
        r_values = np.linspace(0.5, 0.7 * L0, 10)  # Stay well below contour length
        h = 1e-6
        
        for r in r_values:
            # Numerical derivative
            energy_plus = shell_calculator.energy(r + h)
            energy_minus = shell_calculator.energy(r - h)
            numerical_derivative = (energy_plus - energy_minus) / (2 * h)
            
            # Analytical derivative
            analytical_derivative = shell_calculator.energy_derivative(r)
            
            np.testing.assert_allclose(
                analytical_derivative, 
                numerical_derivative, 
                rtol=1e-5,
                err_msg=f"Total energy derivative mismatch at r={r}"
            )

    def test_total_energy_second_derivative(self, shell_calculator):
        """Test total energy second derivative against numerical differentiation."""
        # Test at several distances
        L0 = shell_calculator.chain_potential.L0
        r_values = np.linspace(0.5, 0.6 * L0, 10)  # Stay well below contour length
        h = 1e-6
        
        for r in r_values:
            # Numerical second derivative
            deriv_plus = shell_calculator.energy_derivative(r + h)
            deriv_minus = shell_calculator.energy_derivative(r - h)
            numerical_second_derivative = (deriv_plus - deriv_minus) / (2 * h)
            
            # Analytical second derivative
            analytical_second_derivative = shell_calculator.energy_second_derivative(r)
            
            np.testing.assert_allclose(
                analytical_second_derivative, 
                numerical_second_derivative, 
                rtol=1e-4,  # Slightly more relaxed for second derivatives
                err_msg=f"Total energy second derivative mismatch at r={r}"
            )

    def test_derivative_consistency(self, shell_calculator):
        """Test that the total energy derivatives are consistent with component derivatives."""
        # Test at several distances
        L0 = shell_calculator.chain_potential.L0
        r_values = np.linspace(0.5, 0.7 * L0, 10)
        
        for r in r_values:
            # Total energy derivatives
            total_first = shell_calculator.energy_derivative(r)
            total_second = shell_calculator.energy_second_derivative(r)
            
            # Sum of component derivatives
            embedding_first = shell_calculator.embedding_energy_derivative(r)
            bond_first = shell_calculator.bond_energy_derivative(r)
            sum_first = embedding_first + bond_first
            
            embedding_second = shell_calculator.embedding_energy_second_derivative(r)
            bond_second = shell_calculator.bond_energy_second_derivative(r)
            sum_second = embedding_second + bond_second
            
            np.testing.assert_allclose(
                total_first, 
                sum_first, 
                rtol=1e-12,
                err_msg=f"First derivative consistency check failed at r={r}"
            )
            
            np.testing.assert_allclose(
                total_second, 
                sum_second, 
                rtol=1e-12,
                err_msg=f"Second derivative consistency check failed at r={r}"
            )

    def test_equilibrium_condition(self, shell_calculator, meanfield):
        """Test that the equilibrium distance actually has zero derivative."""
        # Get meanfield equilibrium distance as initial guess
        r0_meanfield = meanfield.compute_equilibrium_distance()
        
        # Find equilibrium distance using meanfield guess
        r_eq = shell_calculator.compute_equilibrium_distance(r0=r0_meanfield)
        
        # Check that derivative is very close to zero
        deriv_at_equilibrium = shell_calculator.energy_derivative(r_eq)
        
        np.testing.assert_allclose(
            deriv_at_equilibrium, 
            0.0, 
            atol=1e-10,
            err_msg=f"Derivative at equilibrium r={r_eq} should be zero"
        )
        
        # Check that second derivative is positive (stable equilibrium)
        second_deriv_at_equilibrium = shell_calculator.energy_second_derivative(r_eq)
        
        assert second_deriv_at_equilibrium > 0, \
            f"Second derivative at equilibrium should be positive, got {second_deriv_at_equilibrium}"