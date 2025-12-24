#
# Copyright 2024 Lars Pastewka (University of Freiburg)
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

"""Tests for the hydrogel calculator."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from matscipy.calculators.hydrogel import (FloryHuggins, Hydrogel,
                                           LangevinChain, LucyWeightFunction,
                                           LucyWeightFunction2D)
from matscipy.molecules import Molecules
from matscipy.numerical import numerical_forces, numerical_stress

# ============== Weight Function Tests ==============


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
        E = fh(rho, w0)

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
        dF_num = (fh(rho + h, w0) - fh(rho - h, w0)) / (2 * h)

        # Analytical derivative
        dF_ana = fh.derivative(rho, w0)

        np.testing.assert_allclose(dF_ana, dF_num, rtol=1e-4)

    def test_second_derivative_numerical(self):
        """Test second derivative against numerical differentiation."""
        fh = FloryHuggins(
            chain_monomers=50,
            monomer_volume=4 * np.pi / 3,
            flory_chi=0.5,
            coordination=4,
        )
        w0 = 105.0 / (16.0 * np.pi * 20**3)
        rho = np.linspace(0.01, 0.1, 10)
        h = 1e-7

        # Numerical second derivative
        d2F_num = (fh.derivative(rho + h, w0) - fh.derivative(rho - h, w0)) / (2 * h)

        # Analytical second derivative
        d2F_ana = fh.second_derivative(rho, w0)

        np.testing.assert_allclose(d2F_ana, d2F_num, rtol=1e-4)


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


# ============== Hydrogel Calculator Tests ==============


def create_diamond_hydrogel(a=16.3, n_cells=1):
    """Create a diamond lattice hydrogel for testing.

    Parameters
    ----------
    a : float
        Lattice constant
    n_cells : int
        Number of unit cells in each direction

    Returns
    -------
    atoms : Atoms
        ASE Atoms object
    molecules : Molecules
        Molecules object with bond topology
    """
    from matscipy.neighbours import neighbour_list

    # Create diamond structure
    atoms = bulk("C", "diamond", a=a, cubic=True)
    atoms = atoms.repeat((n_cells, n_cells, n_cells))

    # Nearest neighbor distance in diamond
    nn_dist = a * np.sqrt(3) / 4

    # Get neighbor list with cutoff just above nearest neighbor distance
    i_p, j_p = neighbour_list("ij", atoms, nn_dist * 1.1)

    # Keep only unique bonds (i < j)
    mask = i_p < j_p
    bonds = np.column_stack([i_p[mask], j_p[mask]])

    molecules = Molecules(bonds_connectivity=bonds)

    return atoms, molecules


class TestHydrogelCalculator:
    """Tests for the Hydrogel calculator."""

    @pytest.fixture
    def simple_dimer(self):
        """Create a simple two-atom system with one bond."""
        atoms = Atoms(
            "CC", positions=[[0, 0, 0], [7.0, 0, 0]], cell=[20, 20, 20], pbc=True
        )
        molecules = Molecules(bonds_connectivity=[[0, 1]])
        return atoms, molecules

    @pytest.fixture
    def diamond_hydrogel(self):
        """Create a diamond lattice hydrogel."""
        return create_diamond_hydrogel(a=16.3, n_cells=1)

    def test_energy_finite(self, simple_dimer):
        """Test that energy is finite for simple system."""
        atoms, molecules = simple_dimer

        calc = Hydrogel(
            cutoff=15.0, chain_monomers=50, kuhn_length=1.0, molecules=molecules
        )
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

    def test_forces_sum_to_zero(self, simple_dimer):
        """Test that total force sums to zero (momentum conservation)."""
        atoms, molecules = simple_dimer

        calc = Hydrogel(
            cutoff=15.0, chain_monomers=50, kuhn_length=1.0, molecules=molecules
        )
        atoms.calc = calc

        forces = atoms.get_forces()
        total_force = np.sum(forces, axis=0)

        np.testing.assert_allclose(total_force, [0, 0, 0], atol=1e-10)

    def test_forces_numerical(self, simple_dimer):
        """Test forces against numerical differentiation."""
        atoms, molecules = simple_dimer

        calc = Hydrogel(
            cutoff=15.0, chain_monomers=50, kuhn_length=1.0, molecules=molecules
        )
        atoms.calc = calc

        f_analytical = atoms.get_forces()
        f_numerical = numerical_forces(atoms, d=1e-5)

        np.testing.assert_allclose(f_analytical, f_numerical, rtol=1e-4, atol=1e-8)

    def test_stress_numerical(self, diamond_hydrogel):
        """Test stress against numerical differentiation."""
        atoms, molecules = diamond_hydrogel

        calc = Hydrogel(
            cutoff=20.0, chain_monomers=50, kuhn_length=1.0, molecules=molecules
        )
        atoms.calc = calc

        s_analytical = atoms.get_stress()
        s_numerical = numerical_stress(atoms, d=1e-5)

        np.testing.assert_allclose(s_analytical, s_numerical, rtol=1e-3, atol=1e-8)

    def test_bond_energy_distance_dependence(self, simple_dimer):
        """Test that bond energy change follows Langevin chain formula."""
        atoms, molecules = simple_dimer

        # Use very small cutoff so only bond term contributes to energy changes
        calc = Hydrogel(
            cutoff=1.0,  # Very small cutoff - no embedding pairs
            chain_monomers=50,
            kuhn_length=1.0,
            molecules=molecules,
        )

        # Compare energy at different bond lengths
        chain = LangevinChain(kuhn_length=1.0, chain_monomers=50)

        r1 = 7.0
        r2 = 10.0

        # Energy at r1
        atoms.positions[1] = [r1, 0, 0]
        atoms.calc = calc
        E1 = atoms.get_potential_energy()

        # Energy at r2
        atoms.positions[1] = [r2, 0, 0]
        E2 = atoms.get_potential_energy()

        # Energy difference should match chain formula
        delta_E_calc = E2 - E1
        delta_E_chain = chain(np.array([r2]))[0] - chain(np.array([r1]))[0]

        np.testing.assert_allclose(delta_E_calc, delta_E_chain, rtol=1e-4)

    def test_diamond_equilibrium(self):
        """Test that diamond lattice is near equilibrium."""
        from matscipy.neighbours import neighbour_list

        # Set lattice parameter so nearest neighbor distance = R0
        R0 = np.sqrt(50) * 1.0  # Equilibrium distance for N=50, b=1
        a = R0 * 4 / np.sqrt(3)

        atoms = bulk("C", "diamond", a=a, cubic=True)
        nn_dist = a * np.sqrt(3) / 4
        i_p, j_p = neighbour_list("ij", atoms, nn_dist * 1.1)
        mask = i_p < j_p
        bonds = np.column_stack([i_p[mask], j_p[mask]])
        molecules = Molecules(bonds_connectivity=bonds)

        calc = Hydrogel(
            cutoff=3 * R0, chain_monomers=50, kuhn_length=1.0, molecules=molecules
        )
        atoms.calc = calc

        forces = atoms.get_forces()
        max_force = np.max(np.abs(forces))

        # Forces should be relatively small at equilibrium
        assert max_force < 1.0  # This is a weak test


# ============== Integration Tests ==============


class TestHydrogelIntegration:
    """Integration tests combining multiple components."""

    def test_lammps_comparison_parameters(self):
        """Test that parameters match LAMMPS hydrogel implementation."""
        # Parameters from LAMMPS example
        N = 50  # Chain monomers
        b = 1.0  # Kuhn length
        chi = 0.5  # Flory parameter
        coord = 4  # Coordination
        v0 = 4 * np.pi / 3  # Monomer volume

        R0 = np.sqrt(N) * b  # Equilibrium distance
        L0 = (N - 1) * b  # Contour length

        # Create simple system
        atoms = Atoms(
            "CC", positions=[[0, 0, 0], [R0, 0, 0]], cell=[50, 50, 50], pbc=True
        )
        molecules = Molecules(bonds_connectivity=[[0, 1]])

        calc = Hydrogel(
            cutoff=3 * R0,
            chain_monomers=N,
            kuhn_length=b,
            monomer_volume=v0,
            flory_chi=chi,
            coordination=coord,
            molecules=molecules,
        )
        atoms.calc = calc

        # Verify properties
        assert calc.N == N
        assert calc.b == b
        assert calc.chi == chi
        assert calc.coord == coord
        assert calc.v0 == v0
        assert calc.chain.L0 == L0

        # Energy should be finite
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
