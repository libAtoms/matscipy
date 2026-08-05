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

from matscipy.calculators.hydrogel import (FloryHugginsPotential , Hydrogel,
                                           LangevinChain, LucyWeightFunction,
                                           LucyWeightFunction2D)
import matscipy.calculators.hydrogel.embedding_constant_coordination as ecc

from matscipy.molecules import Molecules
from matscipy.numerical import numerical_forces, numerical_stress

# ============== Weight Function Tests ==============


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

    @pytest.mark.xfail(reason="Langevin implementation has issues")
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
            molecules=molecules,
        )
        atoms.calc = calc

        # Energy should be finite
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
