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

"""Hydrogel calculator for coarse-grained hydrogel simulations.

This calculator combines Flory-Huggins mixing free energy with Langevin
chain conformational free energy for polymer network simulations.
"""

import numpy as np
from ase.geometry import find_mic

from ...elasticity import full_3x3_to_Voigt_6_stress
from ...neighbours import neighbour_list
from ..calculator import MatscipyCalculator
from .potentials import FloryHuggins, LangevinChain, LucyWeightFunction, LucyWeightFunction2D


class Hydrogel(MatscipyCalculator):
    """ASE calculator for coarse-grained hydrogel simulations.

    This calculator implements a coarse-grained model for hydrogels where
    crosslinkers are represented as particles connected by polymer chains.
    The potential energy consists of two contributions:

    1. Flory-Huggins mixing free energy (density-dependent repulsion):
       An EAM-like embedding term that accounts for polymer-solvent
       interactions based on local monomer density.

    2. Langevin chain conformational free energy (bond stretching):
       Entropic elasticity of polymer chains connecting crosslinkers,
       using the inverse Langevin function for finite extensibility.

    Parameters
    ----------
    cutoff : float
        Cutoff radius for density calculation
    chain_monomers : float
        Number of monomers N per chain
    kuhn_length : float
        Kuhn length b of the polymer (sets length scale)
    monomer_volume : float
        Volume v₀ of a monomer (default: 4π/3 for sphere of diameter b)
    flory_chi : float
        Flory-Huggins interaction parameter χ (default: 0.5 for theta solvent)
    coordination : float
        Number of chains per crosslinker (default: 4 for diamond lattice)
    bonds : array_like, optional
        Bond connectivity as (n_bonds, 2) array of atom indices.
        If None, bonds must be provided via Molecules object.
    molecules : Molecules, optional
        Molecules object containing bond topology. Alternative to bonds.

    Notes
    -----
    The model uses reduced units where kT = 1. To convert results:
    - Energy: multiply by kT
    - Force: multiply by kT/b
    - Pressure/elastic constants: multiply by kT/b³

    Example
    -------
    >>> from ase import Atoms
    >>> from matscipy.calculators.hydrogel import Hydrogel
    >>> from matscipy.molecules import Molecules
    >>>
    >>> # Create atoms and bond topology
    >>> atoms = Atoms(...)
    >>> molecules = Molecules(bonds_connectivity=[[0, 1], [1, 2], ...])
    >>>
    >>> # Create calculator
    >>> calc = Hydrogel(
    ...     cutoff=21.0,
    ...     chain_monomers=50,
    ...     kuhn_length=1.0,
    ...     molecules=molecules
    ... )
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
    """

    implemented_properties = [
        'energy',
        'free_energy',
        'forces',
        'stress',
    ]

    default_parameters = {}
    name = 'Hydrogel'

    def __init__(self, cutoff, chain_monomers, kuhn_length,
                 monomer_volume=None, flory_chi=0.5, coordination=4,
                 molecules=None, chain=None, dim=3):
        super().__init__()

        self.N = chain_monomers
        self.b = kuhn_length
        self.chi = flory_chi
        self.coord = coordination

        self.dim = dim

        # Default monomer volume: sphere of diameter b
        if monomer_volume is None:
            self.v0 = 4.0 * np.pi / 3.0 * (kuhn_length / 2.0)**3 if dim == 3 else (kuhn_length / 2.)**2 * np.pi

        else:
            self.v0 = monomer_volume

        # Create potential objects
        self.weight_func = LucyWeightFunction(cutoff) if dim==3 else LucyWeightFunction2D(cutoff)
        self.embedding = FloryHuggins(chain_monomers, self.v0, flory_chi, coordination)
        
        if chain is not None:
            self.chain = chain
        else:
            self.chain = LangevinChain(kuhn_length, chain_monomers)

        # Store bond topology

        self._molecules = molecules

    @property
    def bonds(self):
        """Return bond connectivity array."""
        return self._molecules.bonds['atoms']

    def _compute_density(self, atoms):
        """Compute local crosslinker density at each crosslinker.

        The density is computed as rho_i = sum_{j} W(r_ij), i.e., the
        sum of weight function contributions from all neighbors including
        self.

        Returns
        -------
        rho : ndarray
            Local crosslinker density at each atom (including self)
        i_p, j_p : ndarray
            Neighbor pair indices
        r_p : ndarray
            Pair distances
        r_pc : ndarray
            Pair distance vectors
        w_p : ndarray
            Weight function values for each pair

        """
        nat = len(atoms)

        # Get neighbor list
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms, self.weight_func.cutoff)

        # Compute weight function for each pair
        w_p = self.weight_func(r_p,)

        # Sum up crosslinker density contributions (excluding self)
        # rho_i = sum_{j != i} W(r_ij)
        rho = np.bincount(i_p, weights=w_p, minlength=nat)

        # Self-contribution returned separately
        w0 = self.weight_func.at_zero()

        return rho + w0, i_p, j_p, r_p, r_pc, w_p

    def _compute_bond_vectors(self, atoms):
        """Compute bond vectors with periodic boundary conditions.

        Returns
        -------
        r_b : ndarray
            Bond lengths
        r_bc : ndarray
            Bond distance vectors
        """
        if self.bonds is None or len(self.bonds) == 0:
            return np.array([]), np.zeros((0, 3))

        # Get positions of bonded atoms
        pos_i = atoms.positions[self.bonds[:, 0]]
        pos_j = atoms.positions[self.bonds[:, 1]]

        # Compute distance vectors with minimum image convention
        r_bc, r_b = find_mic(pos_j - pos_i, atoms.cell, atoms.pbc)

        return r_b, r_bc

    def calculate(self, atoms, properties, system_changes):
        """Calculate energy, forces, and stress."""
        super().calculate(atoms, properties, system_changes)

        nat = len(atoms)

        # ========== Flory-Huggins (embedding) contribution ==========

        # Compute density
        rho, i_p, j_p, r_p, r_pc, w_p = self._compute_density(atoms)

        # Embedding energy
        E_embed = np.sum(self.embedding(rho))

        # Embedding derivative dF/dρ for forces
        dF_drho = self.embedding.derivative(rho)

        # Weight function derivative
        dw_p = self.weight_func.derivative(r_p)

        # Force from embedding term (EAM-like)
        # f_i = -dE/dr_i = -sum_j (dF/dρ_i + dF/dρ_j) * dW/dr * r_ij/|r_ij|
        # Since rho_i = sum_k W(r_ik), we have dρ_i/dr_ij = dW/dr * r_ij/r

        # df_pc contains the pair force contributions
        # Factor of 0.5 because neighbor list has both (i,j) and (j,i)
        df_embed_pc = np.zeros_like(r_pc)
        mask = r_p > 1e-10
        # Compute force factor for each pair
        force_factor = (
            -0.5 * (dF_drho[i_p[mask]] + dF_drho[j_p[mask]])
            * dw_p[mask] / r_p[mask]
        )
        df_embed_pc[mask] = force_factor[:, np.newaxis] * r_pc[mask]

        # Accumulate forces
        # df_embed_pc represents the pairwise force contribution
        # Force on atom i_p from pair, force on atom j_p is opposite
        f_embed_nc = np.zeros((nat, 3))
        for c in range(3):
            f_embed_nc[:, c] = (
                np.bincount(j_p, weights=df_embed_pc[:, c], minlength=nat)
                - np.bincount(i_p, weights=df_embed_pc[:, c], minlength=nat)
            )

        # Virial from embedding term (following EAM convention)
        # virial_ab = -sum_{ij} df_ij^a * r_ij^b
        virial_embed = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                virial_embed[a, b] = -np.sum(df_embed_pc[:, a] * r_pc[:, b])

        # ========== Langevin chain (bond) contribution ==========

        E_bond = 0.0
        f_bond_nc = np.zeros((nat, 3))
        virial_bond = np.zeros((3, 3))

        if self.bonds is not None and len(self.bonds) > 0:
            r_b, r_bc = self._compute_bond_vectors(atoms)

            # Bond energy
            E_bond = np.sum(self.chain(r_b))

            # Bond forces
            # For bond stretching with energy E(r), force on atom i is:
            # f_i = -dE/dr_i = -dE/dr * dr/dr_i = -dE/dr * (-r_bc/r) = +dE/dr * r_bc/r
            # This gives attractive force toward j when dE/dr > 0
            dE_dr = self.chain.derivative(r_b)
            mask_b = r_b > 1e-10
            df_bond_bc = np.zeros_like(r_bc)
            df_bond_bc[mask_b] = (
                dE_dr[mask_b, np.newaxis] * r_bc[mask_b] / r_b[mask_b, np.newaxis]
            )

            # Accumulate bond forces
            # Force on atom i (bonds[:, 0]): +df_bond
            # Force on atom j (bonds[:, 1]): -df_bond
            for c in range(3):
                f_bond_nc[:, c] += np.bincount(
                    self.bonds[:, 0], weights=df_bond_bc[:, c], minlength=nat
                )
                f_bond_nc[:, c] -= np.bincount(
                    self.bonds[:, 1], weights=df_bond_bc[:, c], minlength=nat
                )

            # Virial from bond term (following EAM convention)
            # virial_ab = -sum df^a * r^b where df is force on j (= -df_bond_bc)
            # and r is from i to j (= r_bc)
            # virial = -sum(-df_bond_bc * r_bc) = +sum(df_bond_bc * r_bc)
            for a in range(3):
                for b in range(3):
                    virial_bond[a, b] = np.sum(df_bond_bc[:, a] * r_bc[:, b])

        # ========== Total ==========

        epot = E_embed + E_bond
        forces = f_embed_nc + f_bond_nc
        virial = virial_embed + virial_bond

        # Convert virial to stress (EAM convention: stress = virial / V)
        stress = virial / atoms.get_volume()

        self.results.update({
            'energy': epot,
            'free_energy': epot,
            'forces': forces,
            'stress': full_3x3_to_Voigt_6_stress(stress),
            # Additional quantities for information 
            'embedding_energy': E_embed,
            'bond_energy': E_bond,
        })
