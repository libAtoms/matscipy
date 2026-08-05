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

from matscipy.calculators.manybody.newmb import Manybody
from matscipy.calculators.manybody.potentials import HarmonicPair, ZeroAngle

from ...elasticity import full_3x3_to_Voigt_6_stress
from ...neighbours import MolecularNeighbourhood, coordination, neighbour_list
from ..calculator import MatscipyCalculator
from .embedding import EmbeddingPotential, FloryHugginsPotential
from .network import LangevinChain
from .weight_functions import LucyWeightFunction, LucyWeightFunction2D, WeightFunction, WeightFunction

from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator


class Embedding(Calculator):
    implemented_properties = [
        'energy',
        'free_energy',
        'forces',
        'stress',
    ]

    default_parameters = {}
    name = 'Embedding'

    def __init__(self, weight_func: WeightFunction, 
                 embedding_potential: FloryHugginsPotential,
                 chain_neighbourhood: MolecularNeighbourhood):
        super().__init__()

        self.weight_func = weight_func
        self.embedding = embedding_potential
        self.chain_neighbourhood = chain_neighbourhood
        
    def _compute_density(self, atoms):
        """Compute local crosslinker density at each crosslinker.

        The density is computed as rho_i = sum_{j} W(r_ij), i.e., the
        sum of weight function contributions from all neighbors including
        self.

        Returns
        -------
        rho : ndarray
            Local crosslinker density at each atom (including self)
        nu : ndarray
            Local chain density at each atom (including self)
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

        # compute the numver of chains connected to each crosslinker (coordination)
        coordination = np.bincount(self.chain_neighbourhood.get_pairs(atoms, 'i'))

        # Get neighbor list
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms, self.weight_func.cutoff)

        # Compute weight function for each pair
        w_p = self.weight_func(r_p,)

        # Sum up crosslinker density contributions (excluding self)
        # rho_i = sum_{j != i} W(r_ij)
        rho = np.bincount(i_p, weights=w_p, minlength=nat)
        
        # chain density nu = rho * coord / 2 (including self contribution)
        nu = np.bincount(i_p, weights=w_p * coordination[i_p] / 2, minlength=nat)

        # Self-contribution returned separately
        w0 = self.weight_func.at_zero()

        return rho + w0, nu + coordination / 2 * w0, coordination, i_p, j_p, r_p, r_pc, w_p
    
    def calculate(self, atoms, properties, system_changes):
        """Calculate energy, forces, and stress."""
        super().calculate(atoms, properties, system_changes)

        nat = len(atoms)

        # ========== Flory-Huggins (embedding) contribution ==========

        # Compute density
        rho, nu, coordination, i_p, j_p, r_p, r_pc, w_p = self._compute_density(atoms)

        vchain = self.embedding.vchain
        phi = nu * vchain
        
        # Embedding energy
        E_embed = np.sum(self.embedding(rho, phi))


        # Embedding derivative dF/dρ for forces
        dF_drho = self.embedding.derivative_rho(rho, phi)
        dF_dphi = self.embedding.derivative_phi(rho, phi)

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
            -0.5 * (dF_drho[i_p[mask]] + dF_dphi[i_p[mask]] * coordination[i_p[mask]] / 2 * vchain 
                    + dF_drho[j_p[mask]] + dF_dphi[j_p[mask]] * coordination[j_p[mask]] / 2 * vchain)
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

        epot = E_embed 
        forces = f_embed_nc 
        virial = virial_embed 

        # Convert virial to stress (EAM convention: stress = virial / V)
        stress = virial / atoms.get_volume()

        self.results.update({
            'energy': epot,
            'free_energy': epot,
            'forces': forces,
            'stress': full_3x3_to_Voigt_6_stress(stress),
            # Additional quantities for information 
            'embedding_energy': E_embed,
        })





class Hydrogel(SumCalculator):
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

    def __init__(self, cutoff, chain_monomers, kuhn_length, molecules,
                 monomer_volume=None, flory_chi=0.5,
                 chain=None, dim=3):

        self.cutoff = cutoff

        # Default monomer volume: sphere of diameter b
        if monomer_volume is None:
            v0 = 4.0 * np.pi / 3.0 * (kuhn_length / 2.0)**3 if dim == 3 else (kuhn_length / 2.)**2 * np.pi

        else:
            v0 = monomer_volume


        if chain is None:
            chain = LangevinChain(kuhn_length, chain_monomers)


        neigh = MolecularNeighbourhood(molecules)

        self.embedding_calculator = Embedding(
            weight_func=LucyWeightFunction(cutoff) if dim==3 else LucyWeightFunction2D(cutoff), 
            embedding_potential=FloryHugginsPotential(chain_monomers, v0, flory_chi), 
            chain_neighbourhood=neigh, 
            )

        self.network_calculator = Manybody({1: chain.to_manybody_phi()}, 
                                           {1: ZeroAngle()}, neighbourhood=neigh)


        super().__init__([self.embedding_calculator, self.network_calculator])

        # Store bond topology
        self._molecules = molecules

    
