#!/usr/bin/env python3
"""
Test script for 2D graphite (graphene) hydrogel structure creation
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.filters import UnitCellFilter
from ase.optimize import FIRE

# Import the factory function directly
from matscipy.calculators.hydrogel.snippets.factory import create_graphite_lattice
import pytest

from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import CROSSLINK_VOLUMES, MeanFieldHydrogelLattice

from matscipy.calculators.hydrogel.calculator import Hydrogel
from matscipy.calculators.hydrogel.potentials import GaussianChain

from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants

def test_2d_graphite_structure():
    """Test the 2D graphite hydrogel creation"""
    
    # Parameters
    crosslink_spacing = 1.42  # Typical C-C bond length in graphene (Angstrom)
    n_cells = 3  # 3x3 unit cells
    
    print("="*60)
    print("Testing truly 2D graphite (graphene) hydrogel structure...")
    print("="*60)
    
    # Create truly 2D structure
    atoms_2d, molecules_2d = create_graphite_lattice(
        crosslink_spacing=crosslink_spacing, 
        n_cells=n_cells, 
        truly_2d=True
    )
    

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
],scope='module')
def parameters(request):
    return request.param


@pytest.fixture(scope='module')
def graphite_meanfield(parameters, ):
    return MeanFieldHydrogelLattice(parameters['chain_monomers'], 
        3, CROSSLINK_VOLUMES['graphite'], 
        flory_chi = parameters['flory_chi'], dim=2)

@pytest.fixture(scope='module')
def graphite_calc_factory(graphite_meanfield):
    ana = graphite_meanfield
    N  = ana.chain_nb_monomers
    χ = ana.flory_chi
    kuhn = ana.kuhn

    req_mf = ana.compute_equilibrium_distance()
    Re = np.sqrt(N) * kuhn  # RMS end-to-end distance of a free chain

    def factory(rc_factor):
        rc = rc_factor * req_mf
    
        atoms, molecules = create_graphite_lattice(crosslink_spacing=req_mf, 
                                                   n_cells=np.ceil(1.5 * rc / req_mf).astype(int))
        calc = Hydrogel(
            cutoff=rc,
            chain_monomers=N,
            kuhn_length=kuhn,
            monomer_volume=ana.v0,
            flory_chi=χ ,
            coordination=ana.coordination,
            molecules=molecules,
            chain=GaussianChain(
                kuhn_length=kuhn,
                chain_monomers=N,
                dim=2,
            ), 
            dim=2
        )
        atoms.calc = calc

        # Relax to equilibrium
        ucf = UnitCellFilter(atoms, hydrostatic_strain=True)
        opt = FIRE(ucf)
        opt.run(fmax=1e-3)

        return atoms, molecules
    return factory


@pytest.fixture(scope='function')
def graphite_calc_largecutoff(graphite_calc_factory):
    rc_factor = 8.0
    atoms, molecules = graphite_calc_factory(rc_factor)
    return atoms, molecules, rc_factor

def test_equilibrium_radius(graphite_meanfield, graphite_calc_largecutoff):
    ana = graphite_meanfield
    req = ana.compute_equilibrium_distance()

    atoms, molecules, rc_factor = graphite_calc_largecutoff
    
    bond_lengths = molecules.get_distances(atoms) 
    mean_bond_length = np.mean(bond_lengths)
    assert np.isclose(mean_bond_length, req, rtol=1e-2), f"Expected bond length {req}, got {mean_bond_length}"

def test_total_energy(graphite_meanfield, graphite_calc_largecutoff):
    ana = graphite_meanfield
    req = ana.compute_equilibrium_distance()

    atoms, molecules, rc_factor = graphite_calc_largecutoff

    cell = atoms.get_cell()
    for scale in [0.98, 1., 1.02]:
        e0 = ana.total_energy(req * scale)
        new_cell = cell * scale        

        atoms.set_cell(new_cell, scale_atoms=True)
        e_lammps = atoms.get_potential_energy() / len(atoms)

        assert np.isclose(e_lammps, e0, rtol=1e-2), f"Expected energy per crosslink {e0}, got {e_lammps}"


def test_shear_modulus(graphite_meanfield, graphite_calc_largecutoff):
    ana = graphite_meanfield
    req = ana.compute_equilibrium_distance()
    G0 = ana.shear_modulus(r=req)

    atoms, molecules, rc_factor = graphite_calc_largecutoff
    
    # Use cubic symmetry since graphite has cubic symmetry
    # Small strain amplitude and more steps for accuracy
    C, C_err = fit_elastic_constants(
        atoms,
        symmetry="triclinic",
        N_steps=5,
        delta=1e-4,
        # optimizer=FIRE,
        fmax=1e-6,
        verbose=True,
    )

    C11 = C[0,0]
    C12 = C[0,1]
    C44 = C[5,5]

    shear_modulus_voigt = C44
    shear_modulus_reuss = (C11 - C12) / 2

    # Test against mean-field prediction
    assert np.isclose(shear_modulus_voigt, G0, rtol=5e-2), f"Expected shear modulus {G0}, got {shear_modulus_voigt}"
    assert np.isclose(shear_modulus_reuss, G0, rtol=5e-2), f"Expected shear modulus {G0}, got {shear_modulus_reuss}"


def test_bulk_modulus(graphite_meanfield, graphite_calc_largecutoff):
    ana = graphite_meanfield
    req = ana.compute_equilibrium_distance()
    K0 = ana.bulk_modulus(r=req)

    atoms, molecules, rc_factor = graphite_calc_largecutoff
    
    # Use cubic symmetry since graphite has cubic symmetry
    # Small strain amplitude and more steps for accuracy
    C, C_err = fit_elastic_constants(
        atoms,
        symmetry="triclinic", #"hexagonal",
        N_steps=5,
        delta=1e-4,
        optimizer=FIRE,  # We wat
        fmax=1e-6,
        verbose=True,
    )

    # C11, C12, C44 = Voigt_6x6_to_cubic(C)
    C11 = C[0,0]
    C12 = C[0,1]
    C44 = C[3,3]
    bulk_modulus = (C11 + C12) / 2

    # Test against mean-field prediction
    assert np.isclose(bulk_modulus, K0, rtol=5e-2), f"Expected bulk modulus {K0}, got {bulk_modulus}"
