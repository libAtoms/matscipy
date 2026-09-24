"""
Tests that the meanfield results are recovered for large cutoff radii.
"""


import pytest
import numpy as np

from ase.filters import UnitCellFilter
from ase.optimize import FIRE
from ase.build import bulk

from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import MeanFieldHydrogelLattice, CROSSLINK_VOLUMES
from matscipy.calculators.hydrogel.calculator import Hydrogel
from matscipy.calculators.hydrogel.network import GaussianChain
from matscipy.molecules import Molecules
from matscipy.neighbours import neighbour_list

from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants

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
])
def parameters(request):
    return request.param

@pytest.fixture()
def diamond_meanfield(parameters, ):
    return MeanFieldHydrogelLattice(parameters['chain_monomers'], 4, CROSSLINK_VOLUMES['diamond'], flory_chi = parameters['flory_chi'], )

def _create_diamond_lattice(crosslink_spacing, n_cells=2):
    # Diamond lattice constant (nearest neighbor distance = a*sqrt(3)/4)
    # We want nearest neighbor distance = R0
    a = crosslink_spacing * 4 / np.sqrt(3)

    # Create diamond lattice
    atoms = bulk("C", "diamond", a=a, cubic=True)
    atoms = atoms.repeat((n_cells, n_cells, n_cells))

    # Set masses (LJ units)
    atoms.set_masses(np.ones(len(atoms)))

    # Create bonds between nearest neighbors
    # Nearest neighbor distance in diamond is a*sqrt(3)/4 = R0
    nn_dist = a * np.sqrt(3) / 4
    i_p, j_p = neighbour_list("ij", atoms, nn_dist * 1.1)

    # Keep only unique bonds (i < j)
    mask = i_p < j_p
    bonds = np.column_stack([i_p[mask], j_p[mask]])

    molecules = Molecules(bonds_connectivity=bonds)

    return atoms, molecules

@pytest.fixture()
def diamond_calc_factory(diamond_meanfield):
    ana = diamond_meanfield
    N  = ana.chain_nb_monomers
    χ = ana.flory_chi
    kuhn = ana.kuhn

    req_mf = ana.compute_equilibrium_distance()
    Re = np.sqrt(N) * kuhn  # RMS end-to-end distance of a free chain

    def factory(rc_factor):
        rc = rc_factor * req_mf
    
        atoms, molecules = _create_diamond_lattice(crosslink_spacing=req_mf, n_cells=np.ceil(1.5 * rc / req_mf).astype(int))
        calc = Hydrogel(
            cutoff=rc,
            chain_monomers=N,
            kuhn_length=kuhn,
            monomer_volume=ana.v0,
            flory_chi=χ ,
            molecules=molecules,
            chain=GaussianChain(
                kuhn_length=kuhn,
                chain_monomers=N,
            ),
        )
        atoms.calc = calc

        # Relax to equilibrium
        ucf = UnitCellFilter(atoms, hydrostatic_strain=True)
        opt = FIRE(ucf)
        opt.run(fmax=1e-3)

        return atoms, molecules
    return factory

@pytest.fixture()
def diamond_calc_largecutoff(diamond_calc_factory):
    rc_factor = 4.0
    atoms, molecules = diamond_calc_factory(rc_factor)
    return atoms, molecules, rc_factor

def test_equilibrium_radius(diamond_meanfield, diamond_calc_largecutoff):
    ana = diamond_meanfield
    req = ana.compute_equilibrium_distance()

    atoms, molecules, rc_factor = diamond_calc_largecutoff
    
    bond_lengths = molecules.get_distances(atoms) 
    mean_bond_length = np.mean(bond_lengths)
    assert np.isclose(mean_bond_length, req, rtol=1e-2), f"Expected bond length {req}, got {mean_bond_length}"

def test_total_energy(diamond_meanfield, diamond_calc_largecutoff):
    ana = diamond_meanfield
    req = ana.compute_equilibrium_distance()

    atoms, molecules, rc_factor = diamond_calc_largecutoff

    cell = atoms.get_cell()
    for scale in [0.98, 1., 1.02]:
        e0 = ana.total_energy(req * scale)
        new_cell = cell * scale        

        atoms.set_cell(new_cell, scale_atoms=True)
        e_lammps = atoms.get_potential_energy() / len(atoms)

        assert np.isclose(e_lammps, e0, rtol=1e-2), f"Expected energy per crosslink {e0}, got {e_lammps}"


def test_shear_modulus(diamond_meanfield, diamond_calc_largecutoff):
    ana = diamond_meanfield
    req = ana.compute_equilibrium_distance()
    G0 = ana.shear_modulus(r=req)

    atoms, molecules, rc_factor = diamond_calc_largecutoff
    
    # Use cubic symmetry since diamond has cubic symmetry
    # Small strain amplitude and more steps for accuracy
    C, C_err = fit_elastic_constants(
        atoms,
        symmetry="cubic",
        N_steps=5,
        delta=1e-4,
        optimizer=FIRE,
        fmax=1e-6,
        verbose=True,
    )

    C11, C12, C44 = Voigt_6x6_to_cubic(C)

    shear_modulus_voigt = C44
    shear_modulus_reuss = (C11 - C12) / 2

    # Test against mean-field prediction
    assert np.isclose(shear_modulus_voigt, G0, rtol=5e-2), f"Expected shear modulus {G0}, got {shear_modulus_voigt}"
    assert np.isclose(shear_modulus_reuss, G0, rtol=5e-2), f"Expected shear modulus {G0}, got {shear_modulus_reuss}"


def test_bulk_modulus(diamond_meanfield, diamond_calc_largecutoff):
    ana = diamond_meanfield
    req = ana.compute_equilibrium_distance()
    K0 = ana.bulk_modulus(r=req)

    atoms, molecules, rc_factor = diamond_calc_largecutoff
    
    # Use cubic symmetry since diamond has cubic symmetry
    # Small strain amplitude and more steps for accuracy
    C, C_err = fit_elastic_constants(
        atoms,
        symmetry="cubic",
        N_steps=5,
        delta=1e-4,
        optimizer=FIRE,
        fmax=1e-6,
        verbose=True,
    )

    C11, C12, C44 = Voigt_6x6_to_cubic(C)

    bulk_modulus = (C11 + 2 * C12) / 3

    # Test against mean-field prediction
    assert np.isclose(bulk_modulus, K0, rtol=5e-2), f"Expected bulk modulus {K0}, got {bulk_modulus}"




# def test_diamond_meanfield_energy_consistency(diamond_meanfield, diamond_calc_factory):

#     req_lammps = []
#     energy_lammps = []

#     strains = np.array([0, eps, -eps])
#     # eam_cutoffs =  np.concatenate([np.linspace(1., 3.5, 20, endpoint=False), np.linspace(3.5, 15, 40)[:10]])* req
#     eam_cutoffs =  np.concatenate([np.linspace(1.5, 4, 4, )]) * req

#     req_lammps = np.array(req_lammps)
#     energy_lammps = np.array(energy_lammps)
#     assert np.isclose(req_lammps[-1], req, rtol=1e-2)
    
#     req_error = abs( req_lammps / req - 1 )
#     assert req_error[-1] < 0.5 *req_error[0]
    
#     e = energy_lammps[-1]        
#     ref_energy = np.array([ana.total_energy_mean_field(req * (1+s) ) for s in strains])
#     energy_errors = np.mean( abs(energy_lammps - ref_energy.reshape(1,-1)), axis=-1)
    
    
#     assert energy_errors[-1] < 0.5 * energy_errors[0]
#     assert energy_errors[-1] < 0.01 * abs(np.mean(ref_energy))
