"""
Tests that the meanfield results are recovered for large cutoff radii.
"""


import pytest
import numpy as np

from ase.filters import UnitCellFilter
from ase.optimize import FIRE
from ase.build import bulk

from matscipy.molecules import Molecules
from matscipy.neighbours import neighbour_list
from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants



from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import MeanFieldHydrogelLattice, CROSSLINK_VOLUMES
from matscipy.calculators.hydrogel.calculator import Hydrogel
from matscipy.calculators.hydrogel.embedding_constant_coordination import FloryHugginsPotential
from matscipy.calculators.hydrogel import GaussianChain, LucyWeightFunction2D
from matscipy.calculators.hydrogel.reference_solutions.lattice_shell_structures import GraphiteShellStructure
from matscipy.calculators.hydrogel.reference_solutions.shell_calculator import ShellHydrogelCalculator


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

@pytest.fixture(scope='module')
def diamond_meanfield(parameters, ):
    return MeanFieldHydrogelLattice(parameters['chain_monomers'], 4, CROSSLINK_VOLUMES['diamond'], flory_chi = parameters['flory_chi'], )


@pytest.fixture(scope='module')
def graphite_meanfield(parameters, ):
    return MeanFieldHydrogelLattice(parameters['chain_monomers'], 
        3, CROSSLINK_VOLUMES['graphite'], 
        flory_chi = parameters['flory_chi'], dim=2)

@pytest.fixture(scope='module')
def graphite_shellstructure(parameters, ):
    return GraphiteShellStructure(cutoff=6.0)

@pytest.fixture(scope='module')
def graphite_shellcalc_largecutoff(parameters, graphite_meanfield, graphite_shellstructure):
    N  = graphite_meanfield.chain_nb_monomers
    χ = graphite_meanfield.flory_chi
    kuhn = graphite_meanfield.kuhn

    req_mf = graphite_meanfield.compute_equilibrium_distance()
    Re = np.sqrt(N) * kuhn  # RMS end-to-end distance of a free chain

    return ShellHydrogelCalculator(graphite_shellstructure, 
        chain_potential=GaussianChain(1, graphite_meanfield.chain_nb_monomers, dim=2),
        embedding_potential=FloryHugginsPotential (
            graphite_meanfield.chain_nb_monomers,
            monomer_volume=graphite_meanfield.v0,
            flory_chi=graphite_meanfield.flory_chi,
            coordination = graphite_meanfield.coordination,
        ),
        weight_function=LucyWeightFunction2D(cutoff=5. * req_mf),)


def test_density(graphite_meanfield, graphite_shellcalc_largecutoff):
    ana = graphite_meanfield
    calc = graphite_shellcalc_largecutoff

    r = ana.compute_equilibrium_distance() * 1.1

    rho_ana = 1 / ana.vpcl_factor * r ** (-2)
    rho_calc = calc.density(r)

    assert np.isclose(rho_calc, rho_ana, rtol=1e-2), f"Expected density {rho_ana}, got {rho_calc}"

def test_equilibrium_radius(graphite_meanfield, graphite_shellcalc_largecutoff):
    ana = graphite_meanfield
    calc = graphite_shellcalc_largecutoff
    req = ana.compute_equilibrium_distance()

    req_shells = calc.compute_equilibrium_distance(req)

    assert np.isclose(req_shells, req, rtol=1e-2), f"Expected bond length {req}, got {req_shells}"

def test_total_energy(graphite_meanfield, graphite_shellcalc_largecutoff):
    ana = graphite_meanfield
    req = ana.compute_equilibrium_distance()

    calc = graphite_shellcalc_largecutoff
    for scale in [0.98, 1., 1.02]:
        e0 = ana.total_energy(req * scale)
        r = req * scale 
        e_shells = calc.energy(r)

        assert np.isclose(e_shells, e0, rtol=1e-2), f"Expected energy per crosslink {e0}, got {e_shells}"

def test_elastic_energy(graphite_meanfield, graphite_shellcalc_largecutoff):
    ana = graphite_meanfield
    req = ana.compute_equilibrium_distance()

    calc = graphite_shellcalc_largecutoff
    for scale in [0.98, 1., 1.02]:
        e0 = ana.elastic_energy(req * scale)
        r = req * scale 
        e_shells = calc.bond_energy(r)

        assert np.isclose(e_shells, e0, rtol=1e-2), f"Expected energy per crosslink {e0}, got {e_shells}"

def test_mixing_energy(graphite_meanfield, graphite_shellcalc_largecutoff):
    ana = graphite_meanfield
    req = ana.compute_equilibrium_distance()

    calc = graphite_shellcalc_largecutoff
    for scale in [0.98, 1., 1.02]:
        e0 = ana.mixing_energy(req * scale)
        r = req * scale 
        e_shells = calc.embedding_energy(r)

        assert np.isclose(e_shells, e0, rtol=1e-2), f"Expected energy per crosslink {e0}, got {e_shells}"




# def test_shear_modulus(diamond_meanfield, diamond_calc_largecutoff):
#     ana = diamond_meanfield
#     req = ana.compute_equilibrium_distance()
#     G0 = ana.shear_modulus(r=req)

#     atoms, molecules, rc_factor = diamond_calc_largecutoff
    
#     # Use cubic symmetry since diamond has cubic symmetry
#     # Small strain amplitude and more steps for accuracy
#     C, C_err = fit_elastic_constants(
#         atoms,
#         symmetry="cubic",
#         N_steps=5,
#         delta=1e-4,
#         optimizer=FIRE,
#         fmax=1e-6,
#         verbose=True,
#     )

#     C11, C12, C44 = Voigt_6x6_to_cubic(C)

#     shear_modulus_voigt = C44
#     shear_modulus_reuss = (C11 - C12) / 2

#     # Test against mean-field prediction
#     assert np.isclose(shear_modulus_voigt, G0, rtol=5e-2), f"Expected shear modulus {G0}, got {shear_modulus_voigt}"
#     assert np.isclose(shear_modulus_reuss, G0, rtol=5e-2), f"Expected shear modulus {G0}, got {shear_modulus_reuss}"


# def test_bulk_modulus(diamond_meanfield, diamond_calc_largecutoff):
#     ana = diamond_meanfield
#     req = ana.compute_equilibrium_distance()
#     K0 = ana.bulk_modulus(r=req)

#     atoms, molecules, rc_factor = diamond_calc_largecutoff
    
#     # Use cubic symmetry since diamond has cubic symmetry
#     # Small strain amplitude and more steps for accuracy
#     C, C_err = fit_elastic_constants(
#         atoms,
#         symmetry="cubic",
#         N_steps=5,
#         delta=1e-4,
#         optimizer=FIRE,
#         fmax=1e-6,
#         verbose=True,
#     )

#     C11, C12, C44 = Voigt_6x6_to_cubic(C)

#     bulk_modulus = (C11 + 2 * C12) / 3

#     # Test against mean-field prediction
#     assert np.isclose(bulk_modulus, K0, rtol=5e-2), f"Expected bulk modulus {K0}, got {bulk_modulus}"


