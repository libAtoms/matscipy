

from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import MeanFieldHydrogelLattice, CROSSLINK_VOLUMES
import numpy as np

def test_radius_from_density_roundtrip_3D():
    density = 0.1
    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=4,
        vpcl_factor=CROSSLINK_VOLUMES['diamond'], 
        flory_chi=0. ,
    )
    r = polymer.radius_from_density(density, )
    density_computed = 1 / polymer.vpcl(r)
    assert np.isclose(density, density_computed), f"Expected density {density}, got {density_computed}"


def test_radius_from_density_roundtrip_2D():
    density = 0.1
    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=3,
        vpcl_factor=CROSSLINK_VOLUMES['graphite'], 
        flory_chi=0. ,
        dim=2
    )
    r = polymer.radius_from_density(density, )
    density_computed = 1 / polymer.vpcl(r)
    assert np.isclose(density, density_computed), f"Expected density {density}, got {density_computed}"


def test_elastic_constants_consistency():
    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=4,
        vpcl_factor=CROSSLINK_VOLUMES['diamond'], 
        flory_chi=0. ,
    )
    r = polymer.compute_equilibrium_radius()
    C11 = polymer.C11(r=r)
    C12 = polymer.C12(r=r)
    C44 = polymer.C44(r=r)
    bulk_modulus = polymer.bulk_modulus(r=r)
    assert np.isclose(bulk_modulus, (C11 + 2 * C12) / 3), f"Bulk modulus inconsistency: {bulk_modulus} vs {(C11 + 2 * C12) / 3}"