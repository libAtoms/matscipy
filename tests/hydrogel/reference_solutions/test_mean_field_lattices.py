

from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import MeanFieldHydrogelLattice, CROSSLINK_VOLUMES
import numpy as np

from matscipy.calculators.hydrogel.potentials import FloryHuggins

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
    r = polymer.compute_equilibrium_distance()
    C11 = polymer.C11(r=r)
    C12 = polymer.C12(r=r)
    C44 = polymer.C44(r=r)
    bulk_modulus = polymer.bulk_modulus(r=r)
    assert np.isclose(bulk_modulus, (C11 + 2 * C12) / 3), f"Bulk modulus inconsistency: {bulk_modulus} vs {(C11 + 2 * C12) / 3}"


def test_shear_modulus_vs_network():
    # Compute the shear modulus directly from the network pressure, and use this as a reference for the meanfield model. 
    # (where it is indirectly equal to the network pressure through the equilibrium condition)
    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=4,
        vpcl_factor=CROSSLINK_VOLUMES['diamond'], 
        flory_chi=0. ,
    )
    req = polymer.compute_equilibrium_distance()
    kT = 1 
    kchain = 3 * kT / ((polymer.chain_nb_monomers - 1) * polymer.kuhn**2)
    pressure = 3**(1/2) / 4 * kchain /req  

    np.testing.assert_allclose(polymer.shear_modulus(r=req), pressure, rtol=1e-5)

def test_network_pressure_consistency():
    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=4,
        vpcl_factor=CROSSLINK_VOLUMES['diamond'], 
        flory_chi=0. ,
    )
    req = polymer.compute_equilibrium_distance()


    pressure = polymer.elastic_pressure(r=req)

    h= 1e-6
    fd = - 1 / 3 * (polymer.elastic_energy(r=req + h) - polymer.elastic_energy(r=req - h)) / (2 * h * polymer.vpcl(req))  * req

    np.testing.assert_allclose(pressure, fd, rtol=1e-5)

    kT = 1 
    kchain = 3 * kT / ((polymer.chain_nb_monomers - 1) * polymer.kuhn**2)
    pressure_ana = - 3**(1/2) / 4 * kchain /req  

    np.testing.assert_allclose(pressure, pressure_ana, rtol=1e-5)


    np.testing.assert_allclose(- pressure, polymer.shear_modulus(r=req), rtol=1e-5)

def test_mixing_pressure_flory_huggins():
    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=4,
        vpcl_factor=CROSSLINK_VOLUMES['diamond'], 
        flory_chi=0.5 ,
    )
    r = polymer.compute_equilibrium_distance()

    fh = FloryHuggins(chain_monomers=100, monomer_volume=polymer.v0, 
                      flory_chi=0.5, coordination=4)
    
    pfh = fh.pressure(crosslink_density=1 / polymer.vpcl(r))

    pmf = polymer.shear_modulus(r=r)
    np.testing.assert_allclose(pmf, pfh, rtol=1e-5)

def test_shear_modulus_vs_fd_pressure():
    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=4,
        vpcl_factor=CROSSLINK_VOLUMES['diamond'], 
        flory_chi=0. ,
    )
    req = polymer.compute_equilibrium_distance()

    h= 1e-6
    fd = - 1 / 3 * (polymer.mixing_energy(r=req + h) - polymer.mixing_energy(r=req - h)) / (2 * h * polymer.vpcl(req))  * req

    shear_modulus = polymer.shear_modulus(r=req)

    np.testing.assert_allclose(shear_modulus, fd, rtol=1e-5)



def test_mixing_vs_conf_vs_fd_pressure():
    """
    At equilibrium the two pressures should cancel out.
    """

    polymer = MeanFieldHydrogelLattice(
        chain_nb_monomers=100, 
        coordination=4,
        vpcl_factor=CROSSLINK_VOLUMES['diamond'], 
        flory_chi=0. ,
    )
    req = polymer.compute_equilibrium_distance()

    h= 1e-6
    mix = - 1 / 3 * (polymer.mixing_energy(r=req + h) - polymer.mixing_energy(r=req - h)) / (2 * h * polymer.vpcl(req))
    conf = - 1 / 3 * (polymer.elastic_energy(r=req + h) - polymer.elastic_energy(r=req - h)) / (2 * h * polymer.vpcl(req))
    
    total = mix + conf

    assert np.isclose(total, 0, atol=1e-8), f"Expected total pressure 0, got {total}"