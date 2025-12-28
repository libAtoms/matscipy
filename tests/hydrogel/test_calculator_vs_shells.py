

import pytest

from matscipy.calculators.hydrogel.reference_solutions.lattice_shell_structures import GraphiteShellStructure
from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import CROSSLINK_VOLUMES, MeanFieldHydrogelLattice



import numpy as np
import matplotlib.pyplot as plt
from ase.constraints import UnitCellFilter
from ase.optimize import FIRE

# Import the factory function directly
from matscipy.calculators.hydrogel.snippets.factory import create_graphite_lattice
import pytest

from matscipy.calculators.hydrogel.reference_solutions.mean_field_lattices import CROSSLINK_VOLUMES, MeanFieldHydrogelLattice

from matscipy.calculators.hydrogel.calculator import Hydrogel
from matscipy.calculators.hydrogel.potentials import FloryHuggins, GaussianChain, LucyWeightFunction2D

from matscipy.elasticity import Voigt_6x6_to_cubic, fit_elastic_constants

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

class TestGraphite():
    
    @pytest.fixture(scope="class")
    def meanfield(self, parameters):    
        return MeanFieldHydrogelLattice(parameters['chain_monomers'], 
            3, CROSSLINK_VOLUMES['graphite'], 
            flory_chi = parameters['flory_chi'], dim=2)

    @pytest.fixture(scope='class')
    def shellstructure(self):
        return GraphiteShellStructure(cutoff=10.0)

    @pytest.fixture(scope='function')
    def calc_factory(self, meanfield):
        ana = meanfield
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

    @pytest.fixture(scope='class')
    def shellcalc_factory(parameters, meanfield, shellstructure):
        N  = meanfield.chain_nb_monomers
        χ = meanfield.flory_chi
        kuhn = meanfield.kuhn

        req_mf = meanfield.compute_equilibrium_distance()
        Re = np.sqrt(N) * kuhn  # RMS end-to-end distance of a free chain

        def factory(rc_factor):
            return ShellHydrogelCalculator(shellstructure, 
                chain_potential=GaussianChain(1, meanfield.chain_nb_monomers, dim=2),
                embedding_potential=FloryHuggins(
                    meanfield.chain_nb_monomers,
                    monomer_volume=meanfield.v0,
                    flory_chi=meanfield.flory_chi,
                    coordination = meanfield.coordination,
                ),
                weight_function=LucyWeightFunction2D(cutoff=rc_factor * req_mf),)

        
        return factory
    
    @pytest.fixture(params=[{'rc_factor': rc_factor} for rc_factor in [2. , 4. , 5.]], scope='function')
    def calcs(self, request,calc_factory, shellcalc_factory):
        rc_factor = request.param['rc_factor']
        atoms, molecules = calc_factory(rc_factor)
        shellcalc = shellcalc_factory(rc_factor)
        return atoms, molecules, shellcalc, rc_factor
    
    def test_shear_modulus(self, calcs, meanfield):
        atoms, molecules, shellcalc, rc_factor = calcs

        # Use cubic symmetry since graphite has cubic symmetry
        # Small strain amplitude and more steps for accuracy
        C, C_err = fit_elastic_constants(
            atoms,
            symmetry="triclinic",
            N_steps=5,
            delta=1e-4,
            optimizer=None, # So it is affine deformations
            fmax=1e-6,
            verbose=True,
        )

        C11 = C[0,0]
        C12 = C[0,1]
        C44 = C[5,5]

        shear_modulus_voigt = C44
        shear_modulus_reuss = (C11 - C12) / 2

        req = shellcalc.compute_equilibrium_distance(meanfield.compute_equilibrium_distance())
        
        shell_C = shellcalc.stiffness_matrix(req)
        shell_G = shell_C[0,1,0,1]
        # Test against mean-field prediction
        assert np.isclose(shear_modulus_voigt, shell_G, rtol=5e-2), f"Expected shear modulus {shell_G}, got {shear_modulus_voigt}"
        assert np.isclose(shear_modulus_reuss, shell_G, rtol=5e-2), f"Expected shear modulus {shell_G}, got {shear_modulus_reuss}"

    def test_bulk_modulus(self, calcs, meanfield):
            atoms, molecules, shellcalc, rc_factor = calcs

            # Use cubic symmetry since graphite has cubic symmetry
            # Small strain amplitude and more steps for accuracy
            C, C_err = fit_elastic_constants(
                atoms,
                symmetry="triclinic",
                N_steps=5,
                delta=1e-4,
                optimizer=None, # So we are computing the affine elastic constants
                fmax=1e-6,
                verbose=True,
            )

            C11 = C[0,0]
            C12 = C[0,1]

            bulk_modulus = (C11 + C12) / 2

            req = shellcalc.compute_equilibrium_distance(meanfield.compute_equilibrium_distance())
            
            shell_C = shellcalc.stiffness_matrix(req)
            shell_K = (shell_C[0,0,0,0] + shell_C[0,0,1,1]) / 2
            # Test against mean-field prediction
            assert np.isclose(bulk_modulus, shell_K, rtol=5e-2), f"Expected bulk modulus {shell_K}, got {bulk_modulus}"
