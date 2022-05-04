#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2021 Jan Griesser (U. Freiburg)
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

import pytest

import numpy as np
import numpy.testing as nt

from ase import Atoms

from ase import io 

from ase.units import GPa

from matscipy.elasticity import full_3x3x3x3_to_Voigt_6x6

from matscipy.numerical import (
    numerical_forces,
    numerical_stress,
    numerical_hessian,
    numerical_nonaffine_forces,
)

from matscipy.calculators.manybody.newmb import Manybody

from matscipy.calculators.manybody.potentials import (
    ZeroPair,
    ZeroAngle,
    HarmonicPair,
    HarmonicAngle,
    StillingerWeberPair,
    StillingerWeberAngle,
    KumagaiPair,
    KumagaiAngle
)

from matscipy.elasticity import (
    measure_triclinic_elastic_constants,
    full_3x3x3x3_to_Voigt_6x6 as to_voigt,
    Voigt_6x6_to_full_3x3x3x3 as from_voigt,
)

from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood, CutoffNeighbourhood, Neighbourhood

from ase.lattice.cubic import Diamond, SimpleCubic

from ase.optimize import FIRE

# Remove this! Only for testing
Stillinger_Weber_PRB_31_5262_Si = {
    '__ref__':  'F. Stillinger and T. Weber, Phys. Rev. B 31, 5262 (1985)',
    'el':            'Si'            ,
    'epsilon':       2.1683          ,
    'sigma':         2.0951          ,
    'costheta0':     0.333333333333  ,
    'A':             7.049556277     ,
    'B':             0.6022245584    ,
    'p':             4               ,
    'q':             0               ,    
    'a':             1.80            ,
    'lambda1':       21.0            ,
    'gamma':         1.20            
}

Kumagai_Comp_Mat_Sci_39_Si = {
   '__ref__':  'T. Kumagai et. al., Comp. Mat. Sci. 39 (2007)',    
    'el':            'Si'         ,
    'A':             3281.5905    ,
    'B':             121.00047    ,
    'lambda_1':      3.2300135    ,
    'lambda_2':      1.3457970    ,
    'eta':           1.0000000    ,
    'delta':         0.53298909   ,
    'alpha':         2.3890327    ,
    'beta':          1.0000000    ,
    'c_1':           0.20173476   ,
    # 730418.72
    'c_2':           730418.72    ,
    'c_3':           1000000.0    ,
    'c_4':           1.0000000    ,
    'c_5':           26.000000    ,
    'h':             -0.36500000  ,
    'R_1':           2.70         ,
    'R_2':           3.30              
    }

def cauchy_correction(stress):
    delta = np.eye(3)

    stress_contribution = 0.5 * sum(
        np.einsum(einsum, stress, delta)
        for einsum in (
                'am,bn',
                'an,bm',
                'bm,an',
                'bn,am',
        )
    )

    # Why does removing this work for the born constants?
    # stress_contribution -= np.einsum('ab,mn', stress, delta)
    return stress_contribution


def molecule():
    """Return a molecule setup involing all 4 atoms."""
    # Get all combinations of eight atoms
    bonds = np.array(
        np.meshgrid([np.arange(4)] * 2),
    ).T.reshape(-1, 2)

    # Get all combinations of eight atoms
    angles = np.array(np.meshgrid([np.arange(4)] * 3)).T.reshape(-1, 3)

    # Delete degenerate pairs and angles
    bonds = bonds[bonds[:, 0] != bonds[:, 1]]
    angles = angles[
        (angles[:, 0] != angles[:, 1])
        | (angles[:, 0] != angles[:, 2])
        | (angles[:, 1] != angles[:, 2])
    ]

    return MolecularNeighbourhood(
        Molecules(bonds_connectivity=bonds, angles_connectivity=angles)
    )

# Potentials to be tested
potentials = {
    "KumagaiPair+KumagaiAngle": (
        {1: KumagaiPair(Kumagai_Comp_Mat_Sci_39_Si)}, {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)}
    ),

    "KumagaiPair+zeroAngle": (
        {1: KumagaiPair(Kumagai_Comp_Mat_Sci_39_Si)}, {1: ZeroAngle()}
    ),

    "zeroPair+KumagaiAngle": (
        {1: ZeroPair()}, {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)}
    )

}

@pytest.fixture(params=potentials.values(), ids=potentials.keys())
def potential(request):
    return request.param

@pytest.fixture(params=[5.429, 5.1])
def distance(request):
    return request.param

@pytest.fixture
def configuration(distance, potential):
    atoms = Diamond("Si", size=[1, 1, 1], latticeconstant=distance)
    atoms.calc = Manybody(*potential, CutoffNeighbourhood(cutoff=3.3))
    return atoms

@pytest.fixture
def neq_configuration(potential):
    atoms = Diamond("Si", size=[1, 1, 1], latticeconstant=5.3)
    atoms.rattle(1e-4)
    atoms.calc = Manybody(*potential, CutoffNeighbourhood(cutoff=3.3))
    return atoms


###############################################################################


def test_forces(configuration):
    #FIRE(configuration, logfile=None).run(fmax=1e-5)
    f_ana = configuration.get_forces()
    f_num = numerical_forces(configuration, d=1e-6)
    #print("Energy: ", configuration.calc.get_property("energy") / len(configuration))
    #print("fana: ", f_ana)
    #print("f_num: ", f_num)
    nt.assert_allclose(f_ana, f_num, atol=1e-6, rtol=1e-6)

def test_forces_neq(neq_configuration):
    #FIRE(configuration, logfile=None).run(fmax=1e-5)
    f_ana = neq_configuration.get_forces()
    f_num = numerical_forces(neq_configuration, d=1e-6)
    #print("Energy: ", neq_configuration.calc.get_property("energy") / len(neq_configuration))
    #print("fana: ", f_ana)
    #print("f_num: ", f_num)
    nt.assert_allclose(f_ana, f_num, atol=1e-6, rtol=1e-6)

def test_stresses(configuration):
    s_ana = configuration.get_stress()
    s_num = numerical_stress(configuration, d=1e-6)
    #print("s_ana: ", s_ana)
    #print("s_num: ", s_num)
    nt.assert_allclose(s_ana, s_num, atol=1e-6, rtol=1e-6)

def test_stresses_neq(neq_configuration):
    s_ana = neq_configuration.get_stress()
    s_num = numerical_stress(neq_configuration, d=1e-6)
    #print("s_ana: ", s_ana)
    #print("s_num: ", s_num)
    nt.assert_allclose(s_ana, s_num, atol=1e-6, rtol=1e-6)


def test_born_constants(configuration):
    C_ana = configuration.calc.get_property("born_constants")
    C_num = measure_triclinic_elastic_constants(configuration, d=1e-6)

    # Compute Cauchy stress correction
    stress = configuration.get_stress(voigt=False)
    corr = cauchy_correction(stress)
    
    #print("C_ana: ", full_3x3x3x3_to_Voigt_6x6(C_ana) / GPa)
    #print("C_num: ", full_3x3x3x3_to_Voigt_6x6(C_num+corr) / GPa)

    nt.assert_allclose(C_ana + corr, C_num, atol=1e-5, rtol=1e-5)

"""
def test_nonaffine_forces(configuration):
    naf_ana = configuration.calc.get_property('nonaffine_forces')
    naf_num = numerical_nonaffine_forces(configuration, d=1e-9)

    m = naf_ana.nonzero()
    print(naf_ana[m])
    print(naf_num[m])
    nt.assert_allclose(naf_ana, naf_num, rtol=1e-6)


@pytest.mark.xfail(reason="Not implemented")
def test_hessian(configuration):
    H_ana = configuration.calc.get_property('hessian')
    H_num = numerical_hessian(configuration, dx=1e-6)

    nt.assert_allclose(H_ana.todense(), H_num.todense(), rtol=1e-6)
"""