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

from matscipy.numerical import (
    numerical_forces,
    numerical_stress,
    numerical_hessian,
    numerical_nonaffine_forces,
)

from matscipy.calculators.manybody.newmb import Manybody
from matscipy.calculators.pair_potential import PairPotential, LennardJonesCut
from ase.lattice.cubic import Diamond
from ase.optimize import FIRE

from matscipy.calculators.manybody.potentials import (
    ZeroPair,
    ZeroAngle,
    SimplePair,
    SimplePairNoMix,
    HarmonicPair,
    HarmonicAngle,
    KumagaiPair,
    KumagaiAngle,
    LennardJones,
    StillingerWeberPair,
    StillingerWeberAngle
)

from reference_params import (
    Kumagai_Comp_Mat_Sci_39_Si,
    Stillinger_Weber_PRB_31_5262_Si,
)

from matscipy.elasticity import (
    measure_triclinic_elastic_constants,
)

from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood, CutoffNeighbourhood

class SimpleAngle(Manybody.Theta):
    """Implementation of a zero three-body interaction."""

    def __call__(self, R1, R2, R3):
        return 0.5 * (R1**2 + R2**2 + R3**2)

    def gradient(self, R1, R2, R3):
        return np.stack([
            R1,
            R2,
            R3,
        ])

    def hessian(self, R1, R2, R3):
        return np.stack([
            np.ones(list(R1.shape)),
            np.ones(list(R1.shape)),
            np.ones(list(R1.shape)),
            np.zeros(list(R1.shape)),
            np.zeros(list(R1.shape)),
            np.zeros(list(R1.shape)),
        ])


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
    "Zero(Pair+Angle)": (
        {1: ZeroPair()}, {1: ZeroAngle()}, molecule()
    ),

    "Harmonic(Pair+Angle)": (
        {1: HarmonicPair(1, 1)}, {1: HarmonicAngle(1, np.pi / 4)}, molecule()
    ),

    "HarmonicPair+ZeroAngle": (
        {1: HarmonicPair(1, 1)}, {1: ZeroAngle()}, molecule()
    ),

    "ZeroPair+HarmonicAngle": (
        {1: ZeroPair()}, {1: HarmonicAngle(1, np.pi / 4)}, molecule()
    ),

    "SimpleAngle~cutoff": (
        {1: ZeroPair()}, {1: SimpleAngle()}, CutoffNeighbourhood(cutoff=5.),
    ),

    "SimpleAngle~molecule": (
        {1: HarmonicPair(1, 5)}, {1: SimpleAngle()}, molecule(),
    ),

    "Kumagai": (
        {1: KumagaiPair(Kumagai_Comp_Mat_Sci_39_Si)},
        {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "KumagaiPair+ZeroAngle": (
        {1: KumagaiPair(Kumagai_Comp_Mat_Sci_39_Si)},
        {1: ZeroAngle()},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "SimplePair+KumagaiAngle": (
        {1: SimplePair()},
        {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "ZeroPair+KumagaiAngle": (
        {1: ZeroPair()},
        {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "SimplePairNoMix+KumagaiAngle": (
        {1: SimplePairNoMix()},
        {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "StillingerWeber": (
        {1: StillingerWeberPair(Stillinger_Weber_PRB_31_5262_Si)},
        {1: StillingerWeberAngle(Stillinger_Weber_PRB_31_5262_Si)},
        CutoffNeighbourhood(cutoff=Stillinger_Weber_PRB_31_5262_Si["a"] * Stillinger_Weber_PRB_31_5262_Si["sigma"]),
    ),
}


@pytest.fixture(params=potentials.values(), ids=potentials.keys())
def potential(request):
    return request.param


@pytest.fixture(params=[5.2, 5.3, 5.431, 5.5])
def distance(request):
    return request.param


@pytest.fixture(params=[0, 1e-3, 1e-2, 1e-1])
#@pytest.fixture(params=[0])
def rattle(request):
    return request.param


def tetrahedron(distance, rattle):
    atoms = Atoms(
        "H" * 4,
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        cell=[10, 10, 10],
    )

    atoms.positions[:] *= distance
    return atoms


def diamond(distance, rattle):
    atoms = Diamond("Si", size=[1, 1, 1], latticeconstant=distance)
    atoms.rattle(rattle)
    return atoms


@pytest.fixture(params=[diamond, tetrahedron])
def configuration(distance, rattle, potential, request):
    atoms = request.param(distance, rattle)
    atoms.calc = Manybody(*potential)
    return atoms


###############################################################################


def test_forces(configuration):
    f_ana = configuration.get_forces()
    f_num = numerical_forces(configuration, d=1e-6)
    nt.assert_allclose(f_ana, f_num, rtol=1e-6, atol=1e-8)


def test_stresses(configuration):
    s_ana = configuration.get_stress()
    s_num = numerical_stress(configuration, d=1e-6)
    nt.assert_allclose(s_ana, s_num, rtol=1e-6, atol=1e-8)

def test_born_constants(configuration):
    C_ana = configuration.calc.get_property("born_constants")
    C_num = measure_triclinic_elastic_constants(configuration, d=1e-6)

    # Compute Cauchy stress correction
    stress = configuration.get_stress(voigt=False)
    corr = cauchy_correction(stress)

    nt.assert_allclose(C_ana + corr, C_num, rtol=1e-4, atol=1e-4)


def test_nonaffine_forces(configuration):
    # TODO: clarify why we need to optimize?
    FIRE(configuration).run(fmax=1e-9)

    naf_ana = configuration.calc.get_property('nonaffine_forces')
    naf_num = numerical_nonaffine_forces(configuration, d=1e-9)

    # atol here related to fmax above
    nt.assert_allclose(naf_ana, naf_num, rtol=1e-6, atol=5e-5)


@pytest.mark.xfail(reason="Not implemented")
def test_hessian(configuration):
    H_ana = configuration.calc.get_property('hessian').todense()
    H_num = numerical_hessian(configuration, dx=1e-6).todense()

    print("H_ana: \n", H_ana[:6,:6])
    print("H_num: \n", H_num[:6,:6])

    nt.assert_allclose(H_ana, H_num, rtol=1e-6)


@pytest.mark.parametrize('cutoff', np.linspace(1.1, 20, 10))
def test_pair_compare(cutoff):
    atoms = Atoms(
        "H" * 4,
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        cell=[10, 10, 10],
    )

    atoms.positions[:] *= 1
    atoms.calc = Manybody(
        {1: LennardJones(1, 1, cutoff)},
        {1: ZeroAngle()},
        CutoffNeighbourhood(cutoff=cutoff)
    )
    newmb_e = atoms.get_potential_energy()

    pair = PairPotential({(1, 1): LennardJonesCut(1, 1, cutoff)})
    pair_e = pair.get_property('energy', atoms)

    assert np.abs(newmb_e - pair_e) / pair_e < 1e-10


@pytest.mark.parametrize('cutoff', [1.4, 1.5])
def test_energy_cutoff(cutoff):
    atoms = Atoms(
        "H" * 4,
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        cell=[10, 10, 10],
    )

    atoms.calc = Manybody(
        {1: HarmonicPair(1, 1)},
        {1: HarmonicAngle(1, 0)},
        CutoffNeighbourhood(cutoff=cutoff)
    )
    newmb_e = atoms.get_potential_energy()
    harmonic = lambda t: 0.5 * (t)**2

    # 90 angles with next-neighbor cutoff
    # next-neighbor pairs have 0 energy
    e = 3 * harmonic(np.pi / 2)

    # cutoff large enough for longer distance interactions
    # adds all 45 and 60 angles
    # adds longer pairs
    if cutoff > np.sqrt(2):
        e += (
            + 6 * harmonic(np.pi / 4)
            + 3 * harmonic(np.pi / 3)
            + 3 * harmonic(np.sqrt(2) - 1)
        )

    assert np.abs(e - newmb_e) / e < 1e-10
