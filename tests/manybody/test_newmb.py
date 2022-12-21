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
    SimplePairNoMix,
    SimplePairNoMixNoSecond,
    HarmonicPair,
    HarmonicAngle,
    KumagaiPair,
    KumagaiAngle,
    LennardJones,
    StillingerWeberPair,
    StillingerWeberAngle,
    TersoffBrennerPair,
    TersoffBrennerAngle,
)

from reference_params import (
    Kumagai_Comp_Mat_Sci_39_Si,
    Stillinger_Weber_PRB_31_5262_Si,
    Tersoff_PRB_39_5566_Si_C,
)

from matscipy.elasticity import (
    measure_triclinic_elastic_constants,
)

from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood, CutoffNeighbourhood


def tetrahedron(distance, rattle):
    atoms = Atoms(
        "H" * 4,
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        cell=[10, 10, 10],
    )
    atoms.positions *= distance
    atoms.rattle(rattle)
    return atoms


def diamond(distance, rattle):
    atoms = Diamond("Si", size=[1, 1, 1], latticeconstant=distance)
    atoms.rattle(rattle)
    return atoms


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


class MixPair(Manybody.Phi):
    """
    Implementation of a harmonic pair interaction.
    """

    def __call__(self, r_p, xi_p):
        return xi_p * r_p

    def gradient(self, r_p, xi_p):
        return np.stack([
            xi_p,
            r_p,
        ])

    def hessian(self, r_p, xi_p):
        return np.stack([
            np.zeros_like(r_p),
            np.zeros_like(xi_p),
            np.ones_like(r_p),
        ])


class LinearPair(Manybody.Phi):
    """
    Implementation of a harmonic pair interaction.
    """

    def __call__(self, r_p, xi_p):
        return r_p + xi_p

    def gradient(self, r_p, xi_p):
        return np.stack([
            np.ones_like(r_p),
            np.ones_like(xi_p),
        ])

    def hessian(self, r_p, xi_p):
        return np.stack([
            np.zeros_like(r_p),
            np.zeros_like(xi_p),
            np.zeros_like(xi_p),
        ])


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
        {1: ZeroPair()}, {1: SimpleAngle()}, CutoffNeighbourhood(cutoff=3.0),
    ),

    "SimpleAngle~molecule": (
        {1: HarmonicPair(1, 5)}, {1: SimpleAngle()}, molecule(),
    ),

    "KumagaiPair+ZeroAngle": (
        {1: KumagaiPair(Kumagai_Comp_Mat_Sci_39_Si)},
        {1: ZeroAngle()},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "LinearPair+HarmonicAngle": (
        {1: LinearPair()},
        {1: HarmonicAngle(1, np.pi/3)},
        CutoffNeighbourhood(cutoff=3.3),
    ),

    "LinearPair+KumagaiAngle": (
        {1: LinearPair()},
        {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "MixPair+KumagaiAngle": (
        {1: MixPair()},
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

    "SimplePairNoMixNoSecond+HarmonicAngle": (
        {1: SimplePairNoMixNoSecond()},
        {1: HarmonicAngle()},
        CutoffNeighbourhood(cutoff=3.3),

    ),

    "SimplePairNoMixNoSecond+KumagaiAngle": (
        {1: SimplePairNoMixNoSecond()},
        {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),

    "StillingerWeber": (
        {1: StillingerWeberPair(Stillinger_Weber_PRB_31_5262_Si)},
        {1: StillingerWeberAngle(Stillinger_Weber_PRB_31_5262_Si)},
        CutoffNeighbourhood(cutoff=Stillinger_Weber_PRB_31_5262_Si["a"]
                            * Stillinger_Weber_PRB_31_5262_Si["sigma"]),
    ),

    "Tersoff3": (
        {1: TersoffBrennerPair(Tersoff_PRB_39_5566_Si_C)},
        {1: TersoffBrennerAngle(Tersoff_PRB_39_5566_Si_C)},
        CutoffNeighbourhood(cutoff=Tersoff_PRB_39_5566_Si_C["R2"]),
    ),

    "KumagaiPair+KumagaiAngle": (
        {1: KumagaiPair(Kumagai_Comp_Mat_Sci_39_Si)},
        {1: KumagaiAngle(Kumagai_Comp_Mat_Sci_39_Si)},
        CutoffNeighbourhood(cutoff=Kumagai_Comp_Mat_Sci_39_Si["R_2"]),
    ),
}

@pytest.fixture(params=potentials.values(), ids=potentials.keys())
def potential(request):
    return request.param


# @pytest.fixture(params=[5.3, 5.431])
@pytest.fixture(params=[5.431])
def distance(request):
    return request.param


@pytest.fixture(params=[0, 1e-2])
def rattle(request):
    return request.param


@pytest.fixture(params=[diamond])
def configuration(distance, rattle, potential, request):
    atoms = request.param(distance, rattle)
    atoms.calc = Manybody(*potential)
    atoms.calc.atoms = atoms
    return atoms


###############################################################################


def test_forces(configuration):
    f_ana = configuration.get_forces()
    f_num = numerical_forces(configuration, d=1e-5)
    nt.assert_allclose(f_ana, f_num, rtol=1e-6, atol=1e-7)


def test_stresses(configuration):
    s_ana = configuration.get_stress()
    s_num = numerical_stress(configuration, d=1e-6)
    nt.assert_allclose(s_ana, s_num, rtol=1e-6, atol=1e-8)


def test_nonaffine_forces(configuration):
    # TODO: clarify why we need to optimize?
    FIRE(configuration, logfile=None).run(fmax=1e-8, steps=400)
    naf_ana = configuration.calc.get_property('nonaffine_forces')
    naf_num = numerical_nonaffine_forces(configuration, d=1e-8)

    # atol here related to fmax above
    nt.assert_allclose(naf_ana, naf_num, rtol=1e-6, atol=1e-4)


def test_hessian(configuration):
    FIRE(configuration, logfile=None).run(fmax=1e-8, steps=400)
    H_ana = configuration.calc.get_property('hessian').todense()
    H_num = numerical_hessian(configuration, dx=1e-6).todense()

    nt.assert_allclose(H_ana, H_num, atol=1e-5, rtol=1e-6)


def test_dynamical_matrix(configuration):
    # Maybe restrict this test to a single potential to reduce testing ?
    D_ana = configuration.calc.get_property('dynamical_matrix').todense()
    H_ana = configuration.calc.get_property('hessian').todense()
    mass = np.repeat(configuration.get_masses(), 3)
    H_ana /= np.sqrt(mass.reshape(-1, 1) * mass.reshape(1, -1))

    nt.assert_allclose(D_ana, H_ana, atol=1e-10, rtol=1e-10)


def test_birch_constants(configuration):
    B_ana = configuration.calc.get_property("birch_coefficients", configuration)
    C_num = measure_triclinic_elastic_constants(configuration, delta=1e-4)

    nt.assert_allclose(B_ana, C_num, rtol=1e-4, atol=1e-4)


def test_elastic_constants(configuration):
    # Needed since zero-temperature elastic constants defined in local minimum
    FIRE(configuration, logfile=None).run(fmax=1e-6, steps=400)
    C_ana = configuration.calc.get_property("elastic_constants", configuration)
    C_num = measure_triclinic_elastic_constants(
        configuration,
        delta=1e-3,
        optimizer=FIRE,
        fmax=1e-6,
        steps=500,
    )

    nt.assert_allclose(np.where(C_ana < 1e-6, 0.0, C_ana),
                       np.where(C_num < 1e-6, 0.0, C_num),
                       rtol=1e-3, atol=1e-3)


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

    def harmonic(t):
        return 0.5 * (t)**2

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
