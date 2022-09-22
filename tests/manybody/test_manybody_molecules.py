#
# Copyright 2022 Lucas Frérot (U. Freiburg)
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

import numpy as np
import numpy.testing as nt

from ase import Atoms
from matscipy.calculators.manybody.calculator import NiceManybody
from matscipy.calculators.manybody.explicit_forms import (
    ZeroTriplet, ZeroPair,
    HarmonicAngle, HarmonicBond
)
from matscipy.molecules import Molecules
from matscipy.neighbours import MolecularNeighbourhood
from matscipy.numerical import numerical_forces, numerical_hessian

import pytest


@pytest.fixture(params=[1, 1.5, 2])
def length(request):
    return request.param


@pytest.fixture(params=[np.pi / 6, np.pi / 3, np.pi / 2])
def angle(request):
    return request.param


@pytest.fixture
def co2(length, angle):
    s = 1.1
    atoms = Atoms(
        "CO2",
        # Not symmetric on purpose
        positions=[
            [-1, 0, 0],
            [0, 0, 0],
            [s * np.cos(angle), s * np.sin(angle), 0],
        ],
        cell=[5, 5, 5],
    )

    atoms.positions[:] *= length
    return atoms


@pytest.fixture
def molecule():
    return MolecularNeighbourhood(
        Molecules(
            bonds_connectivity=[[0, 1], [1, 2]],
            angles_connectivity=[[0, 1, 2]]
        )
    )


def test_harmonic_bond(co2, molecule):
    k, r0 = 1, 0.5

    print(molecule)

    calc = NiceManybody(HarmonicBond(r0, k), ZeroTriplet(), molecule)
    co2.calc = calc

    pair_vectors = np.array([
        co2.get_distance(0, 1, vector=True),
        co2.get_distance(1, 2, vector=True),
    ])

    pair_distances = np.linalg.norm(pair_vectors, axis=-1)

    # Testing potential energy
    epot = co2.get_potential_energy()
    epot_ref = np.sum(0.5 * k * (pair_distances - r0)**2)
    nt.assert_allclose(epot, epot_ref, rtol=1e-15)

    # Testing force on first atom
    f = co2.get_forces()
    f_ref = k * (pair_distances[0] - r0) * pair_vectors[0] / pair_distances[0]
    nt.assert_allclose(f[0], f_ref, rtol=1e-15)

    # Testing all forces with finite differences
    f_ref = numerical_forces(co2, d=1e-6)
    nt.assert_allclose(f, f_ref, rtol=1e-9, atol=1e-7)

    # Testing hessian
    h = co2.calc.get_property('hessian').todense()
    h_ref = numerical_hessian(co2, dx=1e-6).todense()
    nt.assert_allclose(h, h_ref, atol=1e-4)


def test_harmonic_angle(co2, molecule):
    kt, theta0 = 1, np.pi / 4
    calc = NiceManybody(ZeroPair(), HarmonicAngle(theta0, kt, co2), molecule)
    co2.calc = calc

    angle = np.pi - np.radians(co2.get_angle(0, 1, 2))

    # Testing potential energy
    epot = co2.get_potential_energy()
    epot_ref = 0.5 * kt * (angle - theta0)**2
    nt.assert_allclose(epot, epot_ref, rtol=1e-14)

    # Testing forces
    f = co2.get_forces()
    f_ref = numerical_forces(co2, d=1e-6)
    nt.assert_allclose(f, f_ref, rtol=1e-6, atol=1e-9)

    # Checking zeros
    nt.assert_allclose(np.abs(f.sum()), 0, atol=1e-13)

    # Testing hessian
    h = co2.calc.get_property('hessian')
    h_ref = numerical_hessian(co2, dx=1e-4)
    # print(f"{h}\n\n{h_ref}")
    nt.assert_allclose(h.todense(), h_ref.todense(), atol=1e-5)
