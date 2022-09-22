#
# Copyright 2014-2015, 2020-2021 Lars Pastewka (U. Freiburg)
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

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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
# ======================================================================

import math
import pytest

import numpy as np

import ase.io
from ase.constraints import FixAtoms
from ase.optimize import FIRE

import matscipy.fracture_mechanics.clusters as clusters
from matscipy.atomic_strain import atomic_strain
from matscipy.elasticity import Voigt_6_to_full_3x3_stress
from matscipy.fracture_mechanics.crack import CubicCrystalCrack
from matscipy.fracture_mechanics.energy_release import J_integral
from matscipy.neighbours import neighbour_list

try:
    import atomistica
except ImportError:
    atomistica = None


@pytest.mark.skipif(atomistica is None, reason="Atomistica not available")
def test_J_integral():
    """
    Check if the J-integral returns G=2*gamma
    """

    nx = 128
    for calc, a0, C11, C12, C44, surface_energy, bulk_coordination in [
        (
            atomistica.DoubleHarmonic(
                k1=1.0, r1=1.0, k2=1.0, r2=math.sqrt(2), cutoff=1.6
            ),
            clusters.sc("He", 1.0, [nx, nx, 1], [1, 0, 0], [0, 1, 0]),
            3,
            1,
            1,
            0.05,
            6,
        ),
        #            ( atomistica.Harmonic(k=1.0, r0=1.0, cutoff=1.3, shift=False),
        #              clusters.fcc('He', math.sqrt(2.0), [nx/2,nx/2,1], [1,0,0],
        #                           [0,1,0]),
        #              math.sqrt(2), 1.0/math.sqrt(2), 1.0/math.sqrt(2), 0.05, 12)
    ]:
        print("{} atoms.".format(len(a0)))

        crack = CubicCrystalCrack(
            [1, 0, 0], [0, 1, 0], C11=C11, C12=C12, C44=C44
        )

        x, y, z = a0.positions.T
        r2 = min(np.max(x) - np.min(x), np.max(y) - np.min(y)) / 4
        r1 = r2 / 2

        a = a0.copy()
        a.center(vacuum=20.0, axis=0)
        a.center(vacuum=20.0, axis=1)
        ref = a.copy()
        r0 = ref.positions

        a.calc = calc
        print("epot = {}".format(a.get_potential_energy()))

        sx, sy, sz = a.cell.diagonal()
        tip_x = sx / 2
        tip_y = sy / 2

        k1g = crack.k1g(surface_energy)

        # g = a.get_array('groups')  # groups are not defined
        g = np.ones(len(a))  # not fixing any atom

        old_x = tip_x + 1.0
        old_y = tip_y + 1.0
        while abs(tip_x - old_x) > 1e-6 and abs(tip_y - old_y) > 1e-6:
            a.set_constraint(None)

            ux, uy = crack.displacements(r0[:, 0], r0[:, 1], tip_x, tip_y, k1g)
            a.positions[:, 0] = r0[:, 0] + ux
            a.positions[:, 1] = r0[:, 1] + uy
            a.positions[:, 2] = r0[:, 2]

            a.set_constraint(ase.constraints.FixAtoms(mask=g == 0))
            opt = FIRE(a, logfile=None)
            opt.run(fmax=1e-3)

            old_x = tip_x
            old_y = tip_y
            tip_x, tip_y = crack.crack_tip_position(
                a.positions[:, 0],
                a.positions[:, 1],
                r0[:, 0],
                r0[:, 1],
                tip_x,
                tip_y,
                k1g,
                mask=g != 0,
            )

            print(tip_x, tip_y)

        # Get atomic strain
        i, j = neighbour_list("ij", a, cutoff=1.3)
        deformation_gradient, residual = atomic_strain(
            a, ref, neighbours=(i, j)
        )

        # Get atomic stresses
        # Note: get_stresses returns the virial in Atomistica!
        virial = a.get_stresses()
        virial = Voigt_6_to_full_3x3_stress(virial)

        # Compute J-integral
        epot = a.get_potential_energies()
        eref = np.zeros_like(epot)

        for r1, r2 in [(r1, r2), (r1 / 2, r2 / 2), (r1 / 2, r2)]:
            print("r1 = {}, r2 = {}".format(r1, r2))

            J = J_integral(
                a,
                deformation_gradient,
                virial,
                epot,
                eref,
                tip_x,
                tip_y,
                r1,
                r2,
            )

            print("2*gamma = {0}, J = {1}".format(2 * surface_energy, J))
