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


import numpy as np
from ase.constraints import StrainFilter
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE
from ase.units import GPa

from matscipy.calculators.eam import EAM
from matscipy.elasticity import (CubicElasticModuli, Voigt_6x6_to_cubic,
                                 cubic_to_Voigt_6x6, full_3x3x3x3_to_Voigt_6x6,
                                 measure_triclinic_elastic_constants,
                                 rotate_cubic_elastic_constants,
                                 rotate_elastic_constants)

fmax = 1e-6
delta = 1e-6


def test_rotation_au_eam(datafile_directory):
    """Test rotation of elastic constants for Au with EAM potential."""

    def make_atoms(a0, x):
        return FaceCenteredCubic("Au", size=[1, 1, 1], latticeconstant=a0, directions=x)

    calc = EAM(f"{datafile_directory}/Au-Grochola-JCP05.eam.alloy")

    # Initial relaxation to find equilibrium lattice constant
    a = make_atoms(None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    a.set_calculator(calc)
    FIRE(StrainFilter(a, mask=[1, 1, 1, 0, 0, 0]), logfile=None).run(fmax=fmax)
    latticeconstant = np.mean(a.cell.diagonal())

    # Measure elastic constants in reference configuration
    C6 = measure_triclinic_elastic_constants(a, delta=delta, fmax=fmax)
    C11, C12, C44 = Voigt_6x6_to_cubic(full_3x3x3x3_to_Voigt_6x6(C6)) / GPa

    el = CubicElasticModuli(C11, C12, C44)

    # Verify stiffness matrix matches measured values
    C_m = (
        full_3x3x3x3_to_Voigt_6x6(
            measure_triclinic_elastic_constants(a, delta=delta, fmax=fmax)
        )
        / GPa
    )
    np.testing.assert_allclose(el.stiffness(), C_m, atol=1e-7)

    # Test various crystal orientations
    test_directions = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Standard orientation
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # Cyclic permutation
        [[1, 1, 0], [0, 0, 1], [1, -1, 0]],  # 45° rotation
        [[1, 1, 1], [-1, -1, 2], [1, -1, 0]],  # Complex orientation
    ]

    for directions in test_directions:
        # Create rotated crystal
        a = make_atoms(latticeconstant, directions)
        a.set_calculator(calc)

        # Normalize direction vectors
        directions_normalized = np.array(
            [np.array(x) / np.linalg.norm(x) for x in directions]
        )

        # Test different rotation methods give consistent results
        C = el.rotate(directions_normalized)
        C_check = el._rotate_explicit(directions_normalized)
        C_check2 = rotate_cubic_elastic_constants(C11, C12, C44, directions_normalized)
        C_check3 = rotate_elastic_constants(
            cubic_to_Voigt_6x6(C11, C12, C44), directions_normalized
        )

        np.testing.assert_allclose(C, C_check, atol=1e-7)
        np.testing.assert_allclose(C, C_check2, atol=1e-7)
        np.testing.assert_allclose(C, C_check3, atol=1e-7)

        # Verify rotated constants match direct measurements
        C_m = (
            full_3x3x3x3_to_Voigt_6x6(
                measure_triclinic_elastic_constants(a, delta=delta, fmax=fmax)
            )
            / GPa
        )

        np.testing.assert_allclose(C, C_m, atol=1e-7)
