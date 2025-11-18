#
# Copyright 2014-2015, 2017, 2020-2021 Lars Pastewka (U. Freiburg)
#           2014 James Kermode (Warwick U.)
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
# Copyright (2014-2017) James Kermode, Warwick University
#                       Lars Pastewka, Karlsruhe Institute of Technology
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

import numpy as np
from ase.build import bulk

from matscipy.calculators import EAM, SupercellCalculator


def test_eam(datafile_directory):
    """Test that SupercellCalculator gives identical results to base calculator.

    Verifies that wrapping an EAM calculator with SupercellCalculator produces
    the same energy, forces, and stress as the original calculator.
    """
    for calc in [EAM(f"{datafile_directory}/Au-Grochola-JCP05.eam.alloy")]:
        # Create test system
        a = bulk("Au")
        a *= (2, 2, 2)
        a.rattle(0.1)

        # Calculate with base calculator
        a.calc = calc
        e = a.get_potential_energy()
        f = a.get_forces()
        s = a.get_stress()

        # Calculate with supercell calculator wrapper
        a.set_calculator(SupercellCalculator(calc, (3, 3, 3)))

        # Verify results match
        assert (
            abs(e - a.get_potential_energy()) < 1e-7
        ), f"Energy mismatch: {e} vs {a.get_potential_energy()}"

        f_new = a.get_forces()
        assert np.allclose(
            f, f_new, atol=1e-7
        ), f"Forces differ: max difference = {np.abs(f - f_new).max()}"

        s_new = a.get_stress()
        assert np.allclose(
            s, s_new, atol=1e-7
        ), f"Stress differs: max difference = {np.abs(s - s_new).max()}"
