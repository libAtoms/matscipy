#! /usr/bin/env python

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, University of Freiburg
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

import pytest

import numpy as np

import ase.constraints
from ase.lattice.compounds import B3
from ase.lattice.cubic import Diamond
from ase.optimize import FIRE
from ase.units import GPa

import matscipy.calculators.manybody.explicit_forms.stillinger_weber as stillinger_weber
import matscipy.calculators.manybody.explicit_forms.kumagai as kumagai
import matscipy.calculators.manybody.explicit_forms.tersoff_brenner as tersoff_brenner
from matscipy.calculators.manybody import Manybody
from matscipy.calculators.manybody.explicit_forms import Kumagai, TersoffBrenner, StillingerWeber
from matscipy.elasticity import full_3x3x3x3_to_Voigt_6x6

###

sx = 1

tests = [
    ("Kumagai-dia-Si", Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
     dict(struct=Diamond("Si", size=[sx, sx, sx]),
          Ec=4.630, a0=5.429, C11=166.4, C12=65.3, C44=77.1, C440=120.9)),
    ("StillingerWeber-dia-Si", StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si),
     dict(struct=Diamond("Si", size=[sx, sx, sx]),
          Ec=4.630, a0=5.431, C11=161.6, C12=81.6, C44=60.3, C440=117.2, B=108.3)),
    ("Tersoff3-dia-C", TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
     dict(struct=Diamond("C", size=[sx, sx, sx]),
          Ec=7.396 - 0.0250, a0=3.566, C11=1067, C12=104, C44=636, C440=671)),
    ("Tersoff3-dia-Si", TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
     dict(struct=Diamond("Si", size=[sx, sx, sx]),
          Ec=4.63, a0=5.432, C11=143, C12=75, C44=69, C440=119, B=98)),
    ("Tersoff3-dia-Si-C", TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
     dict(struct=B3(["Si", "C"], latticeconstant=4.3596, size=[sx, sx, sx]),
          Ec=6.165, a0=4.321, C11=437, C12=118, C440=311, B=224)),
    ("MatsunagaFisherMatsubara-dia-C",
     TersoffBrenner(tersoff_brenner.Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N),
     dict(struct=Diamond("C", size=[sx, sx, sx]),
          Ec=7.396 - 0.0250, a0=3.566, C11=1067, C12=104, C44=636, C440=671)),
    ("MatsunagaFisherMatsubara-dia-B-N",
     TersoffBrenner(tersoff_brenner.Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N),
     dict(struct=B3(["B", "N"], latticeconstant=3.7, size=[sx, sx, sx]),
          Ec=6.63, a0=3.658, B=385)),
]


@pytest.mark.parametrize('test', tests)
def test_cubic_elastic_constants(test):
    name, pot, mat = test

    calculator = Manybody(**pot)
    atoms = mat["struct"]
    atoms.translate([0.1, 0.1, 0.1])
    atoms.set_scaled_positions(atoms.get_scaled_positions())
    atoms.set_calculator(calculator)

    FIRE(ase.constraints.StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None).run(fmax=0.0001)

    # Ec
    Ec = -atoms.get_potential_energy() / len(atoms)

    # a0
    c1, c2, c3 = atoms.get_cell()
    a0 = np.sqrt(np.dot(c1, c1)) / sx

    # C11, C12, C44
    Caffine = calculator.get_birch_coefficients(atoms)
    Cnonaffine = calculator.get_non_affine_contribution_to_elastic_constants(atoms, tol=1e-5)
    C = full_3x3x3x3_to_Voigt_6x6(Caffine + Cnonaffine)
    C11 = C[0, 0] / GPa
    C12 = C[0, 1] / GPa
    C44 = C[3, 3] / GPa
    C110 = full_3x3x3x3_to_Voigt_6x6(Caffine)[0, 0] / GPa
    C120 = full_3x3x3x3_to_Voigt_6x6(Caffine)[0, 1] / GPa
    C440 = full_3x3x3x3_to_Voigt_6x6(Caffine)[3, 3] / GPa
    B = (C11 + 2 * C12) / 3

    # print to screen
    print()
    print(f'=== {name}===')
    l1 = f'          '
    l2 = f'Computed: '
    l3 = f'Reference:'
    for prop in ['Ec', 'a0', 'C11', 'C110', 'C12', 'C120', 'C44', 'C440', 'B']:
        l1 += prop.rjust(11)
        l2 += f'{locals()[prop]:.2f}'.rjust(11)
        if prop in mat:
            l3 += f'{float(mat[prop]):.2f}'.rjust(11)
        else:
            l3 += '---'.rjust(11)
    print(l1)
    print(l2)
    print(l3)

    # actually test properties
    for prop, value in mat.items():
        if prop != 'struct':
            np.testing.assert_allclose(locals()[prop], value, rtol=0.1)
