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

import ase.constraints
from ase.lattice.compounds import B1, B2, B3
from ase.lattice.cubic import BodyCenteredCubic, Diamond, FaceCenteredCubic
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
    ("Kumagai-dia-Si",
     Kumagai(kumagai.Kumagai_Comp_Mat_Sci_39_Si),
     Diamond("Si", size=[sx, sx, sx]),
     dict(Ec=4.630, a0=5.429, C11=166.4, C12=65.3, C44=77.1, C440=120.9)),
    ("StillingerWeber-dia-Si",
     StillingerWeber(stillinger_weber.Stillinger_Weber_PRB_31_5262_Si),
     Diamond("Si", size=[sx, sx, sx]),
     dict(Ec=4.3363, a0=5.431, C11=161.6, C12=81.6, C44=60.3, C440=117.2, B=108.3)),
    ("Tersoff3-dia-C",
     TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
     Diamond("C", size=[sx, sx, sx]),
     dict(Ec=7.396 - 0.0250, a0=3.566, C11=1067, C12=104, C44=636, C440=671)),
    ("Tersoff3-dia-Si",
     TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
     Diamond("Si", size=[sx, sx, sx]),
     dict(Ec=4.63, a0=5.432, C11=143, C12=75, C44=69, C440=119, B=98)),
    ("Tersoff3-dia-Si-C",
     TersoffBrenner(tersoff_brenner.Tersoff_PRB_39_5566_Si_C),
     B3(["Si", "C"], latticeconstant=4.3596, size=[sx, sx, sx]),
     dict(Ec=6.165, a0=4.321, C11=437, C12=118, C440=311, B=224)),
    ("MatsunagaFisherMatsubara-dia-C",
     TersoffBrenner(tersoff_brenner.Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N),
     Diamond("C", size=[sx, sx, sx]),
     dict(Ec=7.396 - 0.0250, a0=3.566, C11=1067, C12=104, C44=636, C440=671)),
    ("MatsunagaFisherMatsubara-dia-B-N",
     TersoffBrenner(tersoff_brenner.Matsunaga_Fisher_Matsubara_Jpn_J_Appl_Phys_39_48_B_C_N),
     B3(["B", "N"], latticeconstant=3.7, size=[sx, sx, sx]),
     dict(Ec=6.63, a0=3.658, B=385)),
    ("BrennerI-dia-C",
     TersoffBrenner(tersoff_brenner.Brenner_PRB_42_9458_C_I),
     Diamond("C", size=[sx, sx, sx]),
     {}),
    ("BrennerII-dia-C",
     TersoffBrenner(tersoff_brenner.Brenner_PRB_42_9458_C_II),
     Diamond("C", size=[sx, sx, sx]),
     dict(Ec=7.376 - 0.0524, a0=3.558, C11=621, C12=415, C44=383, B=484)),
    ("ErhartAlbeSiC-dia-C",
     TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_SiC),
     Diamond("C", size=[sx, sx, sx]),
     dict(Ec=7.3731, a0=3.566, C11=1082, C12=127, C44=673, B=445)),
    ("ErhartAlbeSiC-dia-Si",
     TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_SiC),
     Diamond("Si", size=[sx, sx, sx]),
     dict(Ec=4.63, a0=5.429, C11=167, C12=65, C44=60 ,C440=105, B=99)),
    ("ErhartAlbeSiC-dia-SiII",
     TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_Si),
     Diamond("Si", size=[sx, sx, sx]),
     dict(Ec=4.63, a0=5.429, C11=167, C12=65, C44=72, C440=111, B=99)),
    ("ErhartAlbeSiC-dia-Si-C",
     TersoffBrenner(tersoff_brenner.Erhart_PRB_71_035211_SiC),
     B3(["Si", "C"], latticeconstant=4.3596, size=[sx, sx, sx]),
     dict(Ec=6.340, a0=4.359, C11=382, C12=145, C440=305, B=224)),
# FIXME! These potential don't work yet.
#    ("AlbeNordlundAverbackPtC-fcc-Pt",
#     TersoffBrenner(tersoff_brenner.Albe_PRB_65_195124_PtC),
#     FaceCenteredCubic("Pt", size=[sx, sx, sx]),
#     dict(Ec=5.77, a0=3.917, C11=351.5, C12=248.1, C44=89.5, B=282.6)),
#    ("AlbeNordlundAverbackPtC-bcc-Pt",
#     TersoffBrenner(tersoff_brenner.Albe_PRB_65_195124_PtC),
#     BodyCenteredCubic("Pt", latticeconstant=3.1, size=[sx, sx, sx]),
#     dict(Ec=5.276, a0=3.094, B=245.5)),
#    ("AlbeNordlundAverbackPtC-B1-PtC",
#     TersoffBrenner(tersoff_brenner.Albe_PRB_65_195124_PtC),
#     B1(["Pt", "C"], latticeconstant=4.5, size=[sx, sx, sx]),
#     dict(Ec=10.271, a0=4.476, B=274)),
#    ("AlbeNordlundAverbackPtC-B2-PtC",
#     TersoffBrenner(tersoff_brenner.Albe_PRB_65_195124_PtC),
#     B2(["Pt", "C"], latticeconstant=2.7, size=[sx, sx, sx]),
#     dict(Ec=9.27, a0=2.742, B=291)),
]


@pytest.mark.parametrize('test', tests)
def test_cubic_elastic_constants(test):
    name, pot, atoms, mat = test

    calculator = Manybody(**pot)
    atoms.translate([0.1, 0.1, 0.1])
    atoms.set_scaled_positions(atoms.get_scaled_positions())
    atoms.calc = calculator

    c1, c2, c3 = atoms.get_cell()
    a0 = np.sqrt(np.dot(c1, c1)) / sx

    FIRE(ase.constraints.StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None).run(fmax=0.0001)

    # Ec
    Ec = -atoms.get_potential_energy() / len(atoms)

    # a0
    c1, c2, c3 = atoms.get_cell()
    a0 = np.sqrt(np.dot(c1, c1)) / sx

    # C11, C12, C44
    Caffine = calculator.get_birch_coefficients(atoms)
    Cnonaffine = calculator.get_non_affine_contribution_to_elastic_constants(atoms)
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
