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
from ase.optimize import FIRE
from ase.units import GPa

import matscipy.calculators.bop_sw.explicit_forms.stillinger_weber as sw
import matscipy.calculators.bop_sw.explicit_forms.kumagai as kum
import matscipy.calculators.bop_sw.explicit_forms.tersoff3 as t3
from matscipy.calculators.bop_sw import AbellTersoffBrennerStillingerWeber
from matscipy.calculators.bop_sw.explicit_forms import KumagaiTersoff, AbellTersoffBrenner, StillingerWeber
from matscipy.elasticity import full_3x3x3x3_to_Voigt_6x6



from ase.lattice.cubic import Diamond

###

sx = 1

tests = [
    ("Kumagai", KumagaiTersoff(kum.kumagai),
      dict(name="dia-Si", struct=Diamond("Si", size=[sx,sx,sx]),
            Ec=-4.630, a0=5.429, C11=166.4, C12=65.3, C44=77.1, C440=120.9)
      ),
    ("StillingerWeber", StillingerWeber(sw.original_SW),
      dict(name="dia-Si", struct=Diamond("Si", size=[sx,sx,sx]),
            Ec=-4.630, a0=5.431, C11=161.6, C12=81.6, C44=60.3, C440=117.2, B=108.3)
      ),
    ("Tersoff3", AbellTersoffBrenner(t3.Tersoff_PRB_39_5566_Si_C),
        dict(name="dia-Si", struct=Diamond("Si", size=[sx,sx,sx]),
             Ec=-4.63, a0=5.432, C11=143, C12=75, C44=69, C440=119, B=98)
      )
    ]


@pytest.mark.parametrize('test', tests)
def test_cubic_elastic_constants(test):
    pot, par, mats = test

    calculator = AbellTersoffBrennerStillingerWeber(**par)
    atoms = mats["struct"]
    atoms.translate([0.1, 0.1, 0.1])
    atoms.set_scaled_positions(atoms.get_scaled_positions())
    atoms.set_calculator(calculator)

    FIRE(
        ase.constraints.StrainFilter(atoms, mask=[1,1,1,0,0,0]),
        logfile=None).run(fmax=0.0001)

    # Ec
    Ec = atoms.get_potential_energy()/len(atoms)
    if "Ec" in mats:
        np.isclose(Ec, mats["Ec"], atol=1e-8, rtol=1e-8)
    else:
        print("Ec = ", Ec, " eV")

    # a0
    c1, c2, c3 = atoms.get_cell()
    a0 = np.sqrt(np.dot(c1, c1))/sx
    if "a0" in mats:
        np.isclose(a0, mats["a0"], atol=1e-8, rtol=1e-8)
    else:
        print("a0 = ", a0, " A")


    # C11, C12, C44
    Caffine = calculator.get_birch_coefficients(atoms)
    Cnonaffine = calculator.get_non_affine_contribution_to_elastic_constants(atoms, tol=1e-5)
    C =  full_3x3x3x3_to_Voigt_6x6(Caffine+Cnonaffine)
    C11 = C[0,0]/GPa
    C12 = C[0,1]/GPa
    C44 = C[3,3]/GPa
    C110 = full_3x3x3x3_to_Voigt_6x6(Caffine)[0,0]/GPa
    C120 = full_3x3x3x3_to_Voigt_6x6(Caffine)[0,1]/GPa
    C440 = full_3x3x3x3_to_Voigt_6x6(Caffine)[3,3]/GPa
    B  = (C11+2*C12)/3

    if "C11" in mats:
        np.isclose(C11, mats["C11"], atol=1e-10, rtol=1e-8)
        #print("C11 = ", C11, " GPa")
    else:
        print("C11 = ", C11, " GPa")

    if "C12" in mats:
        np.isclose(C12, mats["C12"], atol=1e-10, rtol=1e-8)
        #print("C12 = ", C12, " GPa")
    else:
        print("C12 = ", C12, " GPa")

    if "C44" in mats:
        np.isclose(C44, mats["C44"], atol=1e-10, rtol=1e-8)
        #print("C44 = ", C44, " GPa")
    else:
        print("C44 = ", C44, " GPa")

    if "C110" in mats:
        np.isclose(C110, mats["C110"], atol=1e-10, rtol=1e-8)
        #print("C110 = ", C110, " GPa")
    else:
        print("C110 = ", C110, " GPa")

    if "C120" in mats:
        np.isclose(C120, mats["C120"], atol=1e-10, rtol=1e-8)
        #print("C120 = ", C120, " GPa")
    else:
        print("C120 = ", C120, " GPa")

    if "C440" in mats:
        np.isclose(C440, mats["C440"], atol=1e-10, rtol=1e-8)
        #print("C440 = ", C440, " GPa")
    else:
        print("C440 = ", C440, " GPa")

    if "B" in mats:
        np.isclose(B, mats["C440"], atol=1e-10, rtol=1e-8)
        #print("C440 = ", C440, " GPa")
    else:
        print("B = ", B, " GPa")
