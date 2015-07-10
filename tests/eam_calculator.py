#! /usr/bin/env pytho

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

from __future__ import print_function

import random
import unittest

import numpy as np

import ase.io as io
from ase.calculators.test import numeric_force
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE
from ase.units import GPa

import matscipytest
from matscipy.calculators.eam import EAM
from matscipy.elasticity import fit_elastic_constants, Voigt_6x6_to_cubic

###

def numeric_stress(atoms, d=1e-6):
    """Evaluate stress tensor using finite differences.

    This will trigger 18 calls to get_potential_energy(), with appropriate
    cell deformation.
    """
    stress = np.zeros([3, 3])
    cell0 = atoms.get_cell().copy()

    for i in range(3):
        for j in range(3):
            cell = cell0.copy()
            eps0 = np.eye(3)
            eps = eps0.copy()

            eps[i, j] = eps0[i, j]-d
            cell = np.dot(cell0, eps)
            atoms.set_cell(cell, scale_atoms=True)
            e1 = atoms.get_potential_energy()

            eps[i, j] = eps0[i, j]+d
            cell = np.dot(cell0, eps)
            atoms.set_cell(cell, scale_atoms=True)
            e2 = atoms.get_potential_energy()

            stress[i, j] = (e2-e1)/(2*d)

    atoms.set_cell(cell0, scale_atoms=True)

    return np.array([stress[0,0], stress[1,1], stress[2,2],
                     (stress[1,2]+stress[2,1])/2,
                     (stress[0,2]+stress[2,0])/2,
                     (stress[0,1]+stress[1,0])/2])/atoms.get_volume()

###

class TestEAMCalculator(matscipytest.MatSciPyTestCase):

    disp = 1e-6
    tol = 1e-6

    def test_forces(self):
        for calc in [EAM('Au-Grochola-JCP05.eam.alloy')]:
            a = io.read('Au_923.xyz')
            a.center(vacuum=10.0)
            a.set_calculator(calc)
            f = a.get_forces()
            random.seed()
            for dummy in range(10):
                i = random.randrange(len(a))
                d = random.randrange(3)
                self.assertTrue((numeric_force(a, i, d, self.disp)-f[i, d]) <
                                self.tol)

    def test_stress(self):
        a = FaceCenteredCubic('Au', size=[2,2,2])
        calc = EAM('Au-Grochola-JCP05.eam.alloy')
        a.set_calculator(calc)
        self.assertArrayAlmostEqual(a.get_stress(), numeric_stress(a))

    def test_Grochola(self):
        a = FaceCenteredCubic('Au', size=[2,2,2])
        calc = EAM('Au-Grochola-JCP05.eam.alloy')
        a.set_calculator(calc)
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        a0 = a.cell.diagonal().mean()/2
        self.assertTrue(abs(a0-4.0701)<2e-5)
        self.assertTrue(abs(a.get_potential_energy()/len(a)+3.924)<0.0003)
        C, C_err = fit_elastic_constants(a, symmetry='cubic', verbose=False)
        C11, C12, C44 = Voigt_6x6_to_cubic(C)
        self.assertTrue(abs((C11-C12)/GPa-32.07)<0.7)
        self.assertTrue(abs(C44/GPa-45.94)<0.5)

    def test_CuAg(self):
        a = FaceCenteredCubic('Cu', size=[2,2,2])
        calc = EAM('CuAg.eam.alloy')
        a.set_calculator(calc)
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e_Cu = a.get_potential_energy()/len(a)

        a = FaceCenteredCubic('Ag', size=[2,2,2])
        a.set_calculator(calc)
        FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e_Ag = a.get_potential_energy()/len(a)
        self.assertTrue(abs(e_Ag+2.85)<1e-6)

        a = L1_2(['Ag', 'Cu'], size=[2,2,2], latticeconstant=4.0)
        a.set_calculator(calc)
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                               (syms=='Ag').sum()*e_Ag)/len(a)-0.096)<0.0005)

        a = B1(['Ag', 'Cu'], size=[2,2,2], latticeconstant=4.0)
        a.set_calculator(calc)
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                               (syms=='Ag').sum()*e_Ag)/len(a)-0.516)<0.0005)

        a = B2(['Ag', 'Cu'], size=[2,2,2], latticeconstant=4.0)
        a.set_calculator(calc)
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                               (syms=='Ag').sum()*e_Ag)/len(a)-0.177)<0.0003)

        a = L1_2(['Cu', 'Ag'], size=[2,2,2], latticeconstant=4.0)
        a.set_calculator(calc)
        FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
        e = a.get_potential_energy()
        syms = np.array(a.get_chemical_symbols())
        self.assertTrue(abs((e-(syms=='Cu').sum()*e_Cu-
                               (syms=='Ag').sum()*e_Ag)/len(a)-0.083)<0.0005)

###

if __name__ == '__main__':
    unittest.main()
