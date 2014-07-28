#! /usr/bin/env python

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

import unittest

import numpy as np

from ase.constraints import StrainFilter
from ase.lattice.cubic import Diamond, FaceCenteredCubic
from ase.optimize import FIRE
from ase.units import GPa

from atomistica import Kumagai, TabulatedAlloyEAM

from matscipy.elasticity import CubicElasticModuli
from matscipy.elasticity import measure_orthorhombic_elastic_moduli
from matscipy.elasticity import Voigt_6x6_to_orthorhombic

###

class TestCubicElasticModuli(unittest.TestCase):

    fmax = 1e-6
    delta = 1e-6

    def test_rotation(self):
        for make_atoms, calc in [ 
            ( lambda a0,x : FaceCenteredCubic('Au', size=[1,1,1],
                                              latticeconstant=a0, directions=x),
              TabulatedAlloyEAM(fn='Au-Grochola-JCP05.eam.alloy') ),
            ( lambda a0,x : Diamond('Si', size=[1,1,1], latticeconstant=a0,
                                    directions=x),
              Kumagai() )
            ]:

            a = make_atoms(None, [[1,0,0], [0,1,0], [0,0,1]])
            a.set_calculator(calc)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None) \
                .run(fmax=self.fmax)
            latticeconstant = np.mean(a.cell.diagonal())

            C11, C12, C44 = \
                measure_orthorhombic_elastic_moduli(a, delta=self.delta,
#                                                    optimizer=FIRE,
                                                    fmax=self.fmax)
            C11 = np.mean(C11)/GPa
            C12 = np.mean(C12)/GPa
            C44 = np.mean(C44)/GPa

            el = CubicElasticModuli(C11, C12, C44)

            for directions in [ [[1,0,0], [0,1,0], [0,0,1]],
                                [[0,1,0], [0,0,1], [1,0,0]],
                                [[1,1,0], [0,0,1], [1,-1,0]],
                                [[1,1,1], [-1,-1,2], [1,-1,0]] ]:
                a, b, c = directions
                print 'Directions: ', a, b, c

                directions = np.array([ np.array(x)/np.linalg.norm(x) 
                                        for x in directions ])
                a = make_atoms(latticeconstant, directions)
                a.set_calculator(calc)

                C11_m, C12_m, C44_m = \
                    measure_orthorhombic_elastic_moduli(a, delta=self.delta,
#                                                        optimizer=FIRE,
                                                        fmax=self.fmax)
                C11_m /= GPa
                C12_m /= GPa
                C44_m /= GPa

                C = el.rotate(directions)
                C_check = el._rotate_explicit(directions)
                self.assertTrue(np.all(np.abs(C-C_check) < 1e-6))
                
                C11_rot, C12_rot, C44_rot = Voigt_6x6_to_orthorhombic(C)

                dC11 = C11_m-C11_rot
                dC12 = C12_m-C12_rot
                dC44 = C44_m-C44_rot

                if np.any(np.abs(dC11) > 1.0):
                    print '--- C11 ---'
                    print 'measured:   ', C11_m
                    print 'rotated:    ', C11_rot
                    print 'difference: ', dC11
                if np.any(np.abs(dC12) > 1.0):
                    print '--- C12 ---'
                    print 'measured:   ', C12_m
                    print 'rotated:    ', C12_rot
                    print 'difference: ', dC12
                if np.any(np.abs(dC44) > 1.0):
                    print '--- C44 ---'
                    print 'measured:   ', C44_m
                    print 'rotated:    ', C44_rot
                    print 'difference: ', dC44

                self.assertTrue(np.all(np.abs(dC11) < 1e-2))
                self.assertTrue(np.all(np.abs(dC12) < 1e-2))

###

if __name__ == '__main__':
    unittest.main()
