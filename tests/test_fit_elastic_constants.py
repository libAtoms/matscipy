#
# Copyright 2014, 2020 James Kermode (Warwick U.)
#           2020 Lars Pastewka (U. Freiburg)
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

import unittest

import numpy as np

import ase.io
import ase.units as units
from ase.constraints import StrainFilter
from ase.lattice.cubic import Diamond
from ase.optimize import FIRE

try:
    import quippy.potential
except ImportError:
    quippy = None

import matscipytest
from matscipy.elasticity import (fit_elastic_constants,
                                 measure_triclinic_elastic_constants)

if quippy is not None:
    
    class TestFitElasticConstants(matscipytest.MatSciPyTestCase):
        """
        Tests of elastic constant calculation.

        We test with a cubic Silicon lattice, since this also has most of the
        symmetries of the lower-symmetry crystal families
        """

        def setUp(self):
            self.pot = quippy.potential.Potential('IP SW', param_str="""
            <SW_params n_types="2" label="PRB_31_plus_H">
            <comment> Stillinger and Weber, Phys. Rev. B  31 p 5262 (1984), extended for other elements </comment>
            <per_type_data type="1" atomic_num="1" />
            <per_type_data type="2" atomic_num="14" />
            <per_pair_data atnum_i="1" atnum_j="1" AA="0.0" BB="0.0"
                  p="0" q="0" a="1.0" sigma="1.0" eps="0.0" />
            <per_pair_data atnum_i="1" atnum_j="14" AA="8.581214" BB="0.0327827"
                  p="4" q="0" a="1.25" sigma="2.537884" eps="2.1672" />
            <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
                  p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />

            <!-- triplet terms: atnum_c is the center atom, neighbours j and k -->
            <per_triplet_data atnum_c="1"  atnum_j="1"  atnum_k="1"
                  lambda="21.0" gamma="1.20" eps="2.1675" />
            <per_triplet_data atnum_c="1"  atnum_j="1"  atnum_k="14"
                  lambda="21.0" gamma="1.20" eps="2.1675" />
            <per_triplet_data atnum_c="1"  atnum_j="14" atnum_k="14"
                  lambda="21.0" gamma="1.20" eps="2.1675" />

            <per_triplet_data atnum_c="14" atnum_j="1"  atnum_k="1"
                  lambda="21.0" gamma="1.20" eps="2.1675" />
            <per_triplet_data atnum_c="14" atnum_j="1"  atnum_k="14"
                  lambda="21.0" gamma="1.20" eps="2.1675" />
            <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14"
                  lambda="21.0" gamma="1.20" eps="2.1675" />
            </SW_params>
            """)

            self.fmax = 1e-4
            self.at0 = Diamond('Si', latticeconstant=5.43)
            self.at0.set_calculator(self.pot)
            # relax initial positions and unit cell
            FIRE(StrainFilter(self.at0, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=self.fmax)

            self.C_ref = np.array([[ 151.4276439 ,   76.57244456,   76.57244456,    0.        ,    0.        ,    0.        ],
                                 [  76.57244456,  151.4276439 ,   76.57244456,    0.        ,    0.        ,    0.        ],
                                 [  76.57244456,   76.57244456,  151.4276439 ,    0.        ,    0.        ,    0.        ],
                                 [   0.        ,    0.        ,    0.        ,  109.85498798,    0.        ,    0.        ],
                                 [   0.        ,    0.        ,    0.        ,    0.        ,  109.85498798,    0.        ],
                                 [   0.        ,    0.        ,    0.        ,    0.        ,   0.        ,  109.85498798]])

            self.C_err_ref = np.array([[ 1.73091718,  1.63682097,  1.63682097,  0.        ,  0.        ,     0.        ],
                                     [ 1.63682097,  1.73091718,  1.63682097,  0.        ,  0.        ,     0.        ],
                                     [ 1.63682097,  1.63682097,  1.73091718,  0.        ,  0.        ,     0.        ],
                                     [ 0.        ,  0.        ,  0.        ,  1.65751232,  0.        ,     0.        ],
                                     [ 0.        ,  0.        ,  0.        ,  0.        ,  1.65751232,     0.        ],
                                     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,     1.65751232]])

            self.C_ref_relaxed = np.array([[ 151.28712587,   76.5394162 ,   76.5394162 ,    0.        ,
                                           0.        ,    0.        ],
                                         [  76.5394162 ,  151.28712587,   76.5394162 ,    0.        ,
                                            0.        ,    0.        ],
                                         [  76.5394162 ,   76.5394162 ,  151.28712587,    0.        ,
                                            0.        ,    0.        ],
                                         [   0.        ,    0.        ,    0.        ,   56.32421772,
                                             0.        ,    0.        ],
                                         [   0.        ,    0.        ,    0.        ,    0.        ,
                                             56.32421772,    0.        ],
                                         [   0.        ,    0.        ,    0.        ,    0.        ,
                                             0.        ,   56.32421772]])

            self.C_err_ref_relaxed = np.array([[ 1.17748661,  1.33333615,  1.33333615,  0.        ,  0.        ,
                                               0.        ],
                                             [ 1.33333615,  1.17748661,  1.33333615,  0.        ,  0.        ,
                                               0.        ],
                                             [ 1.33333615,  1.33333615,  1.17748661,  0.        ,  0.        ,
                                               0.        ],
                                             [ 0.        ,  0.        ,  0.        ,  0.18959684,  0.        ,
                                               0.        ],
                                             [ 0.        ,  0.        ,  0.        ,  0.        ,  0.18959684,
                                               0.        ],
                                             [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                               0.18959684]])

        def test_measure_triclinic_unrelaxed(self):
            # compare to brute force method without relaxation
            C = measure_triclinic_elastic_constants(self.at0, delta=1e-2, optimizer=None)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref, tol=0.2)

        def test_measure_triclinic_relaxed(self):
            # compare to brute force method with relaxation            
            C = measure_triclinic_elastic_constants(self.at0, delta=1e-2, optimizer=FIRE,
                                                    fmax=self.fmax)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref_relaxed, tol=0.2)
 
        def testcubic_unrelaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'cubic', verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref, tol=1e-2)
            if abs(C_err).max() > 0.:
                self.assertArrayAlmostEqual(C_err/units.GPa, self.C_err_ref, tol=1e-2)

        def testcubic_relaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'cubic',
                                             optimizer=FIRE, fmax=self.fmax,
                                             verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref_relaxed, tol=1e-2)
            if abs(C_err).max() > 0.:
                self.assertArrayAlmostEqual(C_err/units.GPa, self.C_err_ref_relaxed, tol=1e-2)

        def testorthorhombic_unrelaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'orthorhombic',
                                             verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref, tol=0.1)

        def testorthorhombic_relaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'orthorhombic',
                                             optimizer=FIRE, fmax=self.fmax,
                                             verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref_relaxed, tol=0.1)

        def testmonoclinic_unrelaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'monoclinic',
                                             verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref, tol=0.2)

        def testmonoclinic_relaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'monoclinic',
                                             optimizer=FIRE, fmax=self.fmax,
                                             verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref_relaxed, tol=0.2)

        def testtriclinic_unrelaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'triclinic',
                                             verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref, tol=0.2)

        def testtriclinic_relaxed(self):
            C, C_err = fit_elastic_constants(self.at0, 'triclinic',
                                             optimizer=FIRE, fmax=self.fmax,
                                             verbose=False, graphics=False)
            self.assertArrayAlmostEqual(C/units.GPa, self.C_ref_relaxed, tol=0.2)

if __name__ == '__main__':
    unittest.main()

