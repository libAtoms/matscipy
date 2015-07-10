#! /usr/bin/env python

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#                  Adrien Gola, Karlsruhe Institute of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

from __future__ import division

import unittest

import numpy as np

import os
from matscipy.eam.io import read_eam,read_eam_alloy,write_eam,write_eam_alloy,mix_eam_alloy
try:
    from scipy import interpolate
    from matscipy.eam.calculator import EAM
except:
    print('Warning: No scipy')
    interpolate = False


import ase.io as io
from ase.calculators.test import numeric_force
from ase.constraints import StrainFilter, UnitCellFilter
from ase.lattice.compounds import B1, B2, L1_0, L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import FIRE

import matscipytest

###

class TestEAMIO(matscipytest.MatSciPyTestCase):

    tol = 1e-6

    def test_eam_read_write(self):
        source,parameters,F,f,rep = read_eam("Au_u3.eam")
        write_eam(source,parameters,F,f,rep,"Au_u3_copy.eam")
        source1,parameters1,F1,f1,rep1 = read_eam("Au_u3_copy.eam")
        os.remove("Au_u3_copy.eam")
        for i,p in enumerate(parameters):
            try:
                diff = p - parameters1[i]
            except:
                diff = None
            if diff is None:
                self.assertTrue(p == parameters1[i])
            else:
                self.assertTrue(diff < self.tol)
        self.assertTrue((F == F1).all())
        self.assertTrue((f == f1).all())
        self.assertTrue((rep == rep1).all())
    
    def test_eam_alloy_read_write(self):
        source,parameters,F,f,rep = read_eam_alloy("CuAgNi_Zhou.eam.alloy")
        write_eam_alloy(source,parameters,F,f,rep,"CuAgNi_Zhou.eam_copy.alloy")
        source1,parameters1,F1,f1,rep1 = read_eam_alloy("CuAgNi_Zhou.eam_copy.alloy")
        os.remove("CuAgNi_Zhou.eam_copy.alloy")
        fail = 0
        for i,p in enumerate(parameters):
            try:
              for j,d in enumerate(p):
                  if d != parameters[i][j]:
                      fail+=1
            except:
                if p != parameters[i]:
                    fail +=1
        self.assertTrue(fail == 0)
        self.assertTrue((F == F1).all())
        self.assertTrue((f == f1).all())
        for i in range(len(rep)):
            for j in range(len(rep)):
                if j < i :
                    self.assertTrue((rep[i,j,:] == rep1[i,j,:]).all())
         
    def test_mix_eam_alloy(self):
        try:
            from scipy import interpolate
            source,parameters,F,f,rep = read_eam_alloy("CuAu_Zhou.eam.alloy")
            source1,parameters1,F1,f1,rep1 = mix_eam_alloy(["Cu_Zhou.eam.alloy","Au_Zhou.eam.alloy"],"weight")
            write_eam_alloy(source1,parameters1,F1,f1,rep1,"CuAu_mixed.eam.alloy")

            calc0 = EAM('CuAu_Zhou.eam.alloy')
            calc1 = EAM('CuAu_mixed.eam.alloy')
       
            a = FaceCenteredCubic('Cu', size=[2,2,2])
            a.set_calculator(calc0)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = FaceCenteredCubic('Cu', size=[2,2,2])
            a.set_calculator(calc1)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = FaceCenteredCubic('Au', size=[2,2,2])
            a.set_calculator(calc0)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = FaceCenteredCubic('Au', size=[2,2,2])
            a.set_calculator(calc1)
            FIRE(StrainFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = L1_2(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc0)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = L1_2(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc1)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = L1_2(['Cu', 'Au'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc0)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = L1_2(['Cu', 'Au'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc1)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)

            a = B1(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc0)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e0 = a.get_potential_energy()/len(a)
            a = B1(['Au', 'Cu'], size=[2,2,2], latticeconstant=4.0)
            a.set_calculator(calc1)
            FIRE(UnitCellFilter(a, mask=[1,1,1,0,0,0]), logfile=None).run(fmax=0.001)
            e1 = a.get_potential_energy()/len(a)
            self.assertTrue(e0-e1 < 0.0005)
          
            os.remove("CuAu_mixed.eam.alloy")
          
        except:
	    print('Warning: No scipy')
	    print('Cannot test mix_eam_alloy')
            
###

if __name__ == '__main__':
    unittest.main()
