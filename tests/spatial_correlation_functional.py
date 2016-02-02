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
from matscipy.neighbours import neighbour_list
from ase import Atoms
import matscipytest
from matscipy import spatial_correlation_functional


class TestSpatialCorrelationFunctional(unittest.TestCase):
    

    def test_peak_count(self):
        n=100

        xyz=np.zeros((n,3))
        xyz[:,0]=np.arange(n)
        values=np.random.rand(len(xyz))
        
        atoms=Atoms(positions=xyz)
        cell=np.array([[n,0,0],[0,n,0],[0,0,n]])

        length_cutoff= n/2.
        output_gridsize= 0.1
        FFT_cutoff= 7.5
        approx_FFT_gridsize= 1.
        SCF=spatial_correlation_functional.spatial_correlation_functional(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize)
        SCF2=spatial_correlation_functional.spatial_correlation_functional(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize/50.*n)
        SCF3=spatial_correlation_functional.spatial_correlation_functional(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize/20.*n)
        
        SCF-=SCF.min()
        SCF2-=SCF2.min()
        SCF3-=SCF3.min()
        
        self.assertTrue((np.isfinite(SCF/SCF)).sum()==np.floor(length_cutoff))
        self.assertTrue((np.isfinite(SCF2/SCF2)).sum()==int(np.floor(FFT_cutoff)+np.ceil(np.ceil(length_cutoff-FFT_cutoff)*50./n)))
        self.assertTrue((np.isfinite(SCF3/SCF3)).sum()==int(np.floor(FFT_cutoff)+np.ceil(np.ceil(length_cutoff-FFT_cutoff)*20./n)))



###
if __name__ == '__main__':
    unittest.main()
