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
from matscipy import spatial_correlation_function
import ase.io as io

class TestSpatialCorrelationFunction(unittest.TestCase):
    
    def test_RDF(self):
        atoms = io.read('aC.cfg')
        values = np.ones(4001)
        cell = atoms.get_cell()[:3,:].T #should be [[xx,yx,zx],[xy,yy,zy],[xz,yz,zz]]

        length_cutoff = 10.0
        output_gridsize = 0.25
        FFT_cutoff = 0.0 #check FFT part, not short range
        approx_FFT_gridsize = 1.0

        SCF=spatial_correlation_function.spatial_correlation_function(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize,dim=None,delta='simple',norm=False)


    def test_peak_count(self):
        n=50

        xyz=np.zeros((n,3))
        xyz[:,0]=np.arange(n)
        values=np.random.rand(len(xyz))
        
        atoms=Atoms(positions=xyz)
        cell=np.array([[n,0,0],[0,n,0],[0,0,n]])

        length_cutoff= n/2.
        output_gridsize= 0.1
        FFT_cutoff= 7.5
        approx_FFT_gridsize= 1.
        SCF=spatial_correlation_function.spatial_correlation_function(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize,dim=None,delta='simple',norm=True)
        SCF2=spatial_correlation_function.spatial_correlation_function(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize/50.*n,dim=None,delta='simple',norm=True)
        SCF3=spatial_correlation_function.spatial_correlation_function(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize/20.*n,dim=None,delta='simple',norm=True)
        
        SCF-=SCF.min()
        SCF2-=SCF2.min()
        SCF3-=SCF3.min()
        
        self.assertTrue((np.isfinite(SCF/SCF)).sum()==np.floor(length_cutoff))
        self.assertTrue((np.isfinite(SCF2/SCF2)).sum()==int(np.floor(FFT_cutoff)+np.ceil(np.ceil(length_cutoff-FFT_cutoff)*50./n)))
        self.assertTrue((np.isfinite(SCF3/SCF3)).sum()==int(np.floor(FFT_cutoff)+np.ceil(np.ceil(length_cutoff-FFT_cutoff)*20./n)))


    def test_directional_spacing(self):
        n=30

        xyz=np.zeros((n**3,3))
        m=np.meshgrid(np.arange(0,n,1),np.arange(0,2*n,2),np.arange(0,3*n,3))
        xyz[:,0]=m[0].reshape((-1))
        xyz[:,1]=m[0].reshape((-1))
        xyz[:,2]=m[2].reshape((-1))

        values=np.random.rand(len(xyz))
        atoms=Atoms(positions=xyz)
        cell=np.array([[n,0,0],[0,2*n,0],[0,0,3*n]])

        length_cutoff= n
        output_gridsize= 0.1
        FFT_cutoff= 0.
        approx_FFT_gridsize= 1.

        SCF0=spatial_correlation_function.spatial_correlation_function(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize, dim=0, delta='simple', norm=True)
        SCF1=spatial_correlation_function.spatial_correlation_function(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize, dim=1, delta='simple', norm=True)
        SCF2=spatial_correlation_function.spatial_correlation_function(atoms, values, cell, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize, dim=2, delta='simple', norm=True)

        SCF0-=SCF0.min()
        SCF1-=SCF1.min()
        SCF2-=SCF2.min()

        n_peaks0=np.isfinite(SCF0/SCF0).sum()
        n_peaks1=np.isfinite(SCF1/SCF1).sum()
        n_peaks2=np.isfinite(SCF2/SCF2).sum()

        self.assertTrue(n_peaks0/2.-n_peaks1 < 2)
        self.assertTrue(n_peaks0/3.-n_peaks2 < 2)

###
if __name__ == '__main__':
    unittest.main()
