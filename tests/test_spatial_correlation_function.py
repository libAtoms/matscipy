#
# Copyright 2016, 2020-2021 Lars Pastewka (U. Freiburg)
#           2016 Richard Jana (KIT & U. Freiburg)
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
from matscipy.neighbours import neighbour_list
from ase import Atoms
import matscipytest
from matscipy.spatial_correlation_function import spatial_correlation_function
import ase.io as io

#import matplotlib.pyplot as plt

class TestSpatialCorrelationFunction(unittest.TestCase):

    def test_radial_distribution_function(self):
        # Radial distribution function is obtained for correlations of unity.
        # This is not a particularly good test.
        # Tilt unit cell to check results for non-orthogonal cells.
        for i in range(-1,2):
            a = io.read('aC.cfg')
            a1, a2, a3 = a.cell
            a3[0] += i*a1[0]
            a.set_cell([a1, a2, a3])
            a.set_scaled_positions(a.get_scaled_positions())

            v = np.ones(len(a))
            SCF1, r1 = spatial_correlation_function(a, v, 10.0, 0.1, 8.0, 0.1,
                                                    norm=False)
            SCF2, r2 = spatial_correlation_function(a, v, 10.0, 0.1, 2.0, 0.1,
                                                    norm=False)

            #print(np.abs(SCF1-SCF2).max())
            #print(r1[np.argmax(np.abs(SCF1-SCF2))])
            #plt.plot(r1, SCF1, label='$8$')
            #plt.plot(r2, SCF2, label='$2$')
            #plt.legend(loc='best')
            #plt.show()

            self.assertTrue(np.abs(SCF1-SCF2).max() < 0.31)

#    def test_peak_count(self):
#        n=50
#
#        xyz=np.zeros((n,3))
#        xyz[:,0]=np.arange(n)
#        values=np.random.rand(len(xyz))
#
#        atoms=Atoms(positions=xyz, cell=np.array([[n,0,0],[0,n,0],[0,0,n]]))
#
#        length_cutoff= n/2.
#        output_gridsize= 0.1
#        FFT_cutoff= 7.5
#        approx_FFT_gridsize= 1.
#        SCF, r=spatial_correlation_function(atoms, values, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize,dim=None,delta='simple',norm=True)
#        SCF2, r2=spatial_correlation_function(atoms, values, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize/50.*n,dim=None,delta='simple',norm=True)
#        SCF3, r3=spatial_correlation_function(atoms, values, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize/20.*n,dim=None,delta='simple',norm=True)
#
#        SCF-=SCF.min()
#        SCF2-=SCF2.min()
#        SCF3-=SCF3.min()
#
#        self.assertAlmostEqual(np.isfinite(SCF/SCF).sum(), np.floor(length_cutoff))
#        self.assertAlmostEqual(np.isfinite(SCF2/SCF2).sum(), int(np.floor(FFT_cutoff)+np.ceil(np.ceil(length_cutoff-FFT_cutoff)*50./n)))
#        self.assertAlmostEqual(np.isfinite(SCF3/SCF3).sum(), int(np.floor(FFT_cutoff)+np.ceil(np.ceil(length_cutoff-FFT_cutoff)*20./n)))

    def test_directional_spacing(self):
        n=30

        xyz=np.zeros((n**3,3))
        m=np.meshgrid(np.arange(0,n,1),np.arange(0,2*n,2),np.arange(0,3*n,3))
        xyz[:,0]=m[0].reshape((-1))
        xyz[:,1]=m[0].reshape((-1))
        xyz[:,2]=m[2].reshape((-1))

        values=np.random.rand(len(xyz))
        atoms=Atoms(positions=xyz, cell=np.array([[n,0,0],[0,2*n,0],[0,0,3*n]]))

        length_cutoff= n
        output_gridsize= 0.1
        FFT_cutoff= 0.
        approx_FFT_gridsize= 1.

        SCF0, r0=spatial_correlation_function(atoms, values, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize, dim=0, delta='simple', norm=True)
        SCF1, r1=spatial_correlation_function(atoms, values, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize, dim=1, delta='simple', norm=True)
        SCF2, r2=spatial_correlation_function(atoms, values, length_cutoff, output_gridsize, FFT_cutoff, approx_FFT_gridsize, dim=2, delta='simple', norm=True)

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
