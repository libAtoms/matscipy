#
# Copyright 2016, 2021 Lars Pastewka (U. Freiburg)
#           2016, 2018 Richard Jana (KIT & U. Freiburg)
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

'''
Compute the spatial correlation of a given function. Distances larger
than a cut-off are calculated by fourier transformation, while shorter
distances are calculated directly.
coords.shape=(n_atoms,3)
cell_vectors=[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]

options:
#1)dimensions of correlation [dim=]
    a: along all 3 dimensions -> total distance: None (default)
    b: along only one dimension, ignoring other components: 0..2
#2)mode to assign the atomic values to the FFT grid points [delta=]
    a: assign value to the nearest grid point: simple (default)
    b: assign value to the 8 nearest grid points, distributed
        proportional to their distance: else
#3)nomalisation by variance of values [norm=]
    a: off: False (default)
    b: on: True
'''

import numpy as np
from math import floor, ceil
from matscipy.neighbours import neighbour_list
from ase import Atoms

def betrag(vec):
    return np.sqrt((vec**2).sum())
def max_rad(cell_vectors):
    #calculate length cutoff from cell vectors
    x = cell_vectors[:,0]
    y = cell_vectors[:,1]
    z = cell_vectors[:,2]
    r = np.zeros(3)
    nor = np.zeros(3)

    nor = np.cross(y,z)
    r[0] = np.abs((x*nor).sum()/betrag(x)/betrag(nor))*betrag(x)/2.0
    nor = np.cross(z,x)
    r[1] = np.abs((y*nor).sum()/betrag(y)/betrag(nor))*betrag(y)/2.0
    nor = np.cross(x,y)
    r[2] = np.abs((z*nor).sum()/betrag(z)/betrag(nor))*betrag(z)/2.0
    
    return r.min()


def spatial_correlation_function(atoms, values, length_cutoff=None,
                                 output_gridsize=None, FFT_cutoff=None,
                                 approx_FFT_gridsize=None, dim=None,
                                 delta='simple', norm=False):
    # Make sure values are floats
    values = np.asarray(values, dtype=float)

    xyz = atoms.get_positions()
    abc = atoms.get_scaled_positions() % 1.0
    cell_vectors = atoms.cell.T
    n_atoms = len(xyz)

    if length_cutoff is None:
        length_cutoff = np.floor(max_rad(cell_vectors))

    if FFT_cutoff is None:
        FFT_cutoff = 7.5

    if output_gridsize is None:
        output_gridsize = 0.1

    if approx_FFT_gridsize is None:
        approx_FFT_gridsize = 1.0

    n_lattice_points = np.array(np.ceil(cell_vectors.diagonal()
                                        /approx_FFT_gridsize),
                                dtype=int)
    FFT_gridsize = cell_vectors.diagonal()/n_lattice_points

    if delta == 'simple':
        # calc lattice values (add to nearest lattice point)
        Q = np.zeros(shape=(n_lattice_points))
        for _abc, _q in zip(abc, values):
            x,y,z = np.array(_abc*n_lattice_points, dtype=int) \
                             %n_lattice_points
            Q[x,y,z] += _q
    else:
        # proportional distribution on 8 neightbor points
        Q = np.zeros(shape=(n_lattice_points))
        a1, a2, a3 = cell_vectors.T
        for _abc, _q in zip(abc, q):
            x,y,z = _abc*(n_lattice_points-1)
            aes = np.array([np.floor(x),np.ceil(x)] \
                          ).reshape(-1, 1, 1, 1)/(n_lattice_points[0]-1)
            bes = np.array([np.floor(y),np.ceil(y)] \
                          ).reshape( 1,-1, 1, 1)/(n_lattice_points[1]-1)
            ces = np.array([np.floor(z),np.ceil(z)] \
                          ).reshape( 1, 1,-1, 1)/(n_lattice_points[2]-1)
            octo = (aes*a1.reshape(1,1,1,-1) \
                    +bes*a2.reshape(1,1,1,-1) \
                    +ces*a3.reshape(1,1,1,-1)) \
                    -cartesianN(_abc,cell_vectors).reshape(1,1,1,-1)
            octo = 1./(np.sqrt((octo**2).sum(axis=3)))
            Q[np.floor(x):np.ceil(x)+1,np.floor(y):np.ceil(y)+1,
              np.floor(z):np.ceil(z)+1] += octo/octo.sum()*_q

    # FFT
    Q_schlange = np.fft.fftn(Q)
    C_schlange = Q_schlange*Q_schlange.conjugate()
    C = np.fft.ifftn(C_schlange)*n_lattice_points.prod() \
        /n_atoms/n_atoms
    C = np.fft.ifftshift(C)

    if dim is None:
        # distance mapping (for floor/ceil convention see *i*fftshift
        # definition)
        a = np.reshape(np.arange(-floor(n_lattice_points[0]/2.),
                       ceil(n_lattice_points[0]/2.),1)
                       /n_lattice_points[0],(-1, 1, 1, 1))
        b = np.reshape(np.arange(-floor(n_lattice_points[1]/2.),
                       ceil(n_lattice_points[1]/2.),1)
                       /n_lattice_points[1],( 1,-1, 1, 1))
        c = np.reshape(np.arange(-floor(n_lattice_points[2]/2.),
                       ceil(n_lattice_points[2]/2.),1)
                       /n_lattice_points[2],( 1, 1,-1, 1))
        a1, a2, a3 = cell_vectors.T

        r = a*a1.reshape(1,1,1,-1)+b*a2.reshape(1,1,1,-1) \
            +c*a3.reshape(1,1,1,-1)
        dist = np.sqrt((r**2).sum(axis=3))
    elif 0 <= dim <3:
        # directional SCFs
        # for floor/ceil convention see *i*fftshift definition
        a = np.reshape(np.arange(-floor(n_lattice_points[0]/2.),
                                 ceil(n_lattice_points[0]/2.),1)
                       /n_lattice_points[0],(-1, 1, 1, 1))
        b = np.reshape(np.arange(-floor(n_lattice_points[1]/2.),
                                 ceil(n_lattice_points[1]/2.),1)
                       /n_lattice_points[1],( 1,-1, 1, 1))
        c = np.reshape(np.arange(-floor(n_lattice_points[2]/2.),
                                 ceil(n_lattice_points[2]/2.),1)
                       /n_lattice_points[2],( 1, 1,-1, 1))
        a1, a2, a3 = cell_vectors.T
        r = a*a1.reshape(1,1,1,-1) +b*a2.reshape(1,1,1,-1) \
            +c*a3.reshape(1,1,1,-1)
        dist = np.abs(r[:,:,:,dim])  # use indices to access directions
    else:
        print('invalid correlation direction: '+str(dim))
        sys.exit()

    nbins = int(length_cutoff/output_gridsize)
    bins = np.arange(0, length_cutoff+length_cutoff/nbins,
                     length_cutoff/nbins)
    SCF, edges = np.histogram(np.ravel(dist), bins=bins,
                              weights=np.ravel(np.real(C)))
    n, edges = np.histogram(np.reshape(dist,(-1,1)), bins=bins)
    n[n==0] = 1
    SCF /= n
    # Alternative to the above three lines:
    # SCF *= atoms.get_volume()/np.prod(n_lattice_points) / slice_volume
    if norm:
        v_2_mean = (values**2).mean()
        v_mean_2 = (values.mean())**2
        SCF = (SCF-v_mean_2)/(v_2_mean-v_mean_2)

    return SCF, (edges[1:]+edges[:-1])/2



def spatial_correlation_function_near(atoms, values, gridsize=None,
                                      cutoff=None, norm=False):
    if gridsize is None:
        gridsize = 0.1

    if cutoff is None:
        cutoff = 7.5

    # close range exact calculation
    nbins = int(cutoff/gridsize)+1
    index1,index2,dist = neighbour_list('ijd', atoms, cutoff=cutoff)
    SCF_near, edges = np.histogram(dist, bins=bins,
                                   weights=values[index1]
                                   *values[index2])
    slice_volume = 4*np.pi/3 * (edges[1:]**3-edges[:-1]**3)
    SCF_near *= atoms.get_volume()/n_atoms**2 / slice_volume
    if norm:
        v_2_mean = (values**2).mean()
        v_mean_2 = (values.mean())**2
        SCF_near = (SCF_near-v_mean_2)/(v_2_mean-v_mean_2)

    return SCF_near, (edges[1:]+edges[:-1])/2
