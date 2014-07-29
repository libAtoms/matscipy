# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) Lars Pastewka, Karlsruhe Institute of Technology
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

import numpy as np

from ase.lattice.cubic import Diamond

###

def diamond_110_110(el, a0, n, crack_surface=[1,1,0],
                    crack_front=[1,-1,0],
                    skin_x=0.5, skin_y=1.0,
                    central_x=-1.0, central_y=-1.0,
                    vac=5.0):
    nx, ny, nz = n
    third_dir = np.cross(crack_surface, crack_front)
    a = Diamond(el,
            latticeconstant = a0,
            size = [nx, ny, nz], 
            directions = [third_dir, crack_surface, crack_front]
            )
    sx, sy, sz = a.get_cell().diagonal()
    a.translate([sx/(8*nx), sy/(4*ny), sz/(4*nz)])
    a.set_scaled_positions(a.get_scaled_positions())

    skin_x = skin_x*sx/nx
    skin_y = skin_y*sy/ny
    r = a.get_positions()
    g = np.where(
        np.logical_or(
            np.logical_or(
                np.logical_or(
                    r[:, 0] < skin_x, r[:, 0] > sx-skin_x),
                r[:, 1] < skin_y),
            r[:, 1] > sy-skin_y),
        np.zeros(len(a), dtype=int),
        np.ones(len(a), dtype=int))

    g = np.where(
        np.logical_or(
            np.logical_or(
                np.logical_or(
                    r[:, 0] < sx/2-central_x, r[:, 0] > sx/2+central_x),
                r[:, 1] < sy/2-central_y),
            r[:, 1] > sy/2+central_y),
        g,
        2*np.ones(len(a), dtype=int))
    a.set_array('groups', g)

    a.set_cell([sx+2*vac, sy+2*vac, sz])
    a.translate([vac, vac, 0.0])
    a.set_pbc([False, False, True])

    return a

###

def diamond_110_001(el, a0, n, crack_surface=[1,1,0], crack_front=[0,0,1],
                    skin_x=1.0, skin_y=1.0, vac=5.0):
    nx, ny, nz = n
    third_dir = np.cross(crack_surface, crack_front)
    directions = [ third_dir, crack_front, crack_surface ]
    if np.linalg.det(directions) < 0:
        third_dir = -third_dir
    directions = [ third_dir, crack_surface, crack_front ]
    a = Diamond(el, latticeconstant = a0, size = [ nx,ny,nz ], 
                directions = directions)
    sx, sy, sz = a.get_cell().diagonal()
    a.translate([ sx/(4*nx), sy/(8*ny), sz/(4*nz) ])
    a.set_scaled_positions(a.get_scaled_positions())

    lx  = skin_x*sx/nx
    ly  = skin_y*sy/nz
    r   = a.get_positions()
    g   = np.where(
        np.logical_or(
            np.logical_or(
                np.logical_or(
                    r[:, 0] < lx, r[:, 0] > sx-lx),
                r[:, 1] < ly),
            r[:, 1] > sy-ly),
        np.zeros(len(a), dtype=int),
        np.ones(len(a), dtype=int))
    a.set_array('groups', g)

    a.set_cell([sx+2*vac, sy+2*vac, sz])
    a.translate([vac, vac, 0.0])
    a.set_pbc([False, False, True])

    return a
