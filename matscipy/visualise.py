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

"""
Interface from ASE to the chemview Jupyter visualiser.

Your Jupyter notebook will need to contain

from chemview import enable_notebook
enable_notebook()
"""

###

import itertools

import numpy as np

from ase.data import covalent_radii
from matscipy.neighbours import neighbour_list

from chemview import MolecularViewer

###

def view(a, scale=1.2):
    n = a.numbers
    maxn = n.max()
    cutoffs = np.zeros([maxn+1, maxn+1])

    for n1, n2 in itertools.product(n, n):
        cutoffs[n1, n2] = scale*(covalent_radii[n1]+covalent_radii[n2])

    # Construct a bond list
    i, j, S = neighbour_list('ijS', a, cutoffs, np.array(a.numbers, dtype=np.int32))
    m = np.logical_and(i<j, (S==0).all(axis=1))
    i = i[m]
    j = j[m]

    mv = MolecularViewer(a.positions/10, topology=dict(atom_types=a.get_chemical_symbols(),
                                                       bonds=[(x, y) for x, y in zip(i, j)]))
    mv.ball_and_sticks()
    return mv
