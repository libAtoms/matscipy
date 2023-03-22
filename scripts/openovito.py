#
# Copyright 2015 James Kermode (Warwick U.)
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
import sys

sys.stdout = open('stdout.txt', 'w')
sys.stderr = open('stderr.txt', 'w')

import numpy as np

from ovito import *
from ovito.data import *
from ovito.modifiers import *

import ase2ovito

from ase.io import read, write

# Read ASE Atoms instance from disk
atoms = read(sys.argv[1])

# add some comuted properties
atoms.new_array('real', (atoms.positions**2).sum(axis=1))
atoms.new_array('int', np.array([-1]*len(atoms)))

# convert from Atoms to DataCollection
data = DataCollection.create_from_atoms(atoms)


# Create a node and insert it into the scene
node = ObjectNode()
node.source = data
dataset.scene_nodes.append(node)

# add some bonds
bonds = ase2ovito.neighbours_to_bonds(atoms, 2.5)
print 'Bonds:'
print bonds.array
data.add(bonds)

# alternatively, ask Ovito to create the bonds
#node.modifiers.append(CreateBondsModifier(cutoff=2.5))

new_data = node.compute()

#print new_data.bonds.array

# Select the new node and adjust viewport cameras to show everything.
dataset.selected_node = node
for vp in dataset.viewports:
    vp.zoom_all()
    
# Do the reverse conversion, after pipeline has been applied
atoms = new_data.to_atoms()

# Dump results to disk
atoms.write('dump.extxyz')
