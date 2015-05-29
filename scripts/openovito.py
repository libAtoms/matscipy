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
