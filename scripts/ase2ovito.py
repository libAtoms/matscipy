#!/usr/bin/env ovitos -g 

from __future__ import print_function

import sys
import code
import tempfile
import os
import atexit

import numpy as np

import ovito
from ovito.io import import_file
from ovito.data import (DataCollection,
                        SimulationCell,
                        ParticleProperty,
                        ParticleTypeProperty)

import Particles

from ase.io import read, write
from ase.atoms import Atoms

def data_to_atoms(data):
    """
    Convert from ovito.data.DataCollection to ase.atoms.Atoms
    """
    pbc = data.cell.pbc
    cell_matrix = np.array(data.cell.matrix)    
    cell, origin = cell_matrix[:, :3], cell_matrix[:, 3]

    info = {}
    info['cell_origin'] = origin

    positions = np.array(data.position)
    type_names = dict([(t.id, t.name) for t in
                       data.particle_type.type_list])
    symbols = [type_names[id] for id in np.array(data.particle_type)]

    atoms = Atoms(symbols,
                  positions,
                  cell=cell,
                  pbc=pbc,
                  info=info)
    return atoms

def atoms_to_data(atoms):
    data = DataCollection()
    cell_matrix = np.zeros((3,4))
    cell_matrix[:, :3] = atoms.get_cell()
    cell_matrix[:, 3] = atoms.info.get('cell_origin',
                                       [0., 0., 0.])
    cell = SimulationCell(matrix=cell_matrix,
                          pbc=atoms.get_pbc())
    data.addObject(cell)
                              
    position = ParticleProperty(name='Position',
                                type=Particles.ParticleProperty.Type.Position,
                                array=atoms.get_positions())
    data.addObject(position)
    
    return data

node = import_file(sys.argv[1])
data = node.compute()
atoms = data_to_atoms(data)

atoms.positions[...] *= 1.1 # would be something more complex

node.remove_from_scene()

#new_data = atoms_to_data(atoms)

# make new_node from new_data and add to scence
    
