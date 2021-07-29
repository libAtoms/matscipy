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
import numpy as np

from ovito.data import (DataCollection,
                        SimulationCell,
                        ParticleProperty,
                        ParticleType,
                        Bonds)

from ase.atoms import Atoms

from matscipy.neighbours import neighbour_list

def datacollection_to_atoms(self):
    """
    Convert from ovito.data.DataCollection to ase.atoms.Atoms
    """

    # Extract basic dat: pbc, cell, positions, particle types
    pbc = self.cell.pbc
    cell_matrix = np.array(self.cell.matrix)    
    cell, origin = cell_matrix[:, :3], cell_matrix[:, 3]
    info = {'cell_origin': origin }
    positions = np.array(self.position)
    type_names = dict([(t.id, t.name) for t in
                       self.particle_type.type_list])
    symbols = [type_names[id] for id in np.array(self.particle_type)]

    # construct ase.Atoms object
    atoms = Atoms(symbols,
                  positions,
                  cell=cell,
                  pbc=pbc,
                  info=info)

    # Convert any other particle properties to additional arrays
    for name, prop in self.iteritems():
        if name in ['Simulation cell',
                    'Position',
                    'Particle Type']:
            continue
        if not isinstance(prop, ParticleProperty):
            continue
        atoms.new_array(prop.name, prop.array)
    
    return atoms

DataCollection.to_atoms = datacollection_to_atoms


def datacollection_create_from_atoms(cls, atoms):
    """
    Convert from ase.atoms.Atoms to ovito.data.DataCollection
    """
    data = cls()

    # Set the unit cell and origin (if specified in atoms.info)
    cell = SimulationCell()
    matrix = np.zeros((3,4))
    matrix[:, :3] = atoms.get_cell()
    matrix[:, 3]  = atoms.info.get('cell_origin',
                                   [0., 0., 0.])
    cell.matrix = matrix
    cell.pbc = [bool(p) for p in atoms.get_pbc()]
    data.add(cell)

    # Add ParticleProperty from atomic positions
    num_particles = len(atoms)                                  
    position = ParticleProperty.create(ParticleProperty.Type.Position,
                                       num_particles)
    position.mutable_array[...] = atoms.get_positions()
    data.add(position)

    # Set particle types from chemical symbols
    types = ParticleProperty.create(ParticleProperty.Type.ParticleType,
                                    num_particles)
    symbols = atoms.get_chemical_symbols()
    type_list = list(set(symbols))
    for i, sym in enumerate(type_list):
        types.type_list.append(ParticleType(id=i+1, name=sym))
    types.mutable_array[:] = [ type_list.index(sym)+1 for sym in symbols ]
    data.add(types)

    # Add other properties from atoms.arrays
    for name, array in atoms.arrays.iteritems():
        if name in ['positions', 'numbers']:
            continue
        if array.dtype.kind == 'i':
            typ = 'int'
        elif array.dtype.kind == 'f':
            typ = 'float'
        else:
            continue
        num_particles = array.shape[0]
        num_components = 1
        if len(array.shape) == 2:
            num_components = array.shape[1]
        prop = ParticleProperty.create_user(name,
                                            typ,
                                            num_particles,
                                            num_components)
        prop.mutable_array[...] = array
        data.add(prop)
    
    return data

DataCollection.create_from_atoms = classmethod(datacollection_create_from_atoms)

def neighbours_to_bonds(atoms, cutoff):
    i, j = neighbour_list('ij', atoms, cutoff)
    bonds = Bonds()
    for ii, jj in zip(i, j):
        bonds.add_full(int(ii), int(jj))
    return bonds
    
