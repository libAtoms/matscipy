#
# Copyright 2014 James Kermode (Warwick U.)
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

import os
import sys

import numpy as np

import ase
import ase.constraints
import ase.io
import ase.optimize

from ase.data import atomic_numbers
from ase.parallel import parprint

###

sys.path += [ "." ]
import params

###

cryst = params.cryst.copy()
cryst.set_calculator(params.calc)
cryst.info['energy'] = cryst.get_potential_energy()
ase.io.write('cryst.xyz', cryst, format='extxyz')

# The bond we will open up
bond1, bond2 = params.bond

# Run strain calculation.
for i, separation in enumerate(params.separations):
    parprint('=== separation = {0} ==='.format(separation))
    xyz_file = 'separation_%04d.xyz' % int(separation*1000)
    if os.path.exists(xyz_file):
        parprint('%s found, skipping' % xyz_file)
        a = ase.io.read(xyz_file)
        a.set_calculator(params.calc)
    else:
        a = cryst.copy()
        cell = a.cell.copy()
        cell[1,1] += separation
        a.set_cell(cell, scale_atoms=False)
        
        # Assign calculator.
        a.set_calculator(params.calc)
        a.set_constraint(None)
        a.info['separation'] = separation
        a.info['bond_length'] = a.get_distance(bond1, bond2, mic=True)
        a.info['energy'] = a.get_potential_energy()
        a.info['cell_origin'] = [0, 0, 0]
        ase.io.write(xyz_file, a, format='extxyz')

# Relax the final surface configuration
parprint('Optimizing final positions...')
opt = ase.optimize.FIRE(a, logfile=None)
opt.run(fmax=params.fmax)
parprint('...done. Converged within {0} steps.' \
    .format(opt.get_number_of_steps()))

a.set_constraint(None)
a.set_array('forces', a.get_forces())
a.info['separation'] = separation
a.info['energy'] = a.get_potential_energy()
a.info['cell_origin'] = [0, 0, 0]
a.info['bond_length'] = a.get_distance(bond1, bond2, mic=True)
        
ase.io.write('surface.xyz', a, format='extxyz')
