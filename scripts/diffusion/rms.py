#
# Copyright 2019, 2021 Lars Pastewka (U. Freiburg)
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

import numpy as np

from ase.data import chemical_symbols
from ase.io import NetCDFTrajectory
from matscipy.neighbours import mic

###

traj = NetCDFTrajectory(sys.argv[1])

ref_frame = traj[0]

individual_elements = set(ref_frame.numbers)
s = '#{:>9s}'.format('frame')
if 'time' in ref_frame.info:
    s = '{} {:>10s}'.format(s, 'time')
s = '{} {:>10s}'.format(s, 'tot. rms')
for element in individual_elements:
    s = '{} {:>10s}'.format(s, 'rms ({})'.format(chemical_symbols[element]))
print(s)

last_frame = traj[0]
displacements = np.zeros_like(ref_frame.positions)
for i, frame in enumerate(traj[1:]):
    last_frame.set_cell(frame.cell, scale_atoms=True)
    cur_displacements = frame.positions - last_frame.positions
    cur_displacements = mic(cur_displacements, frame.cell, frame.pbc)
    displacements += cur_displacements

    s = '{:10}'.format(i+1)
    if 'time' in frame.info:
        s = '{} {:10.6}'.format(s, frame.info['time'])
    s = '{} {:10.6}'.format(s, np.sqrt((displacements**2).mean()))
    for element in individual_elements:
        s = '{} {:10.6}'.format(s, np.sqrt(
            (displacements[frame.numbers == element]**2).mean()))

    print(s)

    last_frame = frame
