#
# Copyright 2017 Lars Pastewka (U. Freiburg)
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
from matscipy.neighbours import mic

###

class RemoveDrift:
    """
    Remove drift of the center of mass motion of an atomic system that may
    occur in periodic simulations.
    """

    def __init__(self, traj):
        self.traj = traj

        # Stores the global displacement vectors (shift vectors) required to
        # move all positions back to original.
        self.shifts = [np.zeros(3)]

    def _fill_shifts_upto(self, i):
        # Iterate up to frame i the full trajectory first and generate a list
        # of displacement vectors.
        while len(self.shifts) <= i:
            j = len(self.shifts)
            a0 = self.traj[j-1]
            a1 = self.traj[j]
            s0 = a0.get_scaled_positions()%1.0
            s1 = a1.get_scaled_positions()%1.0
            sdisps = mic(s1-s0, np.eye(3), pbc=a0.pbc)
            self.shifts += [self.shifts[-1]+sdisps.mean(axis=0)]

    def __getitem__(self, i=-1):
        if i < 0:
            i = len(self) + i
            if i < 0 or i >= len(self):
                raise IndexError('Trajectory index out of range.')

        self._fill_shifts_upto(i)

        a = self.traj[i]
        a.set_scaled_positions(a.get_scaled_positions()%1.0-self.shifts[i])

        return a


    def __len__(self):
        return len(self.traj)

