#
# Copyright 2015 Lars Pastewka (U. Freiburg)
#           2015 Till Junge (EPFL)
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
"""
@file   DistributedServer.py

@author Till Junge <till.junge@kit.edu>

@date   19 Mar 2015

@brief  example for using the  multiprocessing capabilities of PyPyContact ported for matscipy,
        serverside

@section LICENCE

 Copyright (C) 2015 Till Junge

DistributedServer.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

DistributedServer.py is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from matscipy import BaseResultManager
import numpy as np

class Manager(BaseResultManager):
    def __init__(self, port, key, resolution):
        super(Manager, self).__init__(port, key.encode())

        self.resolution = resolution
        center_coord = (resolution[0]//2, resolution[1]//2)
        initial_guess = 1

        self.available_jobs = dict({center_coord: initial_guess})
        self.scheduled = set()
        self.done_jobs = set()
        self.set_todo_counter(np.prod(resolution))
        self.result_matrix = np.zeros(resolution)

    def schedule_available_jobs(self):
        for coords, init_guess in self.available_jobs.items():
            dummy_offset = 1
            self.job_queue.put(((init_guess, dummy_offset), coords))
            self.scheduled.add(coords)
        self.available_jobs.clear()

    def mark_ready(self, i, j, initial_guess):
        if (i, j) not in self.available_jobs.keys() and (i, j) not in self.scheduled:
            self.available_jobs[(i, j)] = initial_guess

    def process(self, value, coords):
        i, j = coords
        self.result_matrix[i, j] = value
        self.decrement_todo_counter()
        print("got solution to job '{}', {} left to do".format(
            (i, j), self.get_todo_counter()))
        self.done_jobs.add((i, j))
        if self.get_todo_counter() < 10:
            print("Missing jobs: {}".format(self.scheduled-self.done_jobs))
        #tag neighbours as available
        if i > 0:
            self.mark_ready(i-1, j, value)
        if j > 0:
            self.mark_ready(i, j-1, value)
        if i < self.resolution[0]-1:
            self.mark_ready(i+1, j, value)
        if j < self.resolution[1]-1:
            self.mark_ready(i, j+1, value)

def parse_args():
    parser = Manager.get_arg_parser()
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    manager = Manager(args.port, args.auth_token, (12, 12))
    manager.run()


if __name__ == "__main__":
    main()
