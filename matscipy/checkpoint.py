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
Checkpointing functionality.

Initialize checkpoint object:

cp = Checkpoint('checkpoints.db')

Checkpointed code block, try ... except notation:

try:
    a, C, C_err = cp.load()
except NoCheckpoint:
    C, C_err = fit_elastic_constants(a)
    cp.save(a, C, C_err)

Checkpoint code block, shorthand notation:

a, C, C_err = cp(fit_elastic_constants, a)

Example for checkpointing within an iterative loop, e.g. for searching crack
tip position:

try:
    a, converged, tip_x, tip_y = cp.load()
except NoCheckpoint:
    converged = False
    tip_x = tip_x0
    tip_y = tip_y0
while not converged:
    ... do something to find better crack tip position ...
    converged = ...
    cp.save(a, converged, tip_x, tip_y)

"""

from __future__ import print_function

import os

import ase
from ase.db import connect

###

class NoCheckpoint(Exception):
    pass

class Checkpoint(object):
    _value_prefix = '_values_'

    def __init__(self, db='checkpoints.db'):
        self.db = db
        self.checkpoint_id = 0

    def __call__(self, func, *args, **kwargs):
        try:
            retvals = self.load()

            # Get the calculator object of the first parameter that is passed
            # to the function.
            calc = None
            for a in args:
                if calc is None:
                    try:
                        calc = a.get_calculator()
                    except:
                        pass

            # Assign the calculator to any ase.Atoms just loaded from the
            # checkpoint.
            for r in retvals:
                try:
                    r.set_calculator(calc)
                except:
                    pass
        except NoCheckpoint:
            retvals = func(*args, **kwargs)
            self.save(retvals)
        return retvals

    def _mangled_checkpoint_id(self):
        return self.checkpoint_id

    def load(self, atoms=None):
        self.checkpoint_id += 1

        retvals = []
        with connect(self.db) as db:
            try:
                print('Try loading:', self._mangled_checkpoint_id())
                dbentry = db.get(checkpoint_id=self._mangled_checkpoint_id())
            except KeyError:
                raise NoCheckpoint

            i = 0
            while hasattr(dbentry, '{}{}'.format(self._value_prefix, i)):
                retvals += [db.entry['{}{}'.format(self._value_prefix, i)]]
                i += 1
        if len(retvals) == 0:
            return dbentry.toatoms()
        else:
            return tuple([dbentry.toatoms()]+retvals)

    def save(self, atoms, *args):
        print('Saving:', self._mangled_checkpoint_id())
        if not isinstance(atoms, ase.Atoms):
            raise TypeError('First argument must be an ase.Atoms object.')

        d = {'{}{}'.format(self._value_prefix, i): v
             for i, v in enumerate(args)}
        d['checkpoint_id'] = self._mangled_checkpoint_id()


        with connect(self.db) as db:
            db.write(atoms, **d)