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

CP = Checkpoint('checkpoints.db')

Checkpointed code block, try ... except notation:

try:
    a, C, C_err = CP.load()
except NoCheckpoint:
    C, C_err = fit_elastic_constants(a)
    CP.save(a, C, C_err)

Checkpoint code block, shorthand notation:

C, C_err = CP(fit_elastic_constants)(a)

Example for checkpointing within an iterative loop, e.g. for searching crack
tip position:

try:
    a, converged, tip_x, tip_y = CP.load()
except NoCheckpoint:
    converged = False
    tip_x = tip_x0
    tip_y = tip_y0
while not converged:
    ... do something to find better crack tip position ...
    converged = ...
    CP.save(a, converged, tip_x, tip_y)

"""

import os

import ase
from ase.db import connect

from matscipy.logger import quiet

###

class NoCheckpoint(Exception):
    pass

class Checkpoint(object):
    _value_prefix = '_values_'

    def __init__(self, db='checkpoints.db', logger=quiet):
        self.db = db
        self.logger = logger

        self.checkpoint_id = 0

    def __call__(self, func, *args, **kwargs):
        checkpoint_func_name = str(func)
        def decorated_func(*args, **kwargs):
            # Get the first ase.Atoms object.
            atoms = None
            for a in args:
                if atoms is None and isinstance(a, ase.Atoms):
                    atoms = a

            try:
                retvals = self.load(atoms=atoms)
            except NoCheckpoint:
                retvals = func(*args, **kwargs)
                self.save(retvals, atoms=atoms,
                          checkpoint_func_name=checkpoint_func_name)
            return retvals
        return decorated_func

    def _mangled_checkpoint_id(self):
        return self.checkpoint_id

    def load(self, atoms=None):
        self.checkpoint_id += 1

        retvals = []
        with connect(self.db) as db:
            try:
                dbentry = db.get(checkpoint_id=self._mangled_checkpoint_id())
            except KeyError:
                raise NoCheckpoint

            atomsi = dbentry.checkpoint_atoms_args_index
            i = 0
            while i == atomsi or \
                hasattr(dbentry, '{}{}'.format(self._value_prefix, i)):
                if i == atomsi:
                    newatoms = dbentry.toatoms()
                    if atoms is not None:
                        # Assign calculator
                        newatoms.set_calculator(atoms.get_calculator())
                    retvals += [dbentry.toatoms()]
                else:
                    retvals += [db.entry['{}{}'.format(self._value_prefix, i)]]
                i += 1

        self.logger.pr('Successfully restored checkpoint '
                       '{}.'.format(self.checkpoint_id))
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    def save(self, *args, **kwargs):
        d = {'{}{}'.format(self._value_prefix, i): v
             for i, v in enumerate(args)}

        try:
            atomsi = [isinstance(v, ase.Atoms) for v in args].index(True)
            atoms = args[atomsi]
            del d['{}{}'.format(self._value_prefix, atomsi)]
        except ValueError:
            atomsi = -1
            try:
                atoms = kwargs['atoms']
            except:
                raise RuntimeError('No atoms object provided in arguments.')

        try:
            del kwargs['atoms']
        except:
            pass

        d['checkpoint_id'] = self._mangled_checkpoint_id()
        d['checkpoint_atoms_args_index'] = atomsi
        d.update(kwargs)

        with connect(self.db) as db:
            db.write(atoms, **d)
