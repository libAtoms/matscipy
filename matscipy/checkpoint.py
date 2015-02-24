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
"""

import os

import ase.io as io
from ase.parallel import parprint

###

_checkpoint_num = 0
_read_args = {}
_write_args = {}

###

def set_read_args(**read_args):
    global _read_args
    _read_args = read_args

def set_write_args(**write_args):
    global _write_args
    _write_args = write_args

def reset():
    global _checkpoint_num
    _checkpoint_num = 0
    set_read_args()
    set_write_args()

###

def open_extxyz(fn):
    """
    Open an extended XYZ file as a checkpointing database.
    """

    set_write_args(format='extxyz')
    set_read_args(format='extxyz', index=0)

    f = open(fn, 'a+')
    f.seek(0)
    return f

###

def checkpoint(f, func, atoms, *args, **kwargs):
    """
    Call function only if checkpoint file does not exist, otherwise read atoms
    object from checkpoint file. This allows to have multiple consecutive
    operations in a single script and restart from the latest one that
    completed.

    Example:

    a = checkpoint('geometry_optimized.traj', optimize_geometry, a, fmax=0.01)
    a = checkpoint('charge_computed.xyz', compute_charge, a)

    f = open_extxyz('checkpoints.xyz')
    a = checkpoint(f, optimize_geometry, a, fmax=0.01)
    a = checkpoint(f, compute_charge, a)
    """
    global _checkpoint_num, _read_args, _write_args

    if isinstance(f, str):
        f = '{:04}_{}'.format(_checkpoint_num+1, f)

    try:
        newatoms = io.read(f, **_read_args)
        parprint('Successfully read configuration {:04} from checkpoint '
                 'file...'.format(_checkpoint_num+1))

        newatoms.set_calculator(atoms.get_calculator())
    except IOError:
        newatoms = func(atoms, *args, **kwargs)
        io.write(f, atoms, **_write_args)
        try:
            f.flush()
        except:
            pass
        parprint('Successfully wrote configuration {:04} to checkpoint '
                 'file...'.format(_checkpoint_num+1))

    _checkpoint_num += 1
    return newatoms
