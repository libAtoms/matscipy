#! /usr/bin/env python

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

import glob
import sys

import numpy as np
from scipy.integrate import cumtrapz

import ase.io
from ase.data import atomic_numbers

###

sys.path += [ "." ]
import params

###

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

bond1, bond2 = params.bond

bond_lengths = []

fns = glob.glob('step_*.cfg')
for fn in fns:
    a = ase.io.read(fn)
    del a[np.logical_or(a.numbers == atomic_numbers[ACTUAL_CRACK_TIP],
                        a.numbers == atomic_numbers[FITTED_CRACK_TIP])]

    # Bond length.
    dr = a[bond1].position - a[bond2].position
    bond_lengths += [ np.linalg.norm(dr) ]

###

indices = np.argsort(bond_lengths)

epot_cluster = []
bond_lengths = []
bond_forces = []
work = []

last_a = None
for fn in np.array(fns)[indices]:
    a = ase.io.read(fn)
    del a[np.logical_or(a.numbers == atomic_numbers[ACTUAL_CRACK_TIP],
                        a.numbers == atomic_numbers[FITTED_CRACK_TIP])]

    # Bond length.
    dr = a[bond1].position - a[bond2].position
    bond_lengths += [ np.linalg.norm(dr) ]

    # Groups
    g = a.get_array('groups')

    # Get potential energy.
    a.set_calculator(params.calc)
    epot_cluster += [ a.get_potential_energy() ]

    # Forces on bond.
    forces = a.get_array('forces')
    df = forces[bond1, :] - forces[bond2, :]
    bond_forces += [ 0.5 * np.dot(df, dr)/np.sqrt(np.dot(dr, dr)) ]

    # Work due to moving boundary.
    if last_a is None:
        work += [ 0.0 ]
    else:
        last_forces = last_a.get_array('forces')
        # This is the trapezoidal rule.
        work += [ np.sum(0.5 * (forces[g==0,:]+last_forces[g==0,:]) *
                         (a.positions[g==0,:]-last_a.positions[g==0,:])
                          ) ]

    last_a = a

# Sort according to bond length.
epot_cluster = np.array(epot_cluster)-epot_cluster[0]
work = np.cumsum(work)

# Integrate true potential energy.
epot = -cumtrapz(bond_forces, bond_lengths, initial=0.0)
np.savetxt('crack_eval.out', np.transpose([bond_lengths,
                                           bond_forces,
                                           epot,
                                           epot_cluster,
                                           work]))
