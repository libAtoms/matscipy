#
# Copyright 2014-2015, 2021 Lars Pastewka (U. Freiburg)
#           2017 Punit Patel (Warwick U.)
#           2014-2016 James Kermode (Warwick U.)
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

# USAGE:
#
# Code imports the file 'params.py' from current working directory. params.py
# contains simulation parameters. Some parameters can be omitted, see below.

from math import sqrt

import os
import sys

import numpy as np

import ase
from ase.constraints import FixConstraint
import ase.io
import ase.optimize
from ase.data import atomic_numbers
from ase.units import GPa
from ase.geometry import find_mic

import matscipy.fracture_mechanics.crack as crack
from matscipy import parameter
from matscipy.logger import screen

from setup_crack import setup_crack

###

import sys
sys.path += ['.', '..']
import params

class FixBondLength(FixConstraint):
    """Constraint object for fixing a bond length."""

    removed_dof = 1

    def __init__(self, a1, a2):
        """Fix distance between atoms with indices a1 and a2. If mic is
        True, follows the minimum image convention to keep constant the
        shortest distance between a1 and a2 in any periodic direction.
        atoms only needs to be supplied if mic=True.
        """
        self.indices = [a1, a2]
        self.constraint_force = None

    def adjust_positions(self, atoms, new):
        p1, p2 = atoms.positions[self.indices]
        d, p = find_mic(np.array([p2 - p1]), atoms._cell, atoms._pbc)
        q1, q2 = new[self.indices]
        d, q = find_mic(np.array([q2 - q1]), atoms._cell, atoms._pbc)
        d *= 0.5 * (p - q) / q
        new[self.indices] = (q1 - d[0], q2 + d[0])

    def adjust_forces(self, atoms, forces):
        d = np.subtract.reduce(atoms.positions[self.indices])
        d, p = find_mic(np.array([d]), atoms._cell, atoms._pbc)
        d = d[0]
        d *= 0.5 * np.dot(np.subtract.reduce(forces[self.indices]), d) / p**2
        self.constraint_force = d
        forces[self.indices] += (-d, d)

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        newa = [-1, -1]  # Signal error
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa

    def get_constraint_force(self):
        """Return the (scalar) force required to maintain the constraint"""
        return self.constraint_force

    def get_indices(self):
        return self.indices

    def __repr__(self):
        return 'FixBondLength(%d, %d)' % tuple(self.indices)

    def todict(self):
        return {'name': 'FixBondLength',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1]}}

###

Optimizer = ase.optimize.LBFGS
Optimizer_steps_limit = 1000

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

logger = screen

###

a, cryst, crk, k1g, tip_x, tip_y, bond1, bond2, boundary_mask, \
    boundary_mask_bulk, tip_mask = setup_crack(logger=logger)
ase.io.write('notch.xyz', a, format='extxyz')

# Get general parameters

basename = parameter('basename', 'energy_barrier')
calc = parameter('calc')
fmax = parameter('fmax', 0.01)

# Get parameter used for fitting crack tip position

optimize_tip_position = parameter('optimize_tip_position', False)
residual_func = parameter('residual_func', crack.displacement_residual)
_residual_func = residual_func
tip_tol = parameter('tip_tol', 1e-4)
tip_mixing_alpha = parameter('tip_mixing_alpha', 1.0)
write_trajectory_during_optimization = parameter('write_trajectory_during_optimization', False)

if optimize_tip_position:
    tip_x = (a.positions[bond1, 0] + a.positions[bond2, 0])/2
    tip_y = (a.positions[bond1, 1] + a.positions[bond2, 1])/2
    logger.pr('Optimizing tip position -> initially centering tip bond. '
              'Tip positions = {} {}'.format(tip_x, tip_y))

# Assign calculator.
a.set_calculator(calc)

sig_xx, sig_yy, sig_xy = crk.stresses(cryst.positions[:,0],
                                      cryst.positions[:,1],
                                      tip_x, tip_y,
                                      params.k1*k1g)
sig = np.vstack([sig_xx, sig_yy] + [ np.zeros_like(sig_xx)]*3 + [sig_xy])
eps = np.dot(crk.S, sig)

# Do we have a converged run that we want to restart from?
restart_from = parameter('restart_from', 'None')
if restart_from == 'None':
    restart_from = None
original_cell = a.get_cell().copy()

# Run crack calculation.
for i, bond_length in enumerate(params.bond_lengths):
    logger.pr('=== bond_length = {0} ==='.format(bond_length))
    xyz_file = '%s_%4d.xyz' % (basename, int(bond_length*1000))
    log_file = open('%s_%4d.log' % (basename, int(bond_length*1000)),
                    'w')
    if write_trajectory_during_optimization:
        traj_file = ase.io.NetCDFTrajectory('%s_%4d.nc' % \
            (basename, int(bond_length*1000)), mode='w', atoms=a)
        traj_file.write()
    else:
        traj_file = None

    mask = None
    if restart_from is not None:
        fn = '{0}/{1}'.format(restart_from, xyz_file)
        if os.path.exists(fn):
            logger.pr('Restart relaxation from {0}'.format(fn))
            b = ase.io.read(fn)
            mask = np.logical_or(b.numbers == atomic_numbers[ACTUAL_CRACK_TIP],
                                 b.numbers == atomic_numbers[FITTED_CRACK_TIP])
            del b[mask]
            print(len(a), len(b))
            print(a.numbers)
            print(b.numbers)
            assert np.all(a.numbers == b.numbers)
            a = b
            a.set_calculator(calc)
            tip_x, tip_y, dummy = a.info['actual_crack_tip']
            a.set_cell(original_cell, scale_atoms=True)

    a.set_constraint(None)
    a.set_distance(bond1, bond2, bond_length)
    bond_length_constraint = FixBondLength(bond1, bond2)

    # Deformation gradient residual needs full Atoms object and therefore
    # special treatment here.
    if _residual_func == crack.deformation_gradient_residual:
        residual_func = lambda r0, crack, x, y, ref_x, ref_y, k, mask=None:\
            _residual_func(r0, crack, x, y, a, ref_x, ref_y, cryst, k,
                           params.cutoff, mask)

    # Optimize x and z position of crack tip.
    if optimize_tip_position:
        old_x = tip_x
        old_y = tip_y
        converged = False
        while not converged:
            #b = cryst.copy()
            u0x, u0y = crk.displacements(cryst.positions[:,0],
                                         cryst.positions[:,1],
                                         old_x, old_y, params.k1*k1g)
            ux, uy = crk.displacements(cryst.positions[:,0],
                                       cryst.positions[:,1],
                                       tip_x, tip_y, params.k1*k1g)
            #b.positions[:,0] += ux
            #b.positions[:,1] += uy

            a.set_constraint(None)
            # Displace atom positions
            a.positions[:len(cryst),0] += ux-u0x
            a.positions[:len(cryst),1] += uy-u0y
            a.positions[bond1,0] -= ux[bond1]-u0x[bond1]
            a.positions[bond1,1] -= uy[bond1]-u0y[bond1]
            a.positions[bond2,0] -= ux[bond2]-u0x[bond2]
            a.positions[bond2,1] -= uy[bond2]-u0y[bond2]
            # Set bond length and boundary atoms explicitly to avoid numerical drift
            a.set_distance(bond1, bond2, bond_length)
            a.positions[boundary_mask,0] = \
                cryst.positions[boundary_mask_bulk,0] + ux[boundary_mask_bulk]
            a.positions[boundary_mask,1] = \
                cryst.positions[boundary_mask_bulk,1] + uy[boundary_mask_bulk]
            a.set_constraint([ase.constraints.FixAtoms(mask=boundary_mask),
                              bond_length_constraint])
            logger.pr('Optimizing positions...')
            opt = Optimizer(a, logfile=log_file)
            if traj_file:
                opt.attach(traj_file.write)
            opt.run(fmax=fmax, steps=Optimizer_steps_limit)
            logger.pr('...done. Converged within {0} steps.' \
                     .format(opt.get_number_of_steps()))

            old_x = tip_x
            old_y = tip_y
            tip_x, tip_y = crk.crack_tip_position(a.positions[:len(cryst),0],
                                                  a.positions[:len(cryst),1],
                                                  cryst.positions[:,0],
                                                  cryst.positions[:,1],
                                                  tip_x, tip_y,
                                                  params.k1*k1g,
                                                  mask=tip_mask,
                                                  residual_func=residual_func)
            dtip_x = tip_x-old_x
            dtip_y = tip_y-old_y
            logger.pr('- Fitted crack tip (before mixing) is at {:3.2f} {:3.2f} '
                     '(= {:3.2e} {:3.2e} from the former position).'.format(tip_x, tip_y, dtip_x, dtip_y))
            tip_x = old_x + tip_mixing_alpha*dtip_x
            tip_y = old_y + tip_mixing_alpha*dtip_y
            logger.pr('- New crack tip (after mixing) is at {:3.2f} {:3.2f} '
                     '(= {:3.2e} {:3.2e} from the former position).'.format(tip_x, tip_y, tip_x-old_x, tip_y-old_y))
            converged = np.asscalar(abs(dtip_x) < tip_tol and abs(dtip_y) < tip_tol)
    else:
        a.set_constraint([ase.constraints.FixAtoms(mask=boundary_mask),
                          bond_length_constraint])
        logger.pr('Optimizing positions...')
        opt = Optimizer(a, logfile=log_file)
        if traj_file:
            opt.attach(traj_file.write)
        opt.run(fmax=fmax,  steps=Optimizer_steps_limit)
        logger.pr('...done. Converged within {0} steps.' \
                 .format(opt.get_number_of_steps()))

    # Store forces.
    a.set_constraint(None)
    a.set_array('forces', a.get_forces())

    # Make a copy of the configuration.
    b = a.copy()

    # Fit crack tip (again), and get residuals.
    fit_x, fit_y, residuals = \
        crk.crack_tip_position(a.positions[:len(cryst),0],
                               a.positions[:len(cryst),1],
                               cryst.positions[:,0],
                               cryst.positions[:,1],
                               tip_x, tip_y, params.k1*k1g,
                               mask=mask,
                               residual_func=residual_func,
                               return_residuals=True)

    logger.pr('Measured crack tip at %f %f' % (fit_x, fit_y))
    #b.set_array('residuals', residuals)

    # The target crack tip is marked by a gold atom.
    b += ase.Atom(ACTUAL_CRACK_TIP, (tip_x, tip_y, b.cell.diagonal()[2]/2))
    b.info['actual_crack_tip'] = (tip_x, tip_y, b.cell.diagonal()[2]/2)

    # The fitted crack tip is marked by a silver atom.
    b += ase.Atom(FITTED_CRACK_TIP, (fit_x, fit_y, b.cell.diagonal()[2]/2))
    b.info['fitted_crack_tip'] = (fit_x, fit_y, b.cell.diagonal()[2]/2)

    bond_dir = a[bond1].position - a[bond2].position
    bond_dir /= np.linalg.norm(bond_dir)
    force = np.dot(bond_length_constraint.get_constraint_force(), bond_dir)

    b.info['bond_length'] = bond_length
    b.info['force'] = force
    b.info['energy'] = a.get_potential_energy()
    b.info['cell_origin'] = [0, 0, 0]
    ase.io.write(xyz_file, b, format='extxyz')

    log_file.close()
    if traj_file:
        traj_file.close()
