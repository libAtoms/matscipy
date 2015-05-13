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

# USAGE:
#
# Code imports the file 'params.py' from current working directory. params.py
# contains simulation parameters. Some parameters can be omitted, see below.

import os
import sys

import numpy as np

import ase
import ase.constraints
import ase.io
import ase.optimize
from ase.data import atomic_numbers
from ase.parallel import parprint
from ase.units import GPa

import matscipy.fracture_mechanics.crack as crack
from matscipy.checkpoint import Checkpoint, NoCheckpoint
from matscipy.elasticity import fit_elastic_constants
from matscipy.logger import screen
from matscipy.neighbours import neighbour_list

###

sys.path += ['.', '..']
import params

###

def param(s, d):
    global logger
    try:
        val = params.__dict__[s]
        logger.pr('(user value)      {} = {}'.format(s, val))
    except KeyError:
        val = d
        logger.pr('(default value)   {} = {}'.format(s, val))
    return val

###

Optimizer = ase.optimize.FIRE

# Atom types used for outputting the crack tip position.
ACTUAL_CRACK_TIP = 'Au'
FITTED_CRACK_TIP = 'Ag'

###

logger = screen

# Checkpointing
CP = Checkpoint(logger=logger)
fit_elastic_constants = CP(fit_elastic_constants)

###

cryst = params.cryst.copy()

# Double check elastic constants. We're just assuming this is really a periodic
# system. (True if it comes out of the cluster routines.)

compute_elastic_constants = param('compute_elastic_constants', False)

if params.compute_elastic_constants:
    pbc = cryst.pbc.copy()
    cryst.set_pbc(True)
    cryst.set_calculator(params.calc)
    C, C_err = fit_elastic_constants(cryst, verbose=False,
                                     optimizer=ase.optimize.FIRE)
    cryst.set_pbc(pbc)

    print('Measured elastic constants (in GPa):')
    print(np.round(C*10/GPa)/10)

    crk = crack.CubicCrystalCrack(params.crack_surface, params.crack_front,
                                  Crot=C/GPa)
else:
    if hasattr(params, 'C'):
        crk = crack.CubicCrystalCrack(params.crack_surface, params.crack_front,
                                      C=params.C)
    else:    
        crk = crack.CubicCrystalCrack(params.crack_surface, params.crack_front,
                                      params.C11, params.C12, params.C44)


print('Elastic constants used for boundary condition (in GPa):')
print(np.round(crk.C*10)/10)

# Get parameter used for fitting crack tip position

residual_func = param('residual_func', crack.displacement_residual)
_residual_func = residual_func

tip_tol = param('tip_tol', 1e-4)

tip_mixing_alpha = param('tip_mixing_alpha', 1.0)

write_trajectory_during_optimization = param('write_trajectory_during_optimization', False)
    
# Get Griffith's k1.
k1g = crk.k1g(params.surface_energy)
parprint('Griffith k1 = %f' % k1g)

# Apply initial strain field.
tip_x = param('tip_x', cryst.cell.diagonal()[0]/2)
tip_y = param('tip_y', cryst.cell.diagonal()[1]/2)

a = cryst.copy()
ux, uy = crk.displacements(cryst.positions[:,0], cryst.positions[:,1],
                           tip_x, tip_y, params.k1*k1g)
a.positions[:,0] += ux
a.positions[:,1] += uy

# Center notched configuration in simulation cell and ensure enough vacuum.
oldr = a[0].position.copy()
a.center(vacuum=params.vacuum, axis=0)
a.center(vacuum=params.vacuum, axis=1)
tip_x += a[0].position[0] - oldr[0]
tip_y += a[0].position[1] - oldr[1]
cryst.set_cell(a.cell)
cryst.translate(a[0].position - oldr)

# Groups mark the fixed region and the region use for fitting the crack tip.
g = a.get_array('groups')

# Choose which bond to break.
bond1, bond2 = param('bond', crack.find_tip_coordination(a, bondlength=2.7))

print('Opening bond {0}--{1}, initial bond length {2}'.
      format(bond1, bond2, a.get_distance(bond1, bond2, mic=True)))

# centre vertically on the opening bond
a.translate([0., a.cell[1,1]/2.0 - 
                (a.positions[bond1, 1] + 
                 a.positions[bond2, 1])/2.0, 0.])

ase.io.write('notch.xyz', a, format='extxyz')

### Notched system has been created here ###

optimize_tip_position = param('optimize_tip_position', False)

if optimize_tip_position:
    tip_x = (a.positions[bond1, 0] + a.positions[bond2, 0])/2
    tip_y = (a.positions[bond1, 1] + a.positions[bond2, 1])/2
    logger.pr('Optimizing tip position -> initially centering tip bond. '
              'Tip positions = {} {}'.format(tip_x, tip_y))

# Assign calculator.
a.set_calculator(params.calc)

sig_xx, sig_yy, sig_xy = crk.stresses(cryst.positions[:,0],
                                      cryst.positions[:,1],
                                      tip_x, tip_y,
                                      params.k1*k1g)
sig = np.vstack([sig_xx, sig_yy] + [ np.zeros_like(sig_xx)]*3 + [sig_xy])
eps = np.dot(crk.S, sig)

# Run crack calculation.
for i, bond_length in enumerate(params.bond_lengths):
    parprint('=== bond_length = {0} ==='.format(bond_length))
    xyz_file = '%s_%4d.xyz' % (params.basename, int(bond_length*1000))
    try:
        a = CP.load(a)
    except NoCheckpoint:
        log_file = open('%s_%4d.log' % (params.basename, int(bond_length*1000)),
                        'w')
        if write_trajectory_during_optimization:
            traj_file = ase.io.NetCDFTrajectory('%s_%4d.nc' % \
                (params.basename, int(bond_length*1000)), mode='w', atoms=a)
            traj_file.write()
        else:
            traj_file = None

        a.set_constraint(None)
        a.set_distance(bond1, bond2, bond_length)
        bond_length_constraint = ase.constraints.FixBondLength(bond1, bond2)

        # Deformation gradient residual needs full Atoms object and therefore
        # special treatment here.
        if _residual_func == crack.deformation_gradient_residual:
            residual_func = lambda r0, crack, x, y, ref_x, ref_y, k, mask=None:\
                _residual_func(r0, crack, x, y, a, ref_x, ref_y, cryst, k,
                               params.cutoff, mask)

        # Atoms to be used for fitting the crack tip position.
        mask = g==1

        # Optimize x and z position of crack tip.
        if optimize_tip_position:
            try:
                a, converged, tip_x, tip_y, old_x, old_y = CP.load(a)
            except NoCheckpoint:
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
                #a.positions[g==0] = b.positions[g==0]
                a.positions[:,0] += ux-u0x
                a.positions[:,1] += uy-u0y
                a.positions[bond1,0] -= ux[bond1]-u0x[bond1]
                a.positions[bond1,1] -= uy[bond1]-u0y[bond1]
                a.positions[bond2,0] -= ux[bond2]-u0x[bond2]
                a.positions[bond2,1] -= uy[bond2]-u0y[bond2]
                # Set bond length and boundary atoms explicitly to avoid numerical drift
                a.set_distance(bond1, bond2, bond_length)
                a.positions[g==0,0] = cryst.positions[g==0,0] + ux[g==0]
                a.positions[g==0,1] = cryst.positions[g==0,1] + uy[g==0]
                a.set_constraint([ase.constraints.FixAtoms(mask=g==0),
                                  bond_length_constraint])
                parprint('Optimizing positions...')
                opt = Optimizer(a, logfile=log_file)
                if traj_file:
                    opt.attach(traj_file.write)
                opt.run(fmax=params.fmax)
                parprint('...done. Converged within {0} steps.' \
                         .format(opt.get_number_of_steps()))

                old_x = tip_x
                old_y = tip_y
                tip_x, tip_y = crk.crack_tip_position(a.positions[:,0],
                                                      a.positions[:,1],
                                                      cryst.positions[:,0],
                                                      cryst.positions[:,1],
                                                      tip_x, tip_y,
                                                      params.k1*k1g,
                                                      mask=mask,
                                                      residual_func=residual_func)
                dtip_x = tip_x-old_x
                dtip_y = tip_y-old_y
                parprint('- Fitted crack tip (before mixing) is at {:3.2f} {:3.2f} '
                         '(= {:3.2e} {:3.2e} from the former position).'.format(tip_x, tip_y, dtip_x, dtip_y))
                tip_x = old_x + tip_mixing_alpha*dtip_x
                tip_y = old_y + tip_mixing_alpha*dtip_y
                parprint('- New crack tip (after mixing) is at {:3.2f} {:3.2f} '
                         '(= {:3.2e} {:3.2e} from the former position).'.format(tip_x, tip_y, tip_x-old_x, tip_y-old_y))
                converged = np.asscalar(abs(dtip_x) < tip_tol and abs(dtip_y) < tip_tol)
                CP.flush(a, converged, tip_x, tip_y, old_x, old_y)
        else:
            a.set_constraint([ase.constraints.FixAtoms(mask=g==0),
                              bond_length_constraint])
            parprint('Optimizing positions...')
            opt = Optimizer(a, logfile=log_file)
            if traj_file:
                opt.attach(traj_file.write)
            opt.run(fmax=params.fmax)
            parprint('...done. Converged within {0} steps.' \
                     .format(opt.get_number_of_steps()))

        # Store forces.
        a.set_constraint(None)
        a.set_array('forces', a.get_forces())

        # Make a copy of the configuration.
        b = a.copy()

        # Fit crack tip (again), and get residuals.
        fit_x, fit_y, residuals = \
            crk.crack_tip_position(a.positions[:,0],
                                   a.positions[:,1],
                                   cryst.positions[:,0],
                                   cryst.positions[:,1],
                                   tip_x, tip_y, params.k1*k1g,
                                   mask=mask,
                                   residual_func=residual_func,
                                   return_residuals=True)

        parprint('Measured crack tip at %f %f' % (fit_x, fit_y))
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

        CP.save(a)