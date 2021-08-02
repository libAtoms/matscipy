#
# Copyright 2016-2019, 2021 Lars Pastewka (U. Freiburg)
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

import os
import signal
import sys

import numpy as np

import ase
from ase.atoms import string2symbols
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.md import Langevin
from ase.units import mol, fs, kB

from matscipy import parameter

###

def handle_sigusr2(signum, frame):
    # Just shutdown and hope that there is no current write operation
    print("Received SIGUSR2. Terminating...")
    sys.exit(0)
signal.signal(signal.SIGUSR2, handle_sigusr2)

###

def random_solid(els, density, sx=None, sy=None):
    if isinstance(els, str):
        syms = string2symbols(els)
    else:
        syms = ""
        for sym, n in els:
            syms += n*sym
    r = np.random.rand(len(syms), 3)
    a = ase.Atoms(syms, positions=r, cell=[1,1,1], pbc=True)

    mass = np.sum(a.get_masses())
    if sx is not None and sy is not None:
        sz = ( 1e24*mass/(density*mol) )/(sx*sy)
        a.set_cell([sx,sy,sz], scale_atoms=True)
    else:
        a0 = ( 1e24*mass/(density*mol) )**(1./3)
        a.set_cell([a0,a0,a0], scale_atoms=True)
    a.set_initial_charges([0]*len(a))

    return a

###

# For coordination counting
cutoff = 1.85

els = parameter('stoichiometry')
densities  = parameter('densities')

T1 = parameter('T1', 5000*kB)
T2 = parameter('T2', 300*kB)
dt1 = parameter('dt1', 0.1*fs)
dt2 = parameter('dt2', 0.1*fs)
tau1 = parameter('tau1', 5000*fs)
tau2 = parameter('tau2', 500*fs)
dtdump = parameter('dtdump', 100*fs)
teq = parameter('teq', 50e3*fs)
tqu = parameter('tqu', 20e3*fs)
nsteps_relax = parameter('nsteps_relax', 10000)

###

quick_calc = parameter('quick_calc')
calc = parameter('calc')

###

for _density in densities:
    try:
        density, sx, sy = _density
    except:
        density = _density
        sx = sy = None
    print('density =', density)

    initial_fn = 'density_%2.1f-initial.traj' % density

    liquid_fn = 'density_%2.1f-liquid.traj' % density
    liquid_final_fn = 'density_%2.1f-liquid.final.traj' % density

    quench_fn = 'density_%2.1f-quench.traj' % density
    quench_final_fn = 'density_%2.1f-quench.final.traj' % density

    print('=== LIQUID ===')

    if not os.path.exists(liquid_final_fn):
        if not os.path.exists(liquid_fn):
            print('... creating new solid ...')
            a = random_solid(els, density, sx=sx, sy=sy)
            n = a.get_atomic_numbers().copy()

            # Relax with the quick potential
            a.set_atomic_numbers([6]*len(a))
            a.set_calculator(quick_calc)
            FIRE(a, downhill_check=True).run(fmax=1.0, steps=nsteps_relax)
            a.set_atomic_numbers(n)
            write(initial_fn, a)
        else:
            print('... reading %s ...' % liquid_fn)
            a = read(liquid_fn)

        # Thermalize with the slow (but correct) potential
        a.set_calculator(calc)
        traj = Trajectory(liquid_fn, 'a', a)
        dyn = Langevin(a, dt1, T1, 1.0/tau1,
                       logfile='-', loginterval=int(dtdump/dt1))
        dyn.attach(traj.write, interval=int(dtdump/dt1)) # every 100 fs
        nsteps = int(teq/dt1)-len(traj)*int(dtdump/dt1)
        print('Need to run for further {} steps to reach total of {} steps.'.format(nsteps, int(teq/dt1)))
        if nsteps <= 0:
            nsteps = 1
        dyn.run(nsteps)
        traj.close()

        # Write snapshot
        write(liquid_final_fn, a)
    else:
        print('... reading %s ...' % liquid_final_fn)
        a = read(liquid_final_fn)
        a.set_calculator(calc)

    print('=== QUENCH ===')

    if not os.path.exists(quench_final_fn):
        if os.path.exists(quench_fn):
            print('... reading %s ...' % quench_fn)
            a = read(quench_fn)
            a.set_calculator(calc)

        # 10ps Langevin quench to 300K
        traj = Trajectory(quench_fn, 'a', a)
        dyn = Langevin(a, dt2, T2, 1.0/tau2,
                       logfile='-', loginterval=200)
        dyn.attach(traj.write, interval=int(dtdump/dt2)) # every 100 fs
        nsteps = int(tqu/dt2)-len(traj)*int(dtdump/dt2)
        print('Need to run for further {} steps to reach total of {} steps.'.format(nsteps, int(teq/dt1)))
        dyn.run(nsteps)

        # Write snapshot
        write(quench_final_fn, a)

    open('DONE_%2.1f' % density, 'w')
