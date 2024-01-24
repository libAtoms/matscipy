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

#add current folder to path
sys.path.insert(0, '.')

import numpy as np

import ase
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import Langevin
from ase.optimize import FIRE
from ase.symbols import string2symbols
from ase.units import mol, fs, kB

from matscipy import parameter
from matscipy.neighbours import neighbour_list

###

def handle_sigusr2(signum, frame):
    # Just shutdown and hope that there is no current write operation
    print("Received SIGUSR2. Terminating...")
    sys.exit(0)
signal.signal(signal.SIGUSR2, handle_sigusr2)

###

def random_solid(els, density, internal_cutoff=0.0, sx=None, sy=None):
    stable_struct = False
    while not stable_struct:
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

        nl = neighbour_list('i',atoms=a, cutoff=internal_cutoff)
        print(nl)
        if len(nl)==0:
            stable_struct = True

    return a

###

def set_up_lammps(lmps, input_file, mass_cmds, potential_cmds):
    #lmps : LAMMPS object
    #input_file : str, path to input file
    #mass_cmds : commands to set atomic masses in LAMMPS
    #potential_cmds : commands to set potential in LAMMPS
    
    # ---------- Initialize Simulation --------------------- 
    lmps.command('clear') 
    lmps.command('dimension 3')
    lmps.command('boundary p p p')
    lmps.command('atom_style atomic')
    lmps.command('units metal')

    print(input_file)
    #----------Read atoms------------
    lmps.command(f'read_data {input_file}')
    
    #----------Define masses and Interatomic Potential----------------
    lmps.commands_list(mass_cmds)
    lmps.commands_list(potential_cmds)

    #----------Output thermo data----------------
    lmps.command('thermo 100')


def main_lammps(lmps, rank):
    # This is a version of the function below, which instead uses the LAMMPS python interface with MPI4PY
    # to run the actual quenching simulation. This is faster than the ASE interface, but requires a 
    # LAMMPS python installation and LAMMPS potential.

    # it also (at the time of writing) requires a version of ASE, which has better LAMMPS file functionality
    # this is available on the master branch of the ASE github repo, but not yet on the latest release 
    
    #lmps is lammps object
    #rank is the process number in the communicator

    #read in params
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
    # read in lammps commands for potential
    # note that these are in the same form as one would pass to ASE's LAMMPSlib
    potential_cmds = parameter('potential_cmds')
    mass_cmds = parameter('mass_cmds')
    quick_calc = parameter('quick_calc') # this should be either a quick potential or a LAMMPSlib calculator


    internal_cutoff = parameter('internal_cutoff', 0.0)
    for _density in densities:
        try:
            density, sx, sy = _density
        except:
            density = _density
            sx = sy = None
        print('density =', density)

        initial_fn = 'density_%2.1f-initial.traj' % density

        liquid_fn = 'density_%2.1f-liquid.lammpstrj' % density
        liquid_final_fn = 'density_%2.1f-liquid.final.traj' % density

        quench_fn = 'density_%2.1f-quench.lammpstrj' % density
        quench_final_fn = 'density_%2.1f-quench.final.traj' % density

        print('=== LIQUID ===')
        print('rank = ', rank)
        if not os.path.exists(liquid_final_fn):
            if rank == 0:
                if not os.path.exists(liquid_fn):
                    print('... creating new solid ...')
                    a = random_solid(els, density, internal_cutoff=internal_cutoff,sx=sx, sy=sy)
                    n = a.get_atomic_numbers().copy()

                    # Relax with the quick potential
                    a.set_atomic_numbers([6]*len(a))
                    a.set_calculator(quick_calc)
                    FIRE(a, downhill_check=True).run(fmax=1.0, steps=nsteps_relax)
                    a.set_atomic_numbers(n)
                    a.calc = None
                    write(initial_fn, a,parallel=False)
                else:
                    print('... reading %s ...' % liquid_fn)

                    a = read(liquid_fn,index=-1,parallel=False,format='lammps-dump-text')

                # Write config to LAMMPS data file
                ase.io.lammpsdata.write_lammps_data('initial_lammps_cfg.lj', a, masses=True,velocities=True)

                #read in lammps dump file
                try:
                    traj = read(liquid_fn, index=':',parallel=False)
                    #get trajectory length
                    n_prev_steps = len(traj)
                except FileNotFoundError:
                    n_prev_steps = 0
                    
            else:
                n_prev_steps = 0
            
            #communicate n_prev_steps to all processors
            n_prev_steps = lmps.comm.bcast(n_prev_steps, root=0)

        
            # Thermalize with the slow (but correct) potential in LAMMPS
            # first, set up LAMMPS simulation
            set_up_lammps(lmps, 'initial_lammps_cfg.lj', mass_cmds, potential_cmds)

            # add dump command for LAMMPS to write trajectory to liquid_fn
            dump_freq =int(dtdump/dt1)
            lmps.command(f'dump myDump all custom {dump_freq} {liquid_fn} id type xs ys zs vx vy vz')
            lmps.command('dump_modify myDump append yes')
            #set lammps timestep (in ps), dt1 is in ase units
            lmps.command(f'timestep {(dt1/fs)/1000}')

            #add langevin thermostat
            #in lammps, the damping is specified in time units, not inverse time like ASE
            print('characteristic time', ((tau1/fs)/1000))
            lmps.command('fix 1 all nve')
            lmps.command(f'fix 2 all langevin {T1/kB} {T1/kB} {((tau1/fs)/1000)} 48279')

            nsteps = int(teq/dt1)-n_prev_steps*int(dtdump/dt1)
            print('Need to run for further {} steps to reach total of {} steps.'.format(nsteps, int(teq/dt1)))
            if nsteps <= 0:
                nsteps = 1

            lmps.command(f'run {nsteps}')

            # get lammps to write out final file
            lmps.command(f'write_data simulation_output.temp nocoeff nofix nolabelmap')

            if rank == 0:
                #read in final file
                a = ase.io.lammpsdata.read_lammps_data(f'simulation_output.temp',atom_style='atomic')
                
                # Write snapshot
                write(liquid_final_fn, a, parallel=False)
        else:
            print('... reading %s ...' % liquid_final_fn)
            a = read(liquid_final_fn,parallel=False)

        print('=== QUENCH ===')

        if not os.path.exists(quench_final_fn):
            if rank == 0:
                if os.path.exists(quench_fn):
                    print('... reading %s ...' % quench_fn)
                    a = read(quench_fn)

                # Write config to LAMMPS data file
                
                ase.io.lammpsdata.write_lammps_data('quench_lammps_cfg.lj', a, masses=True, velocities=True)

                #read in lammps dump file
                try:
                    traj = read(quench_fn, index=':',parallel=False)
                    #get trajectory length
                    n_prev_steps = len(traj)
                except FileNotFoundError:
                    n_prev_steps = 0
            else:
                n_prev_steps = 0

            #communicate n_prev_steps to all processors
            n_prev_steps = lmps.comm.bcast(n_prev_steps, root=0)

            # 10ps Langevin quench to 300K
            set_up_lammps(lmps, 'quench_lammps_cfg.lj', mass_cmds, potential_cmds)


            # add dump command for LAMMPS to dump to file quench_fn
            dump_freq =int(dtdump/dt2)
            lmps.command(f'dump quench_dump all custom {dump_freq} {quench_fn} id type xs ys zs vx vy vz')
            lmps.command('dump_modify quench_dump append yes')
            #set lammps timestep (in ps), dt2 is in ase units
            lmps.command(f'timestep {(dt2/fs)/1000}')
            #fix langevin thermostat
            lmps.command('fix 1 all nve')
            #in lammps, the damping is specified in time units, not inverse time like ASE
            lmps.command(f'fix 2 all langevin {T2/kB} {T2/kB} {((tau2/fs)/1000)} 48279')

            print('thermostat time')
            print(1.0/((tau2/fs)/1000))
            
            nsteps = int(tqu/dt2)-n_prev_steps*int(dtdump/dt2)
            print('Need to run for further {} steps to reach total of {} steps.'.format(nsteps, int(teq/dt1)))
            lmps.command(f'run {nsteps}')

            # get lammps to write out final file
            lmps.command(f'write_data simulation_output_quench.temp nocoeff nofix nolabelmap')

            if rank == 0:
                #read in final file
                a = ase.io.lammpsdata.read_lammps_data(f'simulation_output_quench.temp',atom_style='atomic')

                # Write snapshot
                write(quench_final_fn, a, parallel=False)

        
        if rank == 0:
            open('DONE_%2.1f' % density, 'w')


def main():
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
