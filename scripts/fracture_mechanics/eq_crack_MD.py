from matscipy.fracture_mechanics.thin_strip_utils import ThinStripBuilder, set_up_eq_crack_simulation
from matscipy import parameter
import ase.io.lammpsdata
import ase.io
import numpy as np
from mpi4py import MPI
from lammps import lammps
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import os 
import params

me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

#strip width, height and thickness (in angstrom)
strip_height = parameter('strip_height')
strip_width = parameter('strip_width')
strip_thickness = parameter('strip_thickness')

#amount of vacuum around the strip in the simulation
# when building the cell - relatively unimportant so long as it's not too little
vacuum = parameter('vacuum')

# length of crack and length of initial strain ramp
crack_seed_length = parameter('crack_seed_length')
strain_ramp_length = parameter('strain_ramp_length')

#elastic constants
C = parameter('C')

#lattice constant
a0 = parameter('a0')

#Crack directions
directions = [parameter('crack_direction'),
              parameter('crack_surface'),
              parameter('crack_front')]

#element
el = parameter('el')

#equilibrium bulk properties (bond length and number of nearest neighbours)
# this is used for finding the crack tip
bondlength = parameter('bondlength')
bulk_nn = parameter('bulk_nn')

#stress intensity factor applied to crack tip
K = parameter('K')

#ASE lattice builder object
lattice = parameter('lattice')

#calculator and lammps commands used for defining pairstyle
calc = parameter('calc')
mass = parameter('mass') #atomic mass
cmds = parameter('cmds')

#whether or not it's a multilattice (for cb corrector reasons)
multilattice = parameter('multilattice',False)

#whether to approximate the strain from the assumption of linear elasticity
#or just measure the energy-strain relationship and get strain from G
approximate_strain = parameter('approximate_strain',False)

#Places to store files written during the simulation
#note that ./temp will be overwritten if it exists, but ./results will not.
temp_path = parameter('temp_path','./temp')
results_path = parameter('results_path','./results')

#simulation timestep
sim_tstep = parameter('sim_tstep',0.001) #ps

#Langevin thermostat parameters
damping_strength = parameter('damping_strength',0.1) #Langevin damping strength
rseed = parameter('rseed',1029) #random seed for Langevin thermostat

#output options
dump_files = parameter('dump_files', True)
dump_freq = parameter('dump_freq',100)
dump_name = parameter('dump_name','dump.lammpstrj')
thermo_freq = parameter('thermo_freq',100)

#temperature
temperature = parameter('temperature')

#initial number of steps to get the simulation going
init_nsteps = parameter('init_nsteps',2000)

#number of steps between each velocity check
n_steps_per_loop = parameter('n_steps_per_loop',1000)

#number of loops in total
n_loops = parameter('n_loops',10)

cb = parameter('cb','None')
if cb == 'None':
    cb = None

# -------------Main code----------------
lmp = lammps()
if me == 0:
    tsb = ThinStripBuilder(el,a0,C,calc,lattice,directions,multilattice=multilattice,cb=cb,switch_sublattices=True)
    tsb.measure_energy_strain_relation(resolution=1000)
    
    #note track spacing is not important for this simulation, so just set to high value
    crack_slab = tsb.build_thin_strip_with_crack(K,strip_width,strip_height,strip_thickness\
                                               ,vacuum,crack_seed_length,strain_ramp_length,track_spacing=100,
                                               apply_x_strain=False,approximate=approximate_strain)
    os.makedirs(temp_path,exist_ok=True)
    os.makedirs(results_path, exist_ok=False)
    print(f'Writing initial crack to file "{temp_path}/crack.xyz"')
    crack_slab.set_velocities(np.zeros([len(crack_slab),3]))    
    ase.io.write(f'{temp_path}/crack.xyz', crack_slab, format='extxyz')
    ase.io.lammpsdata.write_lammps_data(f'{temp_path}/crack.lj',crack_slab,velocities=True)


#synchonize processes
MPI.COMM_WORLD.Barrier()
set_up_eq_crack_simulation(lmp,temp_path,mass,cmds,
                             sim_tstep=sim_tstep,dump_freq=dump_freq, dump_files=dump_files,
                             dump_name=dump_name,thermo_freq=thermo_freq,T=temperature,
                             damping_strength=damping_strength,rseed=rseed)

#run for some number of timesteps to get things going
lmp.command(f'run {init_nsteps}')
#write temp output file
lmp.command(f'write_data {temp_path}/simulation_output.temp nocoeff nofix nolabelmap')

####### MAYBE UNFIX THERMOSTAT HERE?? #########

#get initial crack tip position
if me == 0:
    final_crack_state = ase.io.lammpsdata.read_lammps_data(f'{temp_path}/simulation_output.temp',atom_style='atomic')
    final_crack_state.new_array('groups',crack_slab.arrays['groups'])
    tip_pos,tip_pos_y,bond_atoms = tsb.find_strip_crack_tip(final_crack_state,bondlength,bulk_nn)

    initial_tip_pos = tip_pos
    print(f'Initial crack tip x position: {initial_tip_pos}')
#-----------Main loop---------------
total_time = 0

times = []
tip_positions = []
velocities = []
for i in range(n_loops):
    lmp.command(f'run {n_steps_per_loop}')
    total_time += n_steps_per_loop*sim_tstep
    #write temp output file
    lmp.command(f'write_data {temp_path}/simulation_output.temp nocoeff nofix nolabelmap')
    if me == 0:
        final_crack_state = ase.io.lammpsdata.read_lammps_data(f'{temp_path}/simulation_output.temp',atom_style='atomic')
        final_crack_state.new_array('groups',crack_slab.arrays['groups'])

        tip_pos,tip_pos_y,bond_atoms = tsb.find_strip_crack_tip(final_crack_state,bondlength,bulk_nn)
        dist_traveled = tip_pos - initial_tip_pos
        print(f'Measured crack tip position: {tip_pos}')
        print(f'Crack velocity: {dist_traveled/total_time} angstrom/ps')
        times.append(total_time)
        tip_positions.append(tip_pos)
        velocities.append(dist_traveled/total_time)
        arr_to_save = np.zeros([len(times),3])
        arr_to_save[:,0] = times
        arr_to_save[:,1] = tip_positions
        arr_to_save[:,2] = velocities
        np.savetxt(f'{results_path}/K={K}_T={temperature}.txt',arr_to_save,header='Time(ps) Tip position (angstrom) Velocity (angstrom/ps)')