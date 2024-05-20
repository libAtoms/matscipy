import numpy as np
from matscipy.fracture_mechanics.clusters import diamond, bcc, set_regions
from ase.lattice.cubic import Diamond, FaceCenteredCubic, SimpleCubic, BodyCenteredCubic
import ase.io
from ase.constraints import ExpCellFilter
from ase.optimize import LBFGS
from matscipy.elasticity import fit_elastic_constants
import os
from ase.calculators.lammpslib import LAMMPSlib
from mpi4py import MPI

# read in from command line
# MPI setup
comm_world = MPI.COMM_WORLD
rank = comm_world.Get_rank()
#isolate each rank on a seperate communicator before passing in
single_comm = comm_world.Split(color=rank, key=rank)

# Fundamental material properties
el              = 'Fe'
a0_init         = 3.0
elastic_symmetry = 'cubic'

mass = 28.0855
yace = f'jace_castep_c0.yace'
table = f'jace_castep_c0_pairpot.table'
cmds = ['pair_style hybrid/overlay pace table spline 5500',
        f'pair_coeff * * pace {yace} Fe', f'pair_coeff 1 1 table {table} Fe_Fe']
calc = LAMMPSlib(comm=single_comm, amendments=[f"mass 1 {mass}"],lmpcmds = cmds,log_file='lammps_output.log',keep_alive=True)

# Find relaxed lattice constant
print('optimising lattice parameter')
unit_cell = BodyCenteredCubic(size=[1,1,1],symbol=el,latticeconstant=a0_init,pbc=(1,1,1))
unit_cell.set_calculator(calc)
ecf = ExpCellFilter(unit_cell)
uc_optimise = LBFGS(ecf)
uc_optimise.run(fmax=0.0001)
a0 = unit_cell.get_cell()[0,0] #get the optimised lattice parameter
print('LATTICE PARAMETER:',a0)

# Fit elastic constants
C, C_err = fit_elastic_constants(unit_cell, symmetry=elastic_symmetry)

# Crack system
crack_surface   = np.array([ 1, 1, 0 ])
crack_front     = np.array([ 1, -1, 0 ])
crack_direction = np.array(np.cross(crack_surface,crack_front))


# properties to change
K = 1.25 #mpa(sqrt m)
temperature = 100 #K

# simulation set up (lengths etc)
strip_height = 120
vacuum = 50
strip_width = 400
strip_thickness = 6
crack_seed_length = 100
strain_ramp_length = 50

# crack tip finding properties
bondlength = 3.3
bulk_nn = 14

# lattice building func (BCC)
lattice = BodyCenteredCubic

#simulation output
temp_path = './temp'
results_path = './sim_1_results'
dump_files = True
dump_freq = 100
dump_name = 'dump.lammpstrj'
thermo_freq = 100

#timestep lengths and runtimes of different parts
sim_tstep = 0.001 #1fs
n_steps_per_loop = 10
init_nsteps = 2000
n_loops = 10

#thermostat properties
damping_strength = 0.1
rseed = 1029 #change this for different results



