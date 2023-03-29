import ase.io
import numpy as np
from matscipy.fracture_mechanics.clusters import bcc, set_regions

# Interaction potential
#from ase.calculators.lammpslib import LAMMPSlib
#cmds = ["pair_style eam/fs",
#        "pair_coeff * * Fe_2_eam.fs Fe"]
#calc = LAMMPSlib(lmpcmds=cmds, log_file='test.log')
from matscipy.calculators.eam import EAM
calc = EAM("../pot/Fe_2_eam.fs")

# material properties
el = 'Fe'
elastic_symmetry = 'cubic'
from matscipy.dislocation import get_elastic_constants
a0, C11, C12, C44 = get_elastic_constants(calculator=calc, symbol=el)

# crack system
r_I = 55.0
r_III = 90.0
cutoff = 6.0 # for region 4, beyond r_III
crack_surface = [1, 0, 0]
crack_front = [0, 0, 1]

# crack surface energy
from ase.lattice.cubic import BodyCenteredCubic
from ase.optimize.precon import PreconLBFGS
from ase import units
bulk_x100 = BodyCenteredCubic(directions=[[1,0,0], [0,1,0], [0,0,1]],size=(8,1,1), symbol='Fe', pbc=(1,1,1),latticeconstant=a0)
cell = bulk_x100.get_cell() ; cell[0,:] *=2 # vacuum along x axis (100) direction
slab_x100 = bulk_x100.copy() ; slab_x100.set_cell(cell)
bulk_x100.calc = calc
opt_bulk = PreconLBFGS(bulk_x100)
opt_bulk.run(fmax=0.0001)
ene_bulk = bulk_x100.get_potential_energy()
slab_x100.calc = calc
opt_slab = PreconLBFGS(slab_x100)
opt_slab.run(fmax=0.0001)
ene_slab = slab_x100.get_potential_energy()
area = np.linalg.norm(np.cross(slab_x100.get_cell()[1,:],slab_x100.get_cell()[2,:]))
gamma_ase = (ene_slab - ene_bulk)/(2*area)
gamma_SI = (gamma_ase / units.J ) * (units.m)**2
surface_energy = gamma_SI * 10

# simulation control
fmax = 1e-4
ds = 0.001
nsteps = 10000 # number of arc continuation steps
max_opt_steps = 1000 # number of ode12r/krylov/sg steps for flex1/flex2
max_arc_steps = 10 # number of corrector steps in the arc-continuation
continuation = False # set to True for restart jobs
extended_far_field = True # set to True to inslude region III
circular_regions = True

# Defaults in sinclair_continuation.py (for reference)
# calc = parameter('calc')
# fmax = parameter('fmax', 1e-3)
# max_opt_steps = parameter('max_opt_steps', 100)
# max_arc_steps = parameter('max_arc_steps', 10)
# vacuum = parameter('vacuum', 10.0)
# flexible = parameter('flexible', True)
# continuation = parameter('continuation', False)
# ds = parameter('ds', 1e-2)
# nsteps = parameter('nsteps', 10)
# a0 = parameter('a0') # lattice constant
# k0 = parameter('k0', 1.0)
# extended_far_field = parameter('extended_far_field', False)
# alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
# dump = parameter('dump', False)
# precon = parameter('precon', False)
# prerelax = parameter('prerelax', False)
# traj_file = parameter('traj_file', 'x_traj.h5')
# restart_file = parameter('restart_file', traj_file)
# traj_interval = parameter('traj_interval', 1)
# direction = parameter('direction', +1)
# dk = parameter('dk', 1e-4)
# dalpha = parameter('dalpha', 1e-3)
# ds_max = parameter('ds_max', 0.1)
# ds_min = parameter('ds_min', 1e-6)
# ds_aggressiveness=parameter('ds_aggressiveness', 1.25)


# set circular regions I-II-III-IV
if circular_regions:
    n = [2 * int((r_III + cutoff)/ a0), 2 * int((r_III + cutoff)/ a0) - 1, 1]
    print('n=', n)

    # Setup crack system and regions I, II, III and IV
    cryst = bcc(el, a0, n, crack_surface, crack_front)
    cluster = set_regions(cryst, r_I, cutoff, r_III)  # carve circular cluster

# srite initial configs
ase.io.write('cryst.cfg', cryst)
ase.io.write('cluster.cfg', cluster)
