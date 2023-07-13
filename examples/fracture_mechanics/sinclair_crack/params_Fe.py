import ase.io
import numpy as np
from matscipy.fracture_mechanics.clusters import bcc, set_regions

# Interaction potential
#from ase.calculators.lammpslib import LAMMPSlib
#cmds = ["pair_style eam/fs", "pair_coeff * * Fe_2_eam.fs Fe"]
#calc = LAMMPSlib(lmpcmds=cmds, log_file='test.log')
#from matscipy.calculators.eam import EAM
#calc = EAM("eam/pot/Fe_2_eam.fs")
from pyjulip import ACE1
calc = ACE1("ace/pot/Fe_ace_N4_D18_R5.json")

# material properties
el = 'Fe'
elastic_symmetry = 'cubic'
from matscipy.dislocation import get_elastic_constants
a0, C11, C12, C44 = get_elastic_constants(calculator=calc, symbol=el)

# crack system
r_I = 30.0
r_III = 60.0
cutoff = 6.0 # for region 4, beyond r_III
crack_surface = [1, 0, 0]
crack_front = [0, 0, 1]

# crack surface energy
#from matscipy.fracture_mechanics.clusters import find_surface_energy
from matscipy.surface import find_surface_energy
surface_energy = find_surface_energy(el,calc,a0,'bcc100',unit='0.1J/m^2')[0]

# simulation control
fmax = 1e-4 #5e-4
ds = 0.001
opt_method = 'ode12r' # currently only for flex1 & flex2, arc_cont uses krylov
#precon = True # turn on precon for arc_cont using ode12r
max_opt_steps = 500 # number of ode12r/krylov/sg steps for flex1/flex2
max_arc_steps = 10 # number of corrector steps in the arc-continuation
nsteps = 1000 # number of arc continuation steps
circular_regions = True
extended_far_field = True # set to True to inslude region III
# restart tags
#continuation = True
#restart_index = 0

# set circular regions I-II-III-IV
if circular_regions:
    n = [2 * int((r_III + cutoff)/ a0), 2 * int((r_III + cutoff)/ a0) - 1, 1]
    print('n=', n)

    # Setup crack system and regions I, II, III and IV
    cryst = bcc(el, a0, n, crack_surface, crack_front)
    cluster = set_regions(cryst, r_I, cutoff, r_III)  # carve circular cluster

# Write initial configs
ase.io.write('cryst.cfg', cryst)
ase.io.write('cluster.cfg', cluster)
