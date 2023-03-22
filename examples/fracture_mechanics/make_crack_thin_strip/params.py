#
# Copyright 2014 James Kermode (Warwick U.)
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
from ase.lattice import bulk
import ase.units as units

# ********** Bulk unit cell ************

# 8-atom diamond cubic unit cell for silicon
a0 = 5.44 # guess at lattice constant for Si - we will minimize
cryst = bulk('Si', 'diamond', a=a0, cubic=True)

surf_ny = 4 # number of cells needed to get accurate surface energy

# *********  System parameters **********

# There are three possible crack systems, choose one and uncomment it

# System 1. (111)[0-11]
crack_direction = (-2, 1, 1)      # Miller index of x-axis
cleavage_plane = (1, 1, 1)        # Miller index of y-axis
crack_front = (0, 1, -1)          # Miller index of z-axis

# # System 2. (110)[001]
# crack_direction = (1,-1,0)
# cleavage_plane = (1,1,0)
# crack_front = (0,0,1)

# # System 3. (110)[1-10]
# crack_direction = (0,0,-1)
# cleavage_plane = (1,1,0)
# crack_front = (1,-1,0)

check_rotated_elastic_constants = False

width = 200.0*units.Ang              # Width of crack slab
height = 100.0*units.Ang             # Height of crack slab
vacuum = 100.0*units.Ang             # Amount of vacuum around slab
crack_seed_length = 40.0*units.Ang   # Length of seed crack
strain_ramp_length = 30.0*units.Ang  # Distance over which strain is ramped up
initial_G = 5.0*(units.J/units.m**2) # Initial energy flow to crack tip

relax_bulk = True                     # If True, relax initial bulk cell
bulk_fmax  = 1e-6*units.eV/units.Ang  # Max force for bulk, C_ij and surface energy

relax_slab = True                     # If True, relax notched slab with calculator
relax_fmax = 0.025*units.eV/units.Ang # Maximum force criteria for relaxation

# ******* Molecular dynamics parameters ***********

sim_T = 300.0*units.kB           # Simulation temperature
nsteps = 10000                   # Total number of timesteps to run for
timestep = 1.0*units.fs          # Timestep (NB: time base units are not fs!)
cutoff_skin = 2.0*units.Ang      # Amount by which potential cutoff is increased
                                 # for neighbour calculations
tip_move_tol = 10.0              # Distance tip has to move before crack 
                                 # is taken to be running
strain_rate = 1e-5*(1/units.fs)  # Strain rate
traj_file = 'traj.nc'            # Trajectory output file (NetCDF format)
traj_interval = 10               # Number of time steps between
                                 # writing output frames

# ********** Setup calculator ************

# Stillinger-Weber (SW) classical interatomic potential, from QUIP
from quippy import Potential
calc = Potential('IP SW', 'params.xml')

# Screened Kumagai potential, from Atomistica
#import atomistica
#calc = atomistica.KumagaiScr()



