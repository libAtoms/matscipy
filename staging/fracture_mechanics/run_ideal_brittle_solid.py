#
# Copyright 2014, 2018 James Kermode (Warwick U.)
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

import sys

import numpy as np
from scipy.interpolate import interp1d

import ase.io
from ase.io.netcdftrajectory import NetCDFTrajectory
from ase.atoms import Atoms
from ase.md import VelocityVerlet
from ase.optimize.fire import FIRE

from matscipy.fracture_mechanics.idealbrittlesolid import (IdealBrittleSolid,
                                                           triangular_lattice_slab,
                                                           find_crack_tip,
                                                           set_initial_velocities,
                                                           set_constraints,
                                                           extend_strip)

from matscipy.fracture_mechanics.crack import (thin_strip_displacement_y,
                                               ConstantStrainRate)
from matscipy.numerical import numerical_forces

sys.path.insert(0, '.')
import params

calc = IdealBrittleSolid(rc=params.rc, k=params.k, a=params.a, beta=params.beta)

x_dimer = np.linspace(params.a-(params.rc-params.a),
                      params.a+1.1*(params.rc-params.a),51)
dimers = [Atoms('Si2', [(0, 0, 0), (x, 0, 0)],
                cell=[10., 10., 10.], pbc=True) for x in x_dimer]
calc.set_reference_crystal(dimers[0])
e_dimer = []
f_dimer = []
f_num = []
for d in dimers:
    d.set_calculator(calc)
    e_dimer.append(d.get_potential_energy())
    f_dimer.append(d.get_forces())
    f_num.append(numerical_forces(d))
e_dimer = np.array(e_dimer)
f_dimer = np.array(f_dimer)
f_num = np.array(f_num)
assert abs(f_dimer - f_num).max() < 0.1

crystal = triangular_lattice_slab(params.a, 3*params.N, params.N)
calc.set_reference_crystal(crystal)
crystal.set_calculator(calc)

e0 = crystal.get_potential_energy()
l = crystal.cell[0,0]
h = crystal.cell[1,1]
print 'l=', l, 'h=', h

# compute surface (Griffith) energy
b = crystal.copy()
b.set_calculator(calc)
shift = calc.parameters['rc']*2
y = crystal.positions[:, 1]
b.positions[y > h/2, 1] += shift
b.cell[1, 1] += shift
e1 = b.get_potential_energy()
E_G = (e1 - e0)/l
print 'Griffith energy', E_G

# compute Griffith strain
eps = 0.0   # initial strain is zero
eps_max = 2/np.sqrt(3)*(params.rc-params.a)*np.sqrt(params.N-1)/h # Griffith strain assuming harmonic energy
deps = eps_max/100. # strain increment
e_over_l = 0.0     # initial energy per unit length is zero
energy = []
strain = []
while e_over_l < E_G:
    c = crystal.copy()
    c.set_calculator(calc)
    c.positions[:, 1] *= (1.0 + eps)
    c.cell[1,1] *= (1.0 + eps)
    e_over_l = c.get_potential_energy()/l
    energy.append(e_over_l)
    strain.append(eps)
    eps += deps

energy = np.array(energy)
eps_of_e = interp1d(energy, strain, kind='linear')
eps_G = eps_of_e(E_G)

print 'Griffith strain', eps_G

c = crystal.copy()
c.info['E_G'] = E_G
c.info['eps_G'] = eps_G

# open up the cell along x and y by introducing some vaccum
orig_cell_width = c.cell[0, 0]
orig_cell_height = c.cell[1, 1]
c.center(params.vacuum, axis=0)
c.center(params.vacuum, axis=1)

# centre the slab on the origin
c.positions[:, 0] -= c.positions[:, 0].mean()
c.positions[:, 1] -= c.positions[:, 1].mean()

c.info['cell_origin'] = [-c.cell[0,0]/2, -c.cell[1,1]/2, 0.0]
ase.io.write('crack_1.xyz', c, format='extxyz')

width = (c.positions[:, 0].max() -
         c.positions[:, 0].min())
height = (c.positions[:, 1].max() -
          c.positions[:, 1].min())

c.info['OrigHeight'] = height

print(('Made slab with %d atoms, original width and height: %.1f x %.1f A^2' %
       (len(c), width, height)))

top = c.positions[:, 1].max()
bottom = c.positions[:, 1].min()
left = c.positions[:, 0].min()
right = c.positions[:, 0].max()

crack_seed_length = 0.3*width
strain_ramp_length = 5.0*params.a
delta_strain = params.strain_rate*params.dt

# fix top and bottom rows, and setup Stokes damping mask
# initial use constant strain
set_constraints(c, params.a)

# apply initial displacment field
c.positions[:, 1] += thin_strip_displacement_y(
                                 c.positions[:, 0],
                                 c.positions[:, 1],
                                 params.delta*eps_G,
                                 left + crack_seed_length,
                                 left + crack_seed_length +
                                        strain_ramp_length)

print('Applied initial load: delta=%.2f strain=%.4f' %
      (params.delta, params.delta*eps_G))

ase.io.write('crack_2.xyz', c, format='extxyz')

c.set_calculator(calc)

# relax initial structure
#opt = FIRE(c)
#opt.run(fmax=1e-3)

ase.io.write('crack_3.xyz', c, format='extxyz')

dyn = VelocityVerlet(c, params.dt, logfile=None)
set_initial_velocities(dyn.atoms)

crack_pos = []
traj = NetCDFTrajectory('traj.nc', 'w', c)
dyn.attach(traj.write, 10, dyn.atoms, arrays=['stokes', 'momenta'])
dyn.attach(find_crack_tip, 10, dyn.atoms,
           dt=params.dt*10, store=True, results=crack_pos)

# run for 2000 time steps to reach steady state at initial load
for i in range(20):
    dyn.run(100)
    if extend_strip(dyn.atoms, params.a, params.N, params.M, params.vacuum):
        set_constraints(dyn.atoms, params.a)

# start decreasing strain
#set_constraints(dyn.atoms, params.a, delta_strain=delta_strain)

strain_atoms = ConstantStrainRate(dyn.atoms.info['OrigHeight'],
                                  delta_strain)
dyn.attach(strain_atoms.apply_strain, 1, dyn.atoms)

for i in range(1000):
    dyn.run(100)
    if extend_strip(dyn.atoms, params.a, params.N, params.M, params.vacuum):
        set_constraints(dyn.atoms, params.a)

traj.close()

time = 10.0*dyn.dt*np.arange(dyn.get_number_of_steps()/10)
np.savetxt('crackpos.dat', np.c_[time, crack_pos])
