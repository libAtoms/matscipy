from __future__ import (
    division,
    absolute_import,
    print_function,
    unicode_literals
)
from ase.io import Trajectory, read
from ase.units import GPa, kB, fs, m, s
import numpy as np
from ase.md.langevin import Langevin
from matscipy import pressurecoupling as pc
from io import open


# Parameters
dt = 1.0 * fs  # MD time step
C11 = 500.0 * GPa  # material constant
p_c = 0.10  # experience value
Pdir = 2  # index of cell axis along normal pressure is applied
P = 5.0 * GPa  # target normal pressure
v = 100.0 * m / s  # constant sliding speed
vdir = 0  # index of cell axis along sliding happens
T = 300.0  # target temperature for thermostat
           # thermostat is applied in the third direction which
           # is neither pressure nor sliding direction and only
           # in the middle region between top and bottom.
           # This makes sense for small systems which cannot have
           # a dedicated thermostat region.
t_langevin = 75.0 * fs  # time constant for Langevin thermostat
gamma_langevin = 1. / t_langevin  # derived Langevin parameter
t_integrate = 1000.0 * fs  # simulation time
steps_integrate = int(t_integrate / dt)  # number of simulation steps

# get atoms from trajectory to also initialize correct velocities
atoms = read('slide.traj')

bottom_mask = np.loadtxt("bottom_mask.txt").astype(bool)
top_mask = np.loadtxt("top_mask.txt").astype(bool)

damp = pc.AutoDamping(C11, p_c)
slider = pc.SlideWithNormalPressureCuboidCell(top_mask, bottom_mask, Pdir, P, vdir, v, damp)
atoms.set_constraint(slider)

calc = ASE_CALCULATOR_OBJECT  # put a specific calculator here

atoms.set_calculator(calc)
temps = np.zeros((len(atoms), 3))
temps[slider.middle_mask, slider.Tdir] = kB * T
gammas = np.zeros((len(atoms), 3))
gammas[slider.middle_mask, slider.Tdir] = gamma_langevin
integrator = Langevin(atoms, dt, temps, gammas, fixcm=False)
trajectory = Trajectory('slide.traj', 'a', atoms)  # append

with open('log_slide.txt', 'r', encoding='utf-8') as log_handle:
    step_offset = pc.SlideLog(log_handle).step[-1]
log_handle = open('log_slide.txt', 'a', 1, encoding='utf-8')  # line buffered append
logger = pc.SlideLogger(log_handle, atoms, slider, integrator, step_offset)

integrator.attach(logger)
integrator.attach(trajectory)
integrator.run(steps_integrate)
log_handle.close()
trajectory.close()
