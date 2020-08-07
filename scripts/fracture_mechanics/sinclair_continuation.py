#! /usr/bin/env python

import matscipy; print(matscipy.__file__)

import os
import numpy as np

import h5py
import ase.io
from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize.nonlin import NoConvergence

import sys
sys.path.insert(0, '.')

import params

calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
max_steps = parameter('max_steps', 100)
vacuum = parameter('vacuum', 10.0)
flexible = parameter('flexible', True)
continuation = parameter('continuation', False)
ds = parameter('ds', 1e-2)
nsteps = parameter('nsteps', 10)
k0 = parameter('k0', 1.0)
extended_far_field = parameter('extended_far_field', False)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
dump = parameter('dump', False)
precon = parameter('precon', False)
prerelax = parameter('prerelax', False)
traj_file = parameter('traj_file', 'x_traj.h5')
traj_interval = parameter('traj_interval', 1)
direction = parameter('direction', +1)

# compute elastic constants
cryst = params.cryst.copy()
cryst.pbc = True
cryst.calc = calc
C, C_err = fit_elastic_constants(cryst,
                                 symmetry=parameter('elastic_symmetry',
                                                    'triclinic'))

crk = CubicCrystalCrack(parameter('crack_surface'),
                        parameter('crack_front'),
                        Crot=C/GPa)

# Get Griffith's k1.
k1g = crk.k1g(parameter('surface_energy'))
print('Griffith k1 = %f' % k1g)

cluster = params.cluster.copy()

if continuation and not os.path.exists(traj_file):
    continuation = False

if continuation:
    with h5py.File(traj_file, 'r') as hf:
        x = hf['x']
        restart_index = parameter('restart_index', x.shape[0] - 1)
        x0 = x[restart_index-1, :]
        x1 = x[restart_index, :]
        k0 = x0[-1] / k1g

sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                   alpha=alpha0,
                   vacuum=vacuum,
                   variable_alpha=flexible,
                   extended_far_field=extended_far_field)

if not continuation:
    traj = None
    dk = parameter('dk', 1e-4)
    dalpha = parameter('dalpha', 1e-3)
    k1 = k0 + dk

    # obtain one solution x0 = (u_0, alpha_0, k_0)

    # reuse output from sinclair_crack.py if possible
    if os.path.exists(f'k_{int(k0 * 1000):04d}.xyz'):
        print(f'Reading atoms from k_{int(k0 * 1000):04d}.xyz')
        a = ase.io.read(f'k_{int(k0 * 1000):04d}.xyz')
        sc.set_atoms(a)
    try:
        print(f'k = {k0} * k1g, alpha = {sc.alpha}')

        if prerelax:
            print('Pre-relaxing with Conjugate-Gradients')
            sc.optimize(ftol=1e-5, steps=max_steps, dump=dump, method='cg')
        else:
            sc.optimize(fmax, steps=max_steps, dump=dump,
                        precon=precon, method='krylov')
    except NoConvergence:
        sc.atoms.write('dump.xyz')
        raise
    sc.atoms.write('x0.xyz')
    x0 = np.r_[sc.get_dofs(), k0 * k1g]
    alpha0 = sc.alpha

    # obtain a second solution x1 = (u_1, alpha_1, k_1) where
    # k_1 ~= k_0 and alpha_1 ~= alpha_0
    print(f'Rescaling K_I from {sc.k} to {sc.k + dk * k1g}')
    sc.rescale_k(k1 * k1g)
    if prerelax:
        print('Pre-relaxing with Conjugate-Gradients')
        sc.optimize(ftol=1e-5, steps=max_steps, dump=dump, method='cg')
    else:
        sc.optimize(fmax, steps=max_steps, dump=dump,
                    precon=precon, method='krylov')
    sc.atoms.write('x1.xyz')
    x1 = np.r_[sc.get_dofs(), k1 * k1g]
    # check crack tip didn't jump too far
    alpha1 = sc.alpha
    print(f'k0={k0}, k1={k1} --> alpha0={alpha0}, alpha1={alpha1}')
    assert abs(alpha1 - alpha0) < dalpha

scv = SinclairCrack(crk, cluster, calc, k0 * k1g,
                    alpha=alpha0,
                    vacuum=vacuum,
                    variable_alpha=flexible,
                    variable_k=True,
                    extended_far_field=extended_far_field)

scv.arc_length_continuation(x0, x1, N=nsteps,
                            ds=ds, ftol=fmax, steps=max_steps,
                            direction=direction,
                            continuation=continuation,
                            traj_file=traj_file,
                            traj_interval=traj_interval,
                            precon=precon)
