#! /usr/bin/env python

import os
import numpy as np

from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize.nonlin import NoConvergence

import params

calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
max_steps = parameter('max_steps', 100)
vacuum = parameter('vacuum', 10.0)
alpha_scale = parameter('alpha_scale', 1.0)
k_scale = parameter('k_scale', 1.0)
flexible = parameter('flexible', True)
continuation = parameter('continuation', False)
ds = parameter('ds', 1e-2)
nsteps = parameter('nsteps', 10)
k0 = parameter('k0', 1.0)
extended_far_field = parameter('extended_far_field', False)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
traj_interval = parameter('traj_interval', 1)

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

if continuation:
    if not os.path.exists('x_traj.txt'):
        continuation = False
if continuation:
    x = np.loadtxt('x_traj.txt')
    x0 = x[-2, :]
    x1 = x[-1, :]
    k0 = x0[-1] / k1g

sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                   alpha=alpha0,
                   vacuum=vacuum,
                   variable_alpha=flexible,
                   alpha_scale=alpha_scale, k_scale=k_scale,
                   extended_far_field=extended_far_field)

if not continuation:
    traj = None
    dk = parameter('dk', 1e-4)
    dalpha = parameter('dalpha', 1e-3)
    k1 = k0 + dk

    # obtain one solution x0 = (u_0, alpha_0, k_0)
    try:
        print(f'k = {k0} * k1g, alpha = {sc.alpha}')
        sc.optimize(fmax, steps=max_steps)
    except NoConvergence:
        a = sc.get_atoms()
        a.write('dump.xyz')
        raise
    x0 = np.r_[sc.get_dofs(), k0 * k1g]
    alpha0 = sc.alpha

    # obtain a second solution x1 = (u_1, alpha_1, k_1) where
    # k_1 ~= k_0 and alpha_1 ~= alpha_0
    print(f'Rescaling K_I from {sc.k} to {sc.k + dk * k1g}')
    sc.rescale_k(k1 * k1g)
    sc.optimize(fmax, steps=max_steps)
    x1 = np.r_[sc.get_dofs(), k1 * k1g]
    # check crack tip didn't jump too far
    alpha1 = sc.alpha
    print(f'k0={k0}, k1={k1} --> alpha0={alpha0}, alpha1={alpha1}')
    assert abs(alpha1 - alpha0) < dalpha

sc.variable_k = True
x_traj = sc.arc_length_continuation(x0, x1, N=nsteps,
                                    ds=ds, ftol=fmax, steps=max_steps,
                                    continuation=continuation,
                                    traj_interval=traj_interval)
