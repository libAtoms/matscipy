#! /usr/bin/env python

import os
import numpy as np

from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize import fsolve
from scipy.optimize.nonlin import NoConvergence

import params

calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
max_steps = parameter('max_steps', 100)
vacuum = parameter('vacuum', 10.0)
alpha_scale = parameter('alpha_scale', 1.0)
continuation = parameter('continuation', False)
ds = parameter('ds', 1e-2)
nsteps = parameter('nsteps', 10)
method = parameter('method', 'full')
k0 = parameter('k0', 1.0)

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

if method == 'approximate':
    sc = SinclairCrack(crk, cluster, calc, k0 * k1g, vacuum=vacuum,
                       alpha_scale=alpha_scale)
    def f(k, alpha):
        sc.k = k * k1g
        sc.alpha = alpha
        return sc.get_crack_tip_force(
            mask=sc.regionI | sc.regionII | sc.regionIII)

    alpha_range = parameter('alpha_range', np.linspace(-1.0, 1.0, 100))
    # look for a solution to f(k, alpha) = 0 close to alpha = 0.
    traj = []
    for alpha in alpha_range:
        k = fsolve(f, k, args=(alpha,), full_output=True)
        print(f'k={k:.3f} alpha={alpha:.3f}')
        traj.append((alpha, k))

    np.savetxt('traj_approximate.txt', traj)

elif method == 'full':

    if continuation and os.path.exists('traj.txt'):
        traj = np.loadtxt('traj.txt')
        half = traj.shape[1] // 2
        x, xdot = traj[:, :half], traj[:, half]
        x0 = x[-2, :]
        x1 = x[-1, :]
        k0 = x0[-1]
        sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                           vacuum=vacuum, alpha_scale=alpha_scale)
    else:
        traj = None
        dk = parameter('dk', 1e-4)
        k1 = k0 + dk

        sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                           vacuum=vacuum, alpha_scale=alpha_scale)

        # obtain one solution x0 = (u_0, alpha_0, k_0)
        try:
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
        assert abs(alpha1 - alpha0) < 1e-3

    sc.variable_k = True
    new_traj = sc.arc_length_continuation(x0, x1, N=nsteps,
                                          ds=ds, ftol=fmax, steps=max_steps,
                                          continuation=continuation,
                                          dump=True)
else:
    raise ValueError(f'unknown method {method}')

    if traj is None:
        traj = new_traj
    else:
        traj = np.r_[traj, new_traj]
    np.savetxt('traj.txt', traj)