#! /usr/bin/env python

import os
import numpy as np

from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize import fsolve

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

sc = SinclairCrack(crk, cluster, calc, k0 * k1g, vacuum=vacuum,
                   alpha_scale=alpha_scale)
def f(k, alpha):
    sc.k = k * k1g
    sc.alpha = alpha
    return sc.get_crack_tip_force(
        mask=sc.regionI | sc.regionII)

alpha_range = parameter('alpha_range', np.linspace(-1.0, 1.0, 100))
# look for a solution to f(k, alpha) = 0 close to alpha = 0.
k = k0 # initial guess for k
traj = []
for alpha in alpha_range:
    (k,) = fsolve(f, k, args=(alpha,))
    print(f'alpha={alpha:.3f} k={k:.3f} ')
    traj.append((alpha, k))

np.savetxt('traj_approximate.txt', traj)