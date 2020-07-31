#! /usr/bin/env python

import numpy as np

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
vacuum = parameter('vacuum', 10.0)
alpha_scale = parameter('alpha_scale', 1.0)
k_scale = parameter('k_scale', 1.0)
flexible = parameter('flexible', True)
extended_far_field = parameter('extended_far_field', False)
k0 = parameter('k0', 1.0)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
dump = parameter('dump', False)
precon = parameter('precon', False)
method = parameter('method', 'krylov')

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
sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                   alpha=alpha0,
                   variable_alpha=flexible,
                   vacuum=vacuum,
                   alpha_scale=alpha_scale,
                   k_scale=k_scale,
                   extended_far_field=extended_far_field)

nsteps = parameter('nsteps')
k1_range = parameter('k1_range', np.linspace(0.8, 1.2, nsteps))

ks = list(k1_range)
ks_out = []
alphas = []

max_steps = parameter('max_steps', 10)

for i, k in enumerate(ks):
    sc.rescale_k(k * k1g)
    print(f'k = {k} * k1g')
    print(f'alpha = {sc.alpha}')

    try:
        sc.optimize(fmax, steps=max_steps, dump=dump,
                    precon=precon, method=method)
    except NoConvergence:
        print(f'Skipping failed optimisation at k={k} * k1G')
        continue

    if flexible:
        print(f'Optimized alpha = {sc.alpha:.3f}')
    ks_out.append(k)
    alphas.append(sc.alpha)

    a = sc.get_atoms()
    a.info['k'] = sc.k
    a.info['alpha'] = sc.alpha
    a.get_forces()
    ase.io.write(f'k_{int(k*1000):04d}.xyz', a)
    with open('x.txt', 'a') as fh:
        np.savetxt(fh, [sc.get_dofs()])

np.savetxt('k_vs_alpha.txt', np.c_[ks_out, alphas])
