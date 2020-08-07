#! /usr/bin/env python

import os
import numpy as np

import ase.io
from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

import sys

sys.path.insert(0, '.')
import params

calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
vacuum = parameter('vacuum', 10.0)
flexible = parameter('flexible', True)
extended_far_field = parameter('extended_far_field', False)
k0 = parameter('k0', 1.0)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position
dump = parameter('dump', False)
precon = parameter('precon', False)
prerelax = parameter('prerelax', False)

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
                   extended_far_field=extended_far_field)

nsteps = parameter('nsteps')
k1_range = parameter('k1_range', np.linspace(0.8, 1.2, nsteps))

ks = list(k1_range)
ks_out = []
alphas = []

max_steps = parameter('max_steps', 10)
fit_alpha = parameter('fit_alpha', False)

for i, k in enumerate(ks):
    restart_file = parameter('restart_file', f'k_{int(k * 1000):04d}.xyz')
    if os.path.exists(restart_file):
        print(f'Restarting from {restart_file}')
        if restart_file.endswith('.xyz'):
            a = ase.io.read(restart_file)
            sc.set_atoms(a)
        elif restart_file.endswith('.txt'):
            x = np.loadtxt(restart_file)
            if len(x) == len(sc):
                sc.set_dofs(x)
            elif len(x) == len(sc) + 1:
                sc.variable_k = True
                sc.set_dofs(x)
                sc.variable_k = False
            elif len(x) == len(sc) + 2:
                sc.variable_alpha = True
                sc.variable_k = True
                sc.set_dofs(x)
                sc.variable_alpha = False
                sc.variable_k = False
            else:
                raise RuntimeError('cannot guess how to restart with'
                                   f' {len(x)} variables for {len(self)} DoFs')
        else:
            raise ValueError(f"don't know how to restart from {restart_file}")
    else:
        sc.rescale_k(k * k1g)

    if fit_alpha:
        sc.alpha, = sc.fit_cle(variable_alpha=True, variable_k=False)
        print(f'Fitted value of alpha: {sc.alpha}')

    print(f'k = {sc.k / k1g} * k1g')
    print(f'alpha = {sc.alpha}')

    if prerelax:
        print('Pre-relaxing with Conjugate-Gradients')
        sc.optimize(ftol=1e-5, steps=max_steps, dump=dump,
                    method='cg')

    sc.optimize(fmax, steps=max_steps, dump=dump,
                precon=precon, method='krylov')

    if flexible:
        print(f'Optimized alpha = {sc.alpha:.3f}')
    ks_out.append(k)
    alphas.append(sc.alpha)

    a = sc.atoms
    a.get_forces()
    ase.io.write(f'k_{int(k * 1000):04d}.xyz', a)
    with open('x.txt', 'a') as fh:
        np.savetxt(fh, [sc.get_dofs()])

np.savetxt('k_vs_alpha.txt', np.c_[ks_out, alphas])
