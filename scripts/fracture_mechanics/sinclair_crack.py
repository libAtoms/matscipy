#! /usr/bin/env python

import numpy as np

import ase.io
from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack
from scipy.optimize import fsolve
from scipy.optimize.nonlin import NoConvergence

import params

calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
vacuum = parameter('vacuum', 10.0)
alpha_scale = parameter('alpha_scale', 1.0)

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
sc = SinclairCrack(crk, cluster, calc, 1.0 * k1g,
                   vacuum=vacuum, alpha_scale=alpha_scale)

def f(k, alpha):
    sc.k = k * k1g
    sc.alpha = alpha
    return sc.get_crack_tip_force(mask=sc.regionI | sc.regionII | sc.regionIII)


k1_range = parameter('k1_range', 'auto')
dk = parameter('dk', 0.05)

if k1_range == 'auto':
    k = 1.0
    alphas = np.linspace(-3, 3, 50)

    klim = [0, 0]
    for iter, sgn in enumerate([-1, +1]):
        k = 1.0
        while True:
            f_alpha = [f(k, alpha) for alpha in alphas]
            np.savetxt(f'k_{int(k * 1000):04d}_f_alpha.txt', np.c_[alphas, f_alpha])

            # look for a solution to f(k, alpha) = 0 close to alpha = 0.
            alpha_opt, info, ierr, msg = fsolve(lambda alpha: f(k, alpha),
                                                0.0, full_output=True)
            print(f'k={k:.3f} alpha_opt={alpha_opt[0]:.3f} status={ierr}')
            if ierr == 1:
                k += sgn * dk
            else:
                k -= sgn * dk # back-track
                break
        klim[iter] = k

    print('automatically determined K-range: ', klim)
    ks = np.linspace(klim[0], klim[1], parameter('nsteps'))

else:
    ks = list(k1_range)

ks_out = []
alphas = []

max_steps = parameter('max_steps', 10)

for i, k in enumerate(ks):
    sc.rescale_k(k * k1g)
    print(f'k = {k} * k1g')

    try:
        sc.optimize(fmax, steps=max_steps)
    except NoConvergence:
        print(f'Skipping failed optimisation at k={k}*k1G')
        continue

    print(f'Optimized alpha = {sc.alpha:.3f}')
    ks_out.append(k)
    alphas.append(sc.alpha)

    a = sc.get_atoms()
    a.info['k'] = sc.k
    a.info['alpha'] = sc.alpha
    a.get_forces()
    ase.io.write(f'k_{int(k*1000):04d}.xyz', a)

np.savetxt('k_vs_alpha.txt', np.c_[ks_out, alphas])