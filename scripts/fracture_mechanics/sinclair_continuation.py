#
# Copyright 2020-2021 James Kermode (Warwick U.)
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

import matscipy; print(matscipy.__file__)

import os
import numpy as np

import h5py
import ase.io
from ase.units import GPa
from ase.constraints import FixAtoms
from ase.optimize.precon import PreconLBFGS

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize import fsolve, fminbound

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
a0 = parameter('a0') # lattice constant
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

    # reuse output from sinclair_crack.py if possible
    if os.path.exists(f'k_{int(k0 * 1000):04d}.xyz'):
        print(f'Reading atoms from k_{int(k0 * 1000):04d}.xyz')
        a = ase.io.read(f'k_{int(k0 * 1000):04d}.xyz')
        sc.set_atoms(a)

    mask = sc.regionI | sc.regionII
    if extended_far_field:
        mask = sc.regionI | sc.regionII | sc.regionIII

    # first use the CLE-only approximation`: define a function f_alpha0(k, alpha)
    def f(k, alpha):
        sc.k = k * k1g
        sc.alpha = alpha
        sc.update_atoms()
        return sc.get_crack_tip_force(mask=mask)

    # identify approximate range of stable k
    alpha_range = parameter('alpha_range', np.linspace(-a0, a0, 100))
    k = k0  # initial guess for k
    alpha_k = []
    for alpha in alpha_range:
        # look for solution to f(k, alpha) = 0 close to alpha = alpha
        (k,) = fsolve(f, k, args=(alpha,))
        print(f'alpha={alpha:.3f} k={k:.3f} ')
        alpha_k.append((alpha, k))
    alpha_k = np.array(alpha_k)
    kmin, kmax = alpha_k[:, 1].min(), alpha_k[:, 1].max()
    print(f'Estimated stable K range is {kmin} < k / k_G < {kmax}')

    # define a function to relax with static scheme at a given value of k
    # note that we use a looser fmax of 1e-3 and reduced max steps of 50
    def g(k):
        print(f'Static minimisation with k={k}, alpha={alpha0}.')
        sc.k = k * k1g
        sc.alpha = alpha0
        sc.variable_alpha = False
        sc.variable_k = False
        sc.update_atoms()
        atoms = sc.atoms.copy()
        atoms.calc = sc.calc
        atoms.set_constraint(FixAtoms(mask=~sc.regionI))
        opt = PreconLBFGS(atoms, logfile=None)
        opt.run(fmax=1e-5)
        sc.set_atoms(atoms)
        f_alpha = sc.get_crack_tip_force(mask=mask)
        print(f'Static minimisation with k={k}, alpha={alpha0} --> f_alpha={f_alpha}')
        return abs(f_alpha)

    # minimise g(k) in [kmin, kmax]
    kopt, falpha_min, ierr, funccalls = fminbound(g, kmin, kmax, xtol=1e-8, full_output=True)
    print(f'Brent minimisation yields f_alpha={falpha_min} for k = {kopt} after {funccalls} calls')

    # re-optimize, first with static scheme, then flexible scheme
    sc.k = kopt * k1g
    sc.alpha = alpha0
    sc.variable_alpha = False
    sc.optimize(ftol=1e-3, steps=max_steps)

    # finally, we revert to target fmax precision
    sc.variable_alpha = flexible
    sc.optimize(ftol=fmax, steps=max_steps)

    sc.atoms.write('x0.xyz')
    x0 = np.r_[sc.get_dofs(), k0 * k1g]
    alpha0 = sc.alpha

    # obtain a second solution x1 = (u_1, alpha_1, k_1) where
    # k_1 ~= k_0 and alpha_1 ~= alpha_0

    print(f'Rescaling K_I from {sc.k} to {sc.k + dk * k1g}')
    k1 = k0 + dk
    sc.rescale_k(k1 * k1g)


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
