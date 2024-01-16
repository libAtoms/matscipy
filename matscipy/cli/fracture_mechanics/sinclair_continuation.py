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

import matscipy

import os
import numpy as np

import h5py
import ase.io
from ase.units import GPa
from ase.constraints import FixAtoms
# from ase.optimize import LBFGSLineSearch
from ase.optimize.precon import PreconLBFGS

from matscipy import parameter
from matscipy.elasticity import fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize import fsolve, fminbound

import sys

def main():
    print(matscipy.__file__)

    sys.path.insert(0, '../../../staging/fracture_mechanics')

    import params

    calc = parameter('calc')
    fmax = parameter('fmax', 1e-3)
    max_opt_steps = parameter('max_opt_steps', 100)
    max_arc_steps = parameter('max_arc_steps', 10)
    vacuum = parameter('vacuum', 10.0)
    flexible = parameter('flexible', True)
    continuation = parameter('continuation', False)
    ds = parameter('ds', 1e-2)
    nsteps = parameter('nsteps', 10)
    a0 = parameter('a0')  # lattice constant
    k0 = parameter('k0', 1.0)
    extended_far_field = parameter('extended_far_field', False)
    alpha0 = parameter('alpha0', 0.0)  # initial guess for crack position
    dump = parameter('dump', False)
    precon = parameter('precon', False)
    prerelax = parameter('prerelax', False)
    traj_file = parameter('traj_file', 'x_traj.h5')
    restart_file = parameter('restart_file', traj_file)
    traj_interval = parameter('traj_interval', 1)
    direction = parameter('direction', +1)
    ds_max = parameter('ds_max', 0.1)
    ds_min = parameter('ds_min', 1e-6)
    ds_aggressiveness = parameter('ds_aggressiveness', 1.25)
    opt_method = parameter('opt_method', 'krylov')

    # make copies of initial configs
    cryst = params.cryst.copy()
    cluster = params.cluster.copy()

    # compute elastic constants
    cryst.pbc = True
    cryst.calc = calc
    C, C_err = fit_elastic_constants(cryst, symmetry=parameter('elastic_symmetry', 'triclinic'))

    # setup the crack
    crk = CubicCrystalCrack(parameter('crack_surface'),
                            parameter('crack_front'),
                            Crot=C / GPa)

    # get Griffith's stess intensity factor
    k1g = crk.k1g(parameter('surface_energy'))
    print('Griffith k1 = %f' % k1g)

    # check for restart files if continuation=True
    if continuation and not os.path.exists(restart_file):
        continuation = False
    if continuation:

        # works only if restart file contains atleast two consecutive flex solutions
        with h5py.File(restart_file, 'r') as hf:
            x = hf['x']
            restart_index = parameter('restart_index', x.shape[0] - 1)
            x0 = x[restart_index - 1, :]
            x1 = x[restart_index, :]
            k0 = x0[-1] / k1g;
            k1 = x1[-1] / k1g
            alpha0 = x0[-2] / k1g;
            alpha1 = x1[-2] / k1g
            # print restart info
            print(f'Restarting from k0={k0}, k1={k1} --> alpha0={alpha0}, alpha1={alpha1}')

    else:

        # setup Sinclair boundary conditions with variable_alpha and no variable_k, for approx solution
        sc = SinclairCrack(crk, cluster, calc,
                           k0 * k1g, alpha=alpha0,
                           vacuum=vacuum,
                           variable_alpha=flexible,
                           extended_far_field=extended_far_field)

        traj = None
        dk = parameter('dk', 1e-4)
        dalpha = parameter('dalpha', 1e-3)

        # reuse output from sinclair_crack.py if possible
        if os.path.exists(f'k_{int(k0 * 1000):04d}.xyz'):
            print(f'Reading atoms from k_{int(k0 * 1000):04d}.xyz')
            a = ase.io.read(f'k_{int(k0 * 1000):04d}.xyz')
            sc.set_atoms(a)

        # setup mask fpr relevant regions
        mask = sc.regionI | sc.regionII
        if extended_far_field:
            mask = sc.regionI | sc.regionII | sc.regionIII

        # estimate the stable K range
        k_range = parameter('k_range', 'Unknown')
        if isinstance(k_range, list) and len(k_range) == 2:
            kmin, kmax = k_range
            print(f'Using K range from params file: {kmin} < k / k_G < {kmax}')
        else:
            print('No k_range=[kmin,kmax] given in params.')
            print('Running CLE-only approximation to estimate stable K range.')


            # first use the CLE-only approximation`: define a function f_alpha0(k, alpha)
            def f(k, alpha):
                sc.k = k * k1g
                sc.alpha = alpha
                sc.update_atoms()
                return sc.get_crack_tip_force(mask=mask)


            # identify approximate range of stable k
            alpha_range = parameter('alpha_range', np.linspace(-a0, a0, 20))
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
        traj = open("traj.xyz", "w")


        def g(k, do_abs=True):
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
            # opt = LBFGSLineSearch(atoms)
            opt.run(fmax=1e-3)
            atoms.write(traj, format="extxyz")
            sc.set_atoms(atoms)
            f_alpha = sc.get_crack_tip_force(mask=mask)
            print(f'Static minimisation with k={k}, alpha={alpha0} --> f_alpha={f_alpha}')
            if do_abs:
                f_alpha = abs(f_alpha)
            return f_alpha


        # minimise g(k) in [kmin, kmax]
        kopt, falpha_min, ierr, funccalls = fminbound(g, kmin, kmax, xtol=1e-8, full_output=True)
        print(f'Brent minimisation yields f_alpha={falpha_min} for k = {kopt} after {funccalls} calls')
        traj.close()

        # Find flex1
        # first optimize with static scheme
        # sc.k = kopt * k1g
        k0 = kopt
        sc.rescale_k(k0 * k1g)
        sc.alpha = alpha0
        sc.variable_alpha = False
        sc.optimize(ftol=1e-3, steps=max_opt_steps, method=opt_method)
        # then revert to target fmax precision and optimize with flexible scheme
        sc.variable_alpha = flexible  # True
        sc.optimize(ftol=fmax, steps=max_opt_steps, method=opt_method)
        # save flex1
        sc.atoms.write('x0.xyz')
        x0 = np.r_[sc.get_dofs(), k0 * k1g]
        alpha0 = sc.alpha

        # Find flex2: a second solution x1 = (u_1, alpha_1, k_1) where
        # k_1 ~= k_0 and alpha_1 ~= alpha_0
        # Increase k, and rescale Us accordingly
        print(f'Rescaling K_I from {sc.k} to {sc.k + dk * k1g}')
        k1 = k0 + dk
        sc.rescale_k(k1 * k1g)
        # optimize at target fmax precision with flexible scheme
        sc.variable_alpha = flexible  # True
        sc.optimize(ftol=fmax, steps=max_opt_steps, method=opt_method)
        # save flex2
        sc.atoms.write('x1.xyz')
        x1 = np.r_[sc.get_dofs(), k1 * k1g]
        alpha1 = sc.alpha

        # check crack tip didn't jump too far
        print(f'k0={k0}, k1={k1} --> alpha0={alpha0}, alpha1={alpha1}')
        assert abs(alpha1 - alpha0) < dalpha

    # setup new crack with variable_k for full solution
    scv = SinclairCrack(crk, cluster, calc,
                        k0 * k1g, alpha=alpha0,
                        vacuum=vacuum,
                        variable_alpha=flexible, variable_k=True,
                        extended_far_field=extended_far_field)

    # run full arc length continuation
    scv.arc_length_continuation(x0, x1, N=nsteps,
                                ds=ds, ftol=fmax, max_steps=max_arc_steps,
                                direction=direction,
                                continuation=continuation,
                                traj_file=traj_file,
                                traj_interval=traj_interval,
                                precon=precon,
                                ds_max=ds_max, ds_min=ds_min,
                                ds_aggressiveness=ds_aggressiveness,
                                opt_method='krylov')  # 'ode12r' preconditioning needs debugging
