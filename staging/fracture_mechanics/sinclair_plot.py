#
# Copyright 2014, 2020 James Kermode (Warwick U.)
#           2020 Arnaud Allera (U. Lyon 1)
#           2014 Lars Pastewka (U. Freiburg)
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
import os
import sys
import itertools

import numpy as np
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=1.0)

import ase.io
from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack


args = sys.argv[1:]
if not args:
    args = ['.'] # current directory only

sys.path.insert(0, args[0])
import params

calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
vacuum = parameter('vacuum', 10.0)
flexible = parameter('flexible', True)
extended_far_field = parameter('extended_far_field', False)

k0 = parameter('k0', 1.0)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position

colors = parameter('colors',
                   ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"])

colors = itertools.cycle(colors)

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

plot_approx = parameter('plot_approx', False)
if plot_approx:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    axs = [ax1, ax2]
else:
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax2 = ax1
    axs = [ax1]

plot_labels = parameter('plot_labels', [])
if plot_labels == []:
    plot_labels = [ f'$R$ = {directory.split("_")[1]}' for directory in args ]

atoms = []
for directory, color, label in zip(args, colors, plot_labels):
    fn = f'{directory}/x_traj.h5'
    if os.path.exists(fn):
        cluster = ase.io.read(f'{directory}/cluster.cfg')
        sc = SinclairCrack(crk, cluster, calc, k0 * k1g, alpha=alpha0,
                           variable_alpha=flexible, variable_k=True, vacuum=vacuum,
                           extended_far_field=extended_far_field)

        with h5py.File(fn) as hf:
            x = hf['x']
            sc.set_dofs(x[0])
            a = sc.atoms
            atoms.append(a)

            k = x[:, -1]
            if flexible:
                alpha = x[:, -2]
                ax1.plot(k / k1g, alpha, c=color, label=label)
            else:
                u = x[:, :-1]
                ax1.plot(k / k1g, np.linalg.norm(u, axis=1),
                         c=color, label=label)
    if plot_approx and os.path.exists(f'{directory}/traj_approximate.txt'):
        print(f'Loading {directory}/traj_approximate.txt...')
        alpha, k = np.loadtxt(f'{directory}/traj_approximate.txt').T
        ax2.plot(k, alpha, '--', c=color, label=label)

for a in atoms[:-1]:
    a.set_cell(atoms[-1].cell)
    a.center(axis=0)
    a.center(axis=1)
ase.io.write('atoms.xyz', atoms)

klim = parameter('klim', ())
alphalim = parameter('alphalim', ())

for ax in axs:
    ax.axvline(1.0, color='k')
    if klim is not ():
        ax.set_xlim(klim)
    if alphalim is not ():
        ax.set_ylim(alphalim)
    if flexible:
        ax.set_ylabel(r'Crack position $\alpha$')
    else:
        ax.set_ylabel(r'Norm of corrector $\||u\||$ / $\mathrm{\AA{}}$')
    ax.legend(loc='upper right')
ax2.set_xlabel(r'Stress intensity factor $K/K_{G}$')

pdffile = parameter('pdffile', 'plot.pdf')
plt.savefig(pdffile)