#
# Copyright 2020 James Kermode (Warwick U.)
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

import sys
import numpy as np

from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack

from scipy.optimize import fsolve

sys.path.insert(0, '.')
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

mask = sc.regionI | sc.regionII

extended_far_field = parameter('extended_far_field', False)
if extended_far_field:
    mask = sc.regionI | sc.regionII | sc.regionIII

def f(k, alpha):
    sc.k = k * k1g
    sc.alpha = alpha
    sc.update_atoms()
    return sc.get_crack_tip_force(
        mask=mask)

alpha_range = parameter('alpha_range', np.linspace(-1.0, 1.0, 100))
# look for a solution to f(k, alpha) = 0 close to alpha = 0.
k = k0  # initial guess for k
traj = []
for alpha in alpha_range:
    (k,) = fsolve(f, k, args=(alpha,))
    print(f'alpha={alpha:.3f} k={k:.3f} ')
    traj.append((alpha, k))

np.savetxt('traj_approximate.txt', traj)