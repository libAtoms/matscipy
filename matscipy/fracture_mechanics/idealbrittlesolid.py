#
# Copyright 2015, 2017, 2020-2021 Lars Pastewka (U. Freiburg)
#           2014-2015, 2017-2018 James Kermode (Warwick U.)
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
import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.geometry import cellpar_to_cell

from matscipy.neighbours import neighbour_list
from matscipy.fracture_mechanics.crack import (ConstantStrainRate,
                                               get_strain)

def triangular_lattice_slab(a, n, m):
    # primitive unit cell
    ## a = Atoms('H', [(0, 0, 0)],
    ##          cell=cellpar_to_cell([a, a, 10*a, 90, 90, 120]),
    ##          pbc=[True, True, False])

    # cubic unit cell
    c = 10*a
    a = Atoms('H2',[(0, 0, c/2),
                     (0.5*a, np.sqrt(3)*a/2, c/2)],
              cell=[[a, 0, 0],
                    [0, np.sqrt(3)*a, 0],
                    [0, 0, c]],
              pbc=[True, True, True])
    # we use unit masses
    a.set_masses([1]*len(a))
    return a * (n, m, 1)


def find_triangles_2d(atoms, cutoff, minangle=30*np.pi/180, maxangle=120*np.pi/180,
                      xdim=0, ydim=1):
    """
    Return a list of all triangles of a triangular lattice sitting in the x-y
    plane.
    """
    # Contains atom indices that border the triangle
    corner1 = []
    corner2 = []
    corner3 = []
    # Find triangles
    i, j, D = neighbour_list('ijD', atoms, cutoff)
    coord = np.bincount(i)
    for k in range(len(atoms)):
        firstn = np.searchsorted(i, k, side='left')
        lastn = np.searchsorted(i, k, side='right')
        # Sort six neighbors by angle
        angles = np.arctan2(D[firstn:lastn, xdim], D[firstn:lastn, ydim])
        s = np.argsort(angles)
        # Only pick triangles with angles between min and max angle
        trangles = (angles[np.roll(s, -1)]-angles[s]) % (2*np.pi)
        m = (trangles > minangle) & (trangles < maxangle)
        # Add corners of triangle to lists
        corner1 += list(np.array([k]*(lastn-firstn))[m])
        corner2 += list(j[firstn:lastn][s][m])
        corner3 += list(j[firstn:lastn][np.roll(s, -1)][m])
    # Sort corners
    corner1, corner2, corner3 = np.sort([corner1, corner2, corner3], axis=0)
    # Remove duplicate triangles
    uniqueid = corner3+len(atoms)*(corner2+len(atoms)*corner1)
    _, s = np.unique(uniqueid, return_index=True)
    return corner1[s], corner2[s], corner3[s]


class IdealBrittleSolid(Calculator):
    """
    Implementation of force field for an ideal brittle solid

    Described in Marder, Int. J. Fract. 130, 517-555 (2004)
    """

    implemented_properties = ['energy', 'free_energy', 'energies', 'stress',
                              'forces']

    default_parameters = {'a': 1.0, # lattice constant
                          'rc': 1.01, # cutoff
                          'k': 1.0, # spring constant
                          'beta': 0.01, # Kelvin dissipation
                          'b': 0.01, # Stokes dissipation
                          'linear': False # Linearized response
                          }

    def __init__(self, *args, **kwargs):
        Calculator.__init__(self, *args, **kwargs)
        self.crystal_bonds = 0

    def set_reference_crystal(self, crystal):
        rc = self.parameters['rc']
        self.crystal = crystal.copy()
        i = neighbour_list('i', self.crystal, rc)
        self.crystal_bonds = len(i)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        a = self.parameters['a']
        rc = self.parameters['rc']
        k = self.parameters['k']
        beta = self.parameters['beta']
        linear = self.parameters['linear']

        energies = np.zeros(len(atoms))
        forces = np.zeros((len(atoms), 3))
        velocities = (atoms.get_momenta().T/atoms.get_masses()).T

        if not linear:
            i, j, dr, r = neighbour_list('ijDd', atoms, rc)
            dr_hat = (dr.T/r).T
            dv = velocities[j] - velocities[i]

            de = 0.5*k*(r - a)**2 # spring energies
            e = 0.5*de # half goes to each end of spring
            f = (k*(r - a)*dr_hat.T).T + beta*dv

        else:
            # Linearized response
            i, j, D, S = neighbour_list('ijDS', atoms, rc)

            # Displacements
            u = atoms.positions - self.crystal.positions
            # Bond vector taken from reference configuration
            #dr = self.crystal.positions[j] - self.crystal.positions[i] + \
            #    S.dot(self.crystal.cell)
            dr = self.crystal.positions[j] - self.crystal.positions[i] + S.dot(self.crystal.cell)
            r = np.sqrt((dr*dr).sum(axis=-1))

            dr_hat = (dr.T/r).T
            dv = velocities[j] - velocities[i]

            de = 0.5*k*(((u[j] - u[i])*dr_hat).sum(axis=-1))**2 # spring energies
            e = 0.5*de # half goes to each end of spring
            f = ((k*(u[j] - u[i])*dr_hat).sum(axis=-1)*dr_hat.T).T + beta*dv

        energies[:] = np.bincount(i, e, minlength=len(atoms))
        for kk in range(3):
            forces[:, kk] = np.bincount(i, weights=f[:, kk],
                                        minlength=len(atoms))

        energy = energies.sum()

        # add energy 0.5*k*(rc - a)**2 for each broken bond
        if not linear and len(i) < self.crystal_bonds:
            de = 0.5*k*(rc - a)**2
            energy += 0.5*de*(self.crystal_bonds - len(i))

        # Stokes dissipation
        if 'stokes' in atoms.arrays:
            b = atoms.get_array('stokes')
            forces -= (velocities.T*b).T

        self.results = {'energy': energy,
                        'free_energy': energy,
                        'energies': energies,
                        'forces': forces}

        # Virial
        if not linear:
            virial = np.zeros(6)
            if len(i) > 0:
                virial = 0.5*np.array([dr[:,0]*f[:,0],               # xx
                                       dr[:,1]*f[:,1],               # yy
                                       dr[:,2]*f[:,2],               # zz
                                       dr[:,1]*f[:,2],               # yz
                                       dr[:,0]*f[:,2],               # xz
                                       dr[:,0]*f[:,1]]).sum(axis=1)  # xy
            self.results['stress'] = virial/atoms.get_volume()


    def get_wave_speeds(self, atoms):
        """
        Return longitudinal, shear and Rayleigh wave speeds
        """

        k = self.parameters['k']
        a = self.parameters['a']
        m = atoms.get_masses()[0]

        ka2_over_m = np.sqrt(k*a**2/m)

        c_l = np.sqrt(9./8.*ka2_over_m)
        c_s = np.sqrt(3./8.*ka2_over_m)
        c_R = 0.563*ka2_over_m

        return c_l, c_s, c_R


    def get_elastic_moduli(self):
        """
        Return Lam\'e constants lambda and mu
        """
        k = self.parameters['k']
        a = self.parameters['a']

        lam = np.sqrt(3.0)/2.0*k/a
        mu = lam
        return lam, mu


    def get_youngs_modulus(self):
        k = self.parameters['k']
        a = self.parameters['a']

        return 5.0*np.sqrt(3.0)/4.0*k/a


    def get_poisson_ratio(self):
        return 0.25


def find_crack_tip(atoms, dt=None, store=True, results=None):
    """
    Return atom at the crack tip and its x-coordinate

    Crack tip is defined to be location of rightmost atom
    whose nearest neighbour is at distance > 2.5*a
    """
    calc = atoms.get_calculator()
    a = calc.parameters['a']
    rc = calc.parameters['rc']

    i = neighbour_list('i', atoms, rc)
    nn = np.bincount(i) # number of nearest neighbours, equal to 6 in bulk

    x = atoms.positions[:, 0]
    y = atoms.positions[:, 1]

    bottom = y.min()
    left = x.min()
    width = x.max() - x.min()
    height = y.max() - y.min()
    old_tip_x = atoms.info.get('tip_x', left + 0.3*width)

    # crack cannot have advanced more than c_R*dt
    if dt is not None:
        cl, ct, cR = calc.get_wave_speeds(atoms)
        tip_max_x = old_tip_x + 10.0*cR*dt # FIXME definition of cR seems wrong, shouldn't need factor of 10 here...
    else:
        tip_max_x = left + 0.8*width

    broken = ((nn != 6) &
              (x > left + 0.2*width) & (x < tip_max_x) &
              (y > bottom + 0.1*height) & (y < bottom + 0.9*height))

    index = atoms.positions[broken, 0].argmax()
    tip_atom = broken.nonzero()[0][index]
    tip_x = atoms.positions[tip_atom, 0]

    strain = get_strain(atoms)
    eps_G = atoms.info['eps_G']
    print('tip_x: %.3f strain: %.4f delta: %.3f' % (tip_x, strain, strain/eps_G))

    if store:
        atoms.info['tip_atom'] = tip_atom
        atoms.info['tip_x'] = tip_x

    if results is not None:
        results.append(tip_x)

    return (tip_atom, tip_x, broken)


def set_initial_velocities(c):
    """
    Initialise a dynamical state by kicking some atoms behind tip
    """
    tip_atom, tip_x, broken = find_crack_tip(c, store=False)

    init_atoms = broken.nonzero()[0][c.positions[broken, 0].argsort()[-8:]]
    upper = list(init_atoms[c.positions[init_atoms, 1] > 0])
    lower = list(init_atoms[c.positions[init_atoms, 1] < 0])

    calc = c.get_calculator()
    cl, ct, cR = calc.get_wave_speeds(c)
    v0 = cl/10.

    v = np.zeros((len(c), 3))
    v[upper, 1] = +v0
    v[lower, 1] = -v0
    c.set_velocities(v)

    print('Setting velocities of upper=%s, lower=%s to +/- %.2f' % (upper, lower, v0))
    return (upper, lower, v0)


def set_constraints(c, a):
    # fix atoms in the top and bottom rows
    top = c.positions[:, 1].max()
    bottom = c.positions[:, 1].min()
    left = c.positions[:, 0].min()
    right = c.positions[:, 0].max()

    fixed_mask = ((abs(c.positions[:, 1] - top) < 0.5*a) |
                  (abs(c.positions[:, 1] - bottom) < 0.5*a))
    fix_atoms = FixAtoms(mask=fixed_mask)

    if 'fix' in c.arrays:
        c.set_array('fix', fixed_mask)
    else:
        c.new_array('fix', fixed_mask)
    print('Fixed %d atoms' % fixed_mask.sum())
    c.set_constraint(fix_atoms)

    # Stokes damping regions at left and right of slab
    stokes = np.zeros(len(c))
    x = c.positions[:, 0]
    stokes[:] = 0.0
    stokes[x < left + 5.0*a] = (1.0 - (x-left)/(5.0*a))[x < left + 5.0*a]
    stokes[x > right - 10.0*a] = (1.0 - (right-x)/(10.0*a))[x > right - 10.0*a]
    if 'stokes' in c.arrays:
        c.set_array('stokes', stokes)
    else:
        c.new_array('stokes', stokes)
    print('Applying Stokes damping to %d atoms' % (stokes != 0.0).sum())




def extend_strip(atoms, a, N, M, vacuum):
    x = atoms.positions[:, 0]
    left = x.min()
    width = x.max() - x.min()

    tip_x = atoms.info['tip_x']
    if tip_x < left + 0.6*width:
        # only need to extend strip when crack gets near end
        return False

    print('tip_x (%.2f) > left + 0.75*width (%.2f)' % (tip_x, left + 0.75*width))

    # extra material for pasting onto end
    a = atoms.get_calculator().parameters['a']
    extra = triangular_lattice_slab(a, M, N)

    # apply uniform strain and append to slab
    strain = get_strain(atoms)

    extra.center(vacuum, axis=1)
    fix = atoms.get_array('fix')
    extra.positions[:, 0] += atoms.positions[fix, 0].max() + a/2.0
    extra.positions[:, 1] -= extra.positions[:, 1].mean()
    extra.positions[:, 1] *= (1.0 + strain)

    print('Adding %d atoms' % len(extra))
    atoms += extra
    atoms.set_constraint([])

    discard = atoms.positions[:, 0].argsort()[:len(extra)]
    print('Discarding %d atoms' % len(discard))
    del atoms[discard]

    return True
