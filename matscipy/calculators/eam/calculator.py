# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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
# ======================================================================

"""
Embedded-atom method potential.
"""

import os

import numpy as np

import ase
from ase.calculators.calculator import Calculator

try:
    from scipy.interpolate import InterpolatedUnivariateSpline
except:
    InterpolatedUnivariateSpline = None

from matscipy.calculators.eam.io import read_eam
from matscipy.neighbours import neighbour_list

###

def _make_splines(dx, y):
    if len(np.asarray(y).shape) > 1:
        return [_make_splines(dx, yy) for yy in y]
    else:
        return InterpolatedUnivariateSpline(np.arange(len(y))*dx, y)

def _make_derivative(x):
    if type(x) == list:
        return [_make_derivative(xx) for xx in x]
    else:
        return x.derivative()

###

class EAM(Calculator):
    implemented_properties = ['energy', 'stress', 'forces']
    default_parameters = {}
    name = 'EAM'
       
    def __init__(self, fn=None, atomic_numbers=None, F=None, f=None, rep=None,
                 cutoff=None, kind='eam/alloy'):
        Calculator.__init__(self)
        if fn is not None:
            source, parameters, F, f, rep = read_eam(fn, kind=kind)
            self.atnums = parameters.atomic_numbers
            self.cutoff = parameters.cutoff
            dr = parameters.distance_grid_spacing
            dF = parameters.density_grid_spacing

            # Create spline interpolation
            self.F = _make_splines(dF, F)
            self.f = _make_splines(dr, f)
            self.rep = _make_splines(dr, rep)
        else:
            self.atnums = atomic_numbers
            self.F = F
            self.f = f
            self.rep = rep
            self.cutoff = cutoff

        self.atnum_to_index = -np.ones(np.max(self.atnums)+1, dtype=int)
        self.atnum_to_index[self.atnums] = \
            np.arange(len(self.atnums))

        # Derivative of spline interpolation
        self.dF = _make_derivative(self.F)
        self.df = _make_derivative(self.f)
        self.drep = _make_derivative(self.rep)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)
        atnums = self.atoms.numbers

        atnums_in_system = set(atnums)
        for atnum in atnums_in_system:
            if atnum not in self.atnums:
                raise RuntimeError('Element with atomic number {} found, but '
                                   'this atomic number has no EAM '
                                   'parametrization'.format(atnum))

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', self.atoms,
                                                   self.cutoff)

        # Density
        f_n = np.zeros_like(abs_dr_n)
        df_n = np.zeros_like(abs_dr_n)
        for atidx1, atnum1 in enumerate(self.atnums):
            f1 = self.f[atidx1]
            df1 = self.df[atidx1]
            mask1 = atnums[j_n]==atnum1
            if mask1.sum() > 0:
                if type(f1) == list:
                    for atidx2, atnum2 in enumerate(self.atnums):
                        f = f1[atidx2]
                        df = df1[atidx2]
                        mask = np.logical_and(mask1, atnums[i_n]==atnum2)
                        if mask.sum() > 0:
                            f_n[mask] = f(abs_dr_n[mask])
                            df_n[mask] = df(abs_dr_n[mask])
                else:
                    f_n[mask1] = f1(abs_dr_n[mask1])
                    df_n[mask1] = df1(abs_dr_n[mask1])

        density_i = np.bincount(i_n, weights=f_n, minlength=nat)

        # Repulsion
        rep_n = np.zeros_like(abs_dr_n)
        drep_n = np.zeros_like(abs_dr_n)
        for atidx1, atnum1 in enumerate(self.atnums):
            rep1 = self.rep[atidx1]
            drep1 = self.drep[atidx1]
            mask1 = atnums[i_n]==atnum1
            if mask1.sum() > 0:
                for atidx2, atnum2 in enumerate(self.atnums):
                    rep = rep1[atidx2]
                    drep = drep1[atidx2]
                    mask = np.logical_and(mask1, atnums[j_n]==atnum2)
                    if mask.sum() > 0:
                        r = rep(abs_dr_n[mask])/abs_dr_n[mask]
                        rep_n[mask] = r
                        drep_n[mask] = (drep(abs_dr_n[mask])-r)/abs_dr_n[mask]

        # Energy
        epot = 0.5*np.sum(rep_n)
        demb_i = np.zeros(len(self.atoms))
        for atidx, atnum in enumerate(self.atnums):
            F = self.F[atidx]
            dF = self.dF[atidx]
            mask = atnums==atnum
            if mask.sum() > 0:
                epot += np.sum(F(density_i[mask]))
                demb_i[mask] += dF(density_i[mask])

        # Forces
        df_nc = -0.5*((demb_i[i_n]+demb_i[j_n])*df_n+drep_n).reshape(-1,1)*dr_nc/abs_dr_n.reshape(-1,1)

        # Sum for each atom
        fx_i = np.bincount(j_n, weights=df_nc[:,0], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,0], minlength=nat)
        fy_i = np.bincount(j_n, weights=df_nc[:,1], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,1], minlength=nat)
        fz_i = np.bincount(j_n, weights=df_nc[:,2], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,2], minlength=nat)

        # Virial
        virial_v = -np.array([dr_nc[:,0]*df_nc[:,0],               # xx
                              dr_nc[:,1]*df_nc[:,1],               # yy
                              dr_nc[:,2]*df_nc[:,2],               # zz
                              dr_nc[:,1]*df_nc[:,2],               # yz
                              dr_nc[:,0]*df_nc[:,2],               # xz
                              dr_nc[:,0]*df_nc[:,1]]).sum(axis=1)  # xy

        self.results = {'energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': np.transpose([fx_i, fy_i, fz_i])}
