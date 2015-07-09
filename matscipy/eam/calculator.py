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

from scipy.interpolate import InterpolatedUnivariateSpline

from matscipy.eam.io import read_eam_alloy
from matscipy.neighbours import neighbour_list

###

class EAM(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {}
    name = 'CheckpointCalculator'
       
    def __init__(self, fn):
        Calculator.__init__(self)        
        source, parameters, F, f, rep = read_eam_alloy(fn)
        atoms, self.atnums, atomic_masses, lattice_constants, \
            crystal_structure, nF, nf, dF, df, \
            self.cutoff = parameters

        self.atnum_to_index = -np.ones(np.max(self.atnums)+1, dtype=int)
        self.atnum_to_index[self.atnums] = \
            np.arange(len(self.atnums))

        # Create spline interpolation
        self.F = [InterpolatedUnivariateSpline(np.arange(len(x))*dF, x)
                  for x in F]
        self.f = [InterpolatedUnivariateSpline(np.arange(len(x))*df, x)
                  for x in f]
        self.rep = [[InterpolatedUnivariateSpline(np.arange(len(x))*df, x)
                     for x in y]
                    for y in rep]

        # Derivative of spline interpolation
        self.dF = [x.derivative() for x in self.F]
        self.df = [x.derivative() for x in self.f]
        self.drep = [[x.derivative() for x in y] for y in self.rep]

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)
        atnums = self.atoms.numbers

        atnums_in_system = set(atnums)
        for atnum in atnums_in_system:
            if atnum not in self.atnums:
                raise RuntimeError('Element with atomic number {} found, but '
                                   'this atomic number has no EAM '
                                   'parameterization'.format(atnum))

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', self.atoms,
                                                   self.cutoff)

        # Density
        f_n = np.zeros_like(abs_dr_n)
        df_n = np.zeros_like(abs_dr_n)
        for atidx, atnum in enumerate(self.atnums):
            f = self.f[atidx]
            df = self.df[atidx]
            mask = atnums[j_n]==atnum
            if mask.sum() > 0:
                f_n[mask] = f(abs_dr_n[mask])
                df_n[mask] = df(abs_dr_n[mask])

        density_i = np.bincount(i_n, weights=f_n, minlength=nat)

        # Repulsion
        rep_n = np.zeros_like(abs_dr_n)
        drep_n = np.zeros_like(abs_dr_n)
        for atidx1, atnum1 in enumerate(self.atnums):
            for atidx2, atnum2 in enumerate(self.atnums):
                rep = self.rep[atidx1][atidx2]
                drep = self.drep[atidx1][atidx2]
                mask = np.logical_and(atnums[i_n]==atnum1, atnums[j_n]==atnum2)
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

        self.results = {'energy': epot,
                        'forces': np.transpose([fx_i, fy_i, fz_i])}
