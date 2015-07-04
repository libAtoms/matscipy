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
       
    def __init__(self, fn, kind='linear'):
        Calculator.__init__(self)        
        source, parameters, F, f, rep = read_eam_alloy(fn)
        atoms, self.atomic_numbers, atomic_masses, lattice_constants, \
            crystal_structure, nF, nf, dF, df, \
            self.cutoff = parameters

        # Create spline interpolation
        self.F = [[InterpolatedUnivariateSpline(np.arange(len(x))*dF, x)
                   for x in y]
                  for y in F]
        self.f = [[InterpolatedUnivariateSpline(np.arange(len(x))*df, x)
                   for x in y]
                  for y in f]
        self.rep = [[InterpolatedUnivariateSpline(np.arange(len(x))*df, x)
                     for x in y]
                    for y in rep]

        # Derivative of spline interpolation
        self.dF = [[x.derivative() for x in y] for y in self.F]
        self.df = [[x.derivative() for x in y] for y in self.f]
        self.drep = [[x.derivative() for x in y] for y in self.rep]

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', self.atoms,
                                                   self.cutoff)

        f = self.f[0][0]
        F = self.F[0][0]
        rep = self.rep[0][0]

        df = self.df[0][0]
        dF = self.dF[0][0]
        drep = self.drep[0][0]

        # Density
        f_n = f(abs_dr_n)
        density_i = np.bincount(i_n, weights=f_n, minlength=nat)

        # Repulsion
        rep_n = rep(abs_dr_n)/abs_dr_n

        # Energy
        epot = np.sum(F(density_i)) + 0.5*np.sum(rep_n)

        # Forces
        df_n = df(abs_dr_n)
        demb_i = dF(density_i)
        drep_n = (drep(abs_dr_n)-rep_n)/abs_dr_n
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
