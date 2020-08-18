# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014-2018) James Kermode, King's College London
#                       Lars Pastewka, University of Freiburg
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

from __future__ import division

import os

import sys

import time

import numpy as np

from scipy.special import factorial2

import ase

from ase.calculators.calculator import Calculator

from matscipy.neighbours import neighbour_list, first_neighbours

###

class IPL():
    """
    Functional form for a inverse-power-law potential with an exponent of 10.
    Energy, its first, second and third derivative are shifted to zero at cutoff.

    Parameters
    ----------
    epsilon : float
        Energy scale
    cutoff : float
        Cutoff for the pair-interaction
    minSize : float 
        Minimal size of a particle. Lower bound of distribtuion
    maxSize : float 
        Maximal size of a particle. Upper bound of distribtuion
    na : float 
        Non-additivity paramter for pair sizes
    q : int
        Smooth the potential up to the q-th derivative

    Reference:
    ----------
        E. Lerner, Journal of Non-Crystalline Solids, 522, 119570.
    """
    
    def __init__(self, epsilon, cutoff, na, minSize, maxSize, q):
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.minSize = minSize
        self.maxSize = maxSize
        self.na = na
        self.q = q
        self.coeffs = []
        for index in range(0,q+1):
            first_expr = np.power(-1, index+1)/(factorial2(2*q-2*index, exact=True)*factorial2(2*index, exact=True))
            second_expr = factorial2(10+2*q, exact=True)/(factorial2(10-2)*(10+2*index))
            third_expr = np.power(cutoff, -(10+2*index))
            self.coeffs.append(first_expr*second_expr*third_expr)

    def __call__(self, r, ijsize):
        """
        Return function value (potential energy)
        """
        ipl = self.epsilon*(np.power(ijsize,10)/np.power(r,10) + self.coeffs[0])
        dipl = self.epsilon*self.coeffs[1]*np.power(r,2)/np.power(ijsize,2)
        ddipl = self.epsilon*self.coeffs[2]*np.power(r, 4)/np.power(ijsize,4)
        dddipl = self.epsilon*self.coeffs[3]*np.power(r,6)/np.power(ijsize,6)
        
        return ipl + dipl + ddipl + dddipl

    def mix_sizes(self, isize, jsize):
        """
        Nonadditive interaction rule for the cross size of particles i and j. 
        """
        return 0.5*(isize+jsize)*(1 - self.na * np.absolute(isize-jsize))

    def get_cutoff(self):
        """
        Return the cutoff.
        """
        return self.cutoff

    def get_coeffs(self):
        """
        Return the smoothing coefficients of the potential.
        """
        return self.coeffs

    def get_maxSize(self):
        """
        Return the maximal size of a particle (=Upper boundary of distribution)
        """
        return self.maxSize

    def get_minSize(self):
        """
        Return the minimal size of a particle (=Lower boundary of distribution)
        """
        return self.minSize

    def first_derivative(self, r, ijsize):
        """
        Return first derivative 
        """
        ipl = -10*self.epsilon*np.power(ijsize,10)/np.power(r,11)
        dipl = 2*self.epsilon*self.coeffs[1]*r/np.power(ijsize,2)
        ddipl = 4*self.epsilon*self.coeffs[2]*np.power(r,3)/np.power(ijsize,4)
        dddipl = 6*self.epsilon*self.coeffs[3]*np.power(r,5)/np.power(ijsize,6)

        return ipl + dipl + ddipl + dddipl

    def second_derivative(self, r, ijsize):
        """
        Return second derivative 
        """
        ipl = 110*self.epsilon*np.power(ijsize,10)/np.power(r,12)
        dipl = 2*self.epsilon*self.coeffs[1]/np.power(ijsize,2)
        ddipl = 12*self.epsilon*self.coeffs[2]*np.power(r,2)/np.power(ijsize,4)
        dddipl = 30*self.epsilon*self.coeffs[3]*np.power(r,4)/np.power(ijsize,6)

        return ipl + dipl + ddipl + dddipl 
        
    def derivative(self, n=1):
        if n == 1:
            return self.first_derivative
        elif n == 2:
            return self.second_derivative
        else:
            raise ValueError(
                "Don't know how to compute {}-th derivative.".format(n))     	

###

class Polydisperse(Calculator):
    implemented_properties = ["energy", "stress", "forces"]
    default_parameters = {}
    name = "Polydisperse"

    def __init__(self, f, cutoff=None):
        Calculator.__init__(self)
        self.f = f

        self.dict = f.get_cutoff()
        self.df = f.derivative(1)
        self.df2 = f.derivative(2)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)
        size = self.atoms.get_charges()

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list("ijDd", self.atoms, self.f.get_maxSize()*self.f.get_cutoff())
        ijsize = self.f.mix_sizes(size[i_n], size[j_n])

        e_n = np.zeros_like(abs_dr_n)
        de_n = np.zeros_like(abs_dr_n)

        mask = abs_dr_n <= self.f.get_cutoff()*ijsize
        e_n[mask] = self.f(abs_dr_n[mask], ijsize[mask])
        de_n[mask] = self.df(abs_dr_n[mask], ijsize[mask])

        epot = 0.5*np.sum(e_n)

        # Forces
        df_nc = -0.5*de_n.reshape(-1,1)*dr_nc/abs_dr_n.reshape(-1,1)

        # Sum for each atom
        fx_i = np.bincount(j_n, weights=df_nc[:, 0], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:, 0], minlength=nat) 
        fy_i = np.bincount(j_n, weights=df_nc[:, 1], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:, 1], minlength=nat) 
        fz_i = np.bincount(j_n, weights=df_nc[:, 2], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:, 2], minlength=nat) 

        # Virial 
        virial_v = -np.array([dr_nc[:, 0]*df_nc[:, 0],               # xx
                              dr_nc[:, 1]*df_nc[:, 1],               # yy
                              dr_nc[:, 2]*df_nc[:, 2],               # zz
                              dr_nc[:, 1]*df_nc[:, 2],               # yz
                              dr_nc[:, 0]*df_nc[:, 2],               # xz
                              dr_nc[:, 0]*df_nc[:, 1]]).sum(axis=1)  # xy

        self.results = {'energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': np.transpose([fx_i, fy_i, fz_i])}


