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
    Reference:
    E. Lerner, Journal of Non-Crystalline Solids, 522, 119570.
    """
    
    def __init__(self, epsilon, cutoff, q, na):
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.q = q
        self.na = na
        self.coeffs = smoothing_coeffs(q, cutoff)

    def mix_sizes(self, isize, jsize):
        """
        Nonadditive interaction rule for the cross size of particles i and j. 
        """
        return 0.5*(isize+jsize)*(1 - self.na * np.absolute(isize-jsize))

    def __call__(self, r, ijsize):
        """
        Return function value (potential energy)
        """
        ipl = self.epsilon*(np.power(ijsize,10)/np.power(r,10) + self.coeffs[0])
        dipl = self.epsilon*self.coeffs[1]*np.power(r,2)/np.power(ijsize,2)
        ddipl = self.epsilon*self.coeffs[2]*np.power(r, 4)/np.power(ijsize,4)
        dddipl = self.epsilon*self.coeffs[3]*np.power(r,6)/np.power(ijsize,6)
        
        return ipl + dipl + ddipl + dddipl

    def smoothing_coeffs(self, order, rc):
        """
        Return coefficients which smooth the potential up to the desired order.
        """
        coeffs_n = []
        for index in range(0,order):
            first_expr = np.power(-1, index+1)/(factorial2(2*q-2*index, exact=True)*factorial2(2*index, exact=True))
            second_expr = factorial2(10+2*q, exact=True)/(factorial2(10-2)*(10+2*l))
            third_expr = np.power(rc, -(10+2*l))
            coeffs_n.append(first_expr*second_expr*third_expr)
        return coeffs_n

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

    def first_derivative(self, r, ijsize):
        """
        Return first derivative 
        """
        ipl = -10*self.epsilon*np.power(ijsize,10)/np.power(r,10)
        dipl = 2*self.coeffs[1]*self.epsilon*r/np.power(ijsize,2)
        ddipl = 4*self.coeffs[2]*self.epsilon*np.power(r,3)/np.power(ijsize,4)
        dddipl = 6*self.coeffs[3]*self.epsilon*np.power(r,5)/np.power(ijsize,6)
        return ipl + dipl + ddipl + dddipl

    def second_derivative(self, r, ijsize):
        """
        Return second derivative 
        """
        
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

        self.dict = {x: obj.get_cutoff() for x, obj in f.items()}
        self.df = {x: obj.derivative(1) for x, obj in f.items()}
        self.df2 = {x: obj.derivative(2) for x, obj in f.items()}

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)
        atnums = self.atoms.numbers
        atnums_in_system = set(atnums)
        size = self.atoms.get_charges()

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list("ijDd", self.atoms, self.dict)
        ijsize = self.mix_sizes(size[i_n], size[j_n])

        e_n = np.zeros_like(abs_dr_n)
        de_n = np.zeros_like(abs_dr_n)

        # Mask in order to consider only the atoms which are abs_dr_n <= 1.4*lambdaij
        mask = abs_dr_n <= self.dict*ijsize
        e_n[mask] = self.f(abs_dr_n[mask])
        de_n[mask] = self.df(abs_dr_n[mask])

    epot = 0.5*np.sum(e_n)


