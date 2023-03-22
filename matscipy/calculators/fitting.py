#
# Copyright 2015, 2021 Lars Pastewka (U. Freiburg)
#           2015-2016 Adrien Gola (KIT)
#           2015 gola@cmsserv.iam-cms.kit.edu
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

"""
Helper routines for potential fitting
"""

from math import atanh, sqrt, tanh, isnan
import sys

import numpy as np

import ase
import ase.constraints
import ase.io
import ase.lattice.compounds as compounds
import ase.lattice.cubic as cubic
import ase.lattice.hexagonal as hexagonal
import ase.optimize

from ase.units import GPa,J,m

import scipy
scipy_v = scipy.__version__

if int(scipy_v.split('.')[1]) <= 14 : 
    from scipy.optimize import minimize, leastsq, anneal, brute
else :
    # scipy.optimize.anneal decprecated from version 0.14.0, documentation advise to use scipy.optimize.basinhopping instead
    from scipy.optimize import minimize, leastsq, brute
    from scipy.signal import argrelextrema
    

try:
    from openopt import GLP
    have_openopt = True
except:
    have_openopt = False

###

_logfile = None
Jm2 = 1e23/ase.units.kJ

### Parameters

class Parameters(object):
    """
    Stores a parameter set for fitting purposes.

    In particular, it distinguishes between variable parameters (to be fitted),
    constant parameter and derived parameters, the latter ones having a
    functional dependence on the other parameters.
    """

    __slots__ = [ 'default', 'variable', 'constant', 'derived', 'hidden', 
                  'parameters', 'ranges', 'range_mapping' ]

    def __init__(self, default, constant, derived, ranges={}, hidden=[]):
        """
        Parameters
        ----------
        default : dict
            Dictionary with the default parameter set
        constant : list
            List of parameters that are constant
        derived : dict
            Dictionary with derived parameters and a function to get those
            derived values.
        """
        self.default = default
        self.constant = set(constant)
        self.derived = derived
        self.ranges = ranges
        self.hidden = set(hidden)

        self.parameters = self.default.copy()

        self.range_mapping = False

        self._update_variable()
        self._update_derived()

    def set_range_derived(self):
        self.range_mapping = True
        for key, ( x1, x2 ) in self.ranges.items():
            if x1 > x2:
                raise RuntimeError('Inverted ranges {0}:{1} for parameter {2}.'
                                   .format(x1, x2, key))
            x = self.parameters[key]
            if x <= x1 or x >= x2:
                raise ValueError('Parameter {0} has value {1} which is '
                                 'outside of the bounds {2}:{3}.'
                                 .format(key, x, x1, x2))
            self.parameters[':'+key] = atanh(2*(x-x1)/(x2-x1)-1)

            self.variable += [ ':'+key ]
            self.hidden.add(':'+key)

    def _update_variable(self):
        self.variable = []
        for key in self.default.keys():
            if not ( key in self.constant or key in self.derived or 
                     key in self.ranges ):
                self.variable += [ key ]

    def _update_constant(self):
        self.constant = []
        for key in self.default.keys():
            if not ( key in self.variable or key in self.derived ):
                self.constant += [ key ]

    def _update_derived(self):
        if self.range_mapping:
            for key, ( x1, x2 ) in self.ranges.items():
                self.parameters[key] = x1+\
                    0.5*(x2-x1)*(1+tanh(self.parameters[':'+key]))
        for key, func in self.derived.items():
            self.parameters[key] = func(self.parameters)

    def __len__(self):
        return len(self.variable)

    def set_variable(self, variable):
        self.variable = variable
        self._update_constant()

    def get_variable(self):
        return self.variable

    def set_constant(self, constant):
        self.constant = constant
        self._update_variable()

    def get_constant(self):
        return self.constant

    def set_derived(self, derived):
        self.derived = derived
        self._update_variable()
        self._update_derived()

    def get_derived(self):
        return self.derived

    def set(self, key, value):
        if key in self.derived:
            raise RuntimeError('Cannot set parameter %s since it is a derived '
                               'quantity.' % key)

        self.parameters[key] = value

        self._update_derived()
    __setitem__ = set

    def get(self, key):
        return self.parameters[key]
    __getitem__ = get

    def __getattr__(self, key):
        try:
            return self.get(key)
        except:
            return super(Parameters, self).__getattr__(key)

    def set_dict(self, d):
        for key, value in d.items():
            self.set(key, value)

    def get_dict(self):
        p = self.parameters.copy()
        if self.hidden is not None:
            for key in self.hidden:
                del p[key]
        return p

    def set_array(self, a, keys=None):
        if keys is None:
            keys = self.variable

        for key, value in zip(keys, a):
            self.set(key, value)

    def get_array(self, keys=None):
        if keys is None:
            keys = self.variable

        return [ self.parameters[key] for key in keys ]

    def get_lower_bounds(self, keys=None):
        if keys is None:
            keys = self.variable
        return [ self.ranges[key][0] for key in keys ]

    def get_upper_bounds(self, keys=None):
        if keys is None:
            keys = self.variable
        return [ self.ranges[key][1] for key in keys ]

    def in_range(self, key=None):
        if self.ranges is None:
            return True
        if not key is None:
            if key in self.ranges:
                x1, x2 = self.ranges[key]
                return self.parameters[key] >= x1 and self.parameters[key] <= x2
            else:
                return True

        r = True
        for key, value in self.parameters.items():
            if key in self.ranges:
                x1, x2 = self.ranges[key]
                r = r and self.parameters[key] >= x1 and \
                    self.parameters[key] <= x2
        return r

    def __str__(self):
        s = ''
        for key, value in self.parameters.items():
            if not key.startswith(':'):
                s += '# {0:>24s} = {1}\n'.format(key, value)
        return s

### Fitting superclass

class Fit(object):
    """
    Parameter optimization class.
    """

    __slots__ = [ 'atoms', 'calc', 'cost_history', 'minimal_cost', 'par',
                  'par_at_minimal_cost', 'residuals_history' ]

    def __init__(self, calc, par):
        self.calc = calc
        self.par = par

        self.minimal_cost = 1e40
        self.par_at_minimal_cost = None

        self.cost_history = []
        self.residuals_history = []

    def set_parameters_from_array(self, p):
        self.par.set_array(p)
        self.set_calculator(self.calc(**self.par.get_dict()))

    def set_calculator_class(self, calc):
        self.calc = calc

    def set_parameters(self, par):
        self.par = par
        self.set_calculator(self.calc(**self.par.get_dict()))

    def get_potential_energy(self):
        return self.atoms.get_potential_energy()

    def get_cohesive_energy(self):
        return self.atoms.get_potential_energy()/len(self.atoms)

    def get_square_residuals(self, p=None, log=None):
        r2 = 0.0
        if p is not None:
            self.set_parameters_from_array(p)
        r = np.array(self.get_residuals(log=log))
        self.residuals_history += [ r ]
        return r*r

    def get_cost_function(self, p=None, log=None, store_history=False):
        try:
            c = np.sum(self.get_square_residuals(p, log=log))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print('# Warning: Error computing residuals. Penalizing '
                  'parameters but continuing. Error message: {0}' \
                  .format(e), file=log)
            c = 1e40
        if log is not None:
            print('# Current value of the cost function/residual: {0}' \
                .format(c), file=log)
        if isnan(c):
            c = 1e40
        if store_history:
            if self.cost_history == [] or c < np.min(self.cost_history):
                print('# New minimum of cost function: {0}'.format(c))
                self.minimal_cost = c
                self.par_at_minimal_cost = p
            self.cost_history += [c]
        return c

    def get_residuals_history(self):
        return np.array(self.residuals_history)

    def get_cost_history(self):
        return np.array(self.cost_history)

    def get_parameters_at_minimal_cost(self, log=sys.stdout):
        self.set_parameters_from_array(self.par_at_minimal_cost)
        print('=== PARAMETERS AT MINIMAL COST ===', file=log)
        self.get_cost_function(self.par_at_minimal_cost, log=log)
        return self.par

    def optimize(self, log=sys.stdout, **kwargs):
        self.par.set_range_derived()
        res = minimize(self.get_cost_function, self.par.get_array(),
                       args=(log, True,), **kwargs)
        self.set_parameters_from_array(res.x)
        print('=== HISTORY OF COST FUNCTION ===', file=log)
        print(self.cost_history, file=log)
        print('=== FINAL OPTIMIZED PARAMETER SET ===', file=log)
        final_cost = self.get_cost_function(res.x, log)
        if abs(final_cost - self.minimal_cost) > 1e-6:
            print('# WARNING: Final cost (={0}) is not minimal (={1}). This '
                  'may come from insufficiently converged calculations. Try '
                  'to decrease fmax.'.format(final_cost, self.minimal_cost))
        return self.par

    def optimize_leastsq(self, log=sys.stdout):
        self.par.set_range_derived()
        self.set_parameters_from_array(leastsq(self.get_square_residuals,
                                               self.par.get_array(),
                                               args=(log,))[0])
        return self.par

    def optimize_anneal(self, **kwargs):
        self.set_parameters_from_array(anneal(self.get_cost_function,
                                              self.par.get_array(),
                                              lower=self.par.get_lower_bounds(),
                                              upper=self.par.get_upper_bounds(),
                                              **kwargs))
        return self.par
      

    def optimize_brute(self, **kwargs):
        x0, fval = brute(
            self.get_cost_function,
            ranges=map(tuple,
                       list(np.transpose([self.par.get_lower_bounds(),
                                          self.par.get_upper_bounds()]))),
            **kwargs)
        self.set_parameters_from_array(x0)
        print('=== OPTIMIZED PARAMETER SET ===', file=log)
        self.get_cost_function(x0)
        return self.par

    def optimize_openopt(self, solver='interalg'):
        if not have_openopt:
            raise RuntimeError('OpenOpt not available.')
        p = GLP(self.get_cost_function, self.par.get_array(),
                lb=self.par.get_lower_bounds(),
                ub=self.par.get_upper_bounds())
        r = p.solve(solver)
        print(r, file=log)
        self.set_parameters_from_array(r.xf)
        print('=== OPTIMIZED PARAMETER SET ===', file=log)
        self.get_cost_function(r.xf)
        return self.par

class CombinedFit(Fit):

    __slots__ = [ 'targets' ]

    def __init__(self, calc, par, targets):
        Fit.__init__(self, calc, par)

        self.targets = targets

    def set_parameters_from_array(self, p):
        for target in self.targets:
            target.set_parameters_from_array(p)

    def set_parameters(self, p):
        for target in self.targets:
            target.set_parameters(p)

    def set_calculator_class(self, calc):
        for target in self.targets:
            target.set_calculator_class(calc)

    def get_residuals(self, log=None):
        if log is not None:
            print('', file=log)
            print('# Computing properties for parameter set:', file=log)
            print(self.par, file=log)
            f = open('par.out', 'w')
            print(self.par, file=f)
            f.close()
        r = []
        for target in self.targets:
            r = np.append(r, target.get_residuals(log=log))
        return r

    def get_potential_energy(self):
        raise RuntimeError('get_potential_energy does not make sense for the '
                           'CombinedFit class.')

class RotatingFit(object):

    __slots__ = [ 'targets', 'par' ]

    def __init__(self, par, targets):
        self.par = par

        self.targets = targets


    def optimize(self, pmax=1e-3, mix=None, **kwargs):
        globalv = np.array(self.par.get_variable()).copy()

        dp = 1e6
        while dp > pmax:
            p = np.array(self.par.get_array())

            for target, variable in self.targets:
                if log is not None:
                    print('# ===', target, '===', variable, '===', file=log)
                self.par.set_variable(variable)
                target.set_parameters(self.par)
                self.par = target.optimize(**kwargs)

            self.par.set_variable(globalv)

            cp = np.array(self.par.get_array())
            dp = cp - p
            dp = sqrt(np.sum(dp*dp))

            if mix is not None:
                self.par.set_array(mix*cp + (1-mix)*p)

        self.par.set_variable(globalv)

        return self.par


### Generic penalty function

class Penalty(Fit):

    __slots__ = [ 'func' ]

    def __init__(self, calc, par, func): 
        Fit.__init__(self, calc, par)
        self.func = func


    def set_calculator(self, calc):
        """
        Set the calculator
        """
        pass


    def get_residuals(self, log=None):
        return self.func(self.par, log=log)


### Single point

class FitSinglePoint(Fit):

    __slots__ = [ 'atoms', 'energy', 'forces', 'stress', 'w_energy', 'w_forces',
                  'w_stress' ]

    def __init__(self, calc, par, atoms, w_energy=None, w_forces=None,
                 w_stress=None): 
        Fit.__init__(self, calc, par)
        self.original_atoms = atoms
        self.w_energy = w_energy
        self.w_forces = w_forces
        self.w_stress = w_stress

        self.energy = self.atoms.get_potential_energy()
        self.forces = self.atoms.get_forces().copy()
        self.stress = self.atoms.get_stress().copy()

        self.atoms = None

    def set_calculator(self, calc):
        """
        Set the calculator
        """
        self.atoms = self.original_atoms.copy()
        self.atoms.set_calculator(calc)
        self.atoms.get_potential_energy()

    def get_residuals(self, log=None):
        r = []
        w = []
        if self.w_energy is not None:
            cr = [ self.w_energy*(
                    self.atoms.get_potential_energy()-self.energy
                    )/self.energy ]
            r += cr
            w += [ self.w_energy ]
        if self.w_forces is not None:
            cr = list(
                (self.w_forces*(
                        self.atoms.get_forces() - self.forces
                        )/self.forces).flatten()
                )
            r += cr
            w += list(self.w_forces*np.ones_like(self.forces).flatten())
        if self.w_stress is not None:
            cr = list(
                (self.w_stress*(
                        self.atoms.get_stress() - self.stress
                        )/self.stress).flatten()
                )
            r += cr
            w += list(self.w_stress*np.ones_like(self.stress).flatten())
        return r


### Specific structures

class FitDimer(Fit):

    __slots__ = [ 'D0', 'fmax', 'r0', 'w_D0', 'w_r0' ]

    def __init__(self, calc, par, els, D0, r0,
                 w_D0=1.0, w_r0=1.0,
                 vacuum=10.0, fmax=1e-6):
        Fit.__init__(self, calc, par)

        self.els = els

        self.D0 = D0
        self.r0 = r0

        self.w_D0 = sqrt(w_D0)/self.D0
        self.w_r0 = sqrt(w_r0)/self.r0

        self.calc = calc
        self.par = par

        self.fmax = fmax

        if type(self.els) == str:
            self.els = 2*[self.els]

        self.atoms = None

    def new_dimer(self):
        self.atoms = ase.Atoms(
            self.els,
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, r0 ]],
            pbc = False)
        self.atoms.center(vacuum=vacuum)

    def set_calculator(self, calc):
        """
        Set the calculator, and relax the structure to its ground-state.
        """
        self.new_dimer()
        self.atoms.set_calculator(calc)
        ase.optimize.FIRE(self.atoms, logfile=_logfile).run(fmax=self.fmax,steps=10000)

    def get_distance(self):
        return self.atoms.get_distance(0, 1)

    def get_residuals(self, log=None):
        D0 = self.atoms.get_potential_energy()
        r0 = self.atoms.get_distance(0, 1)

        r_D0 = self.w_D0*(D0+self.D0)
        r_r0 = self.w_r0*(r0-self.r0)

        if log is not None:
            print('# %20s D0  = %20.10f eV    (%20.10f eV)    - %20.10f' \
                % ( 'Dimer', D0, -self.D0, r_D0 ), file=log)
            print('# %20s r0  = %20.10f A     (%20.10f A)     - %20.10f' \
                % ( '', r0, self.r0, r_r0 ), file=log)

        return r_D0, r_r0

class FitCubicCrystal(Fit):

    __slots__ = [ 'a0', 'calc', 'crystal', 'Ec', 'fmax', 'par', 'w_a0', 'w_Ec' ]

    def __init__(self, calc, par, els,
                 Ec, a0,
                 B=None, C11=None, C12=None, C44=None, Cp=None,SFE=None,
                 w_Ec=1.0, w_a0=1.0,
                 w_B=1.0, w_C11=1.0, w_C12=1.0, w_C44=1.0, w_Cp=1.0,w_SFE=1.0,
                 fmax=1e-6, eps=0.001,
                 ecoh_ref=None,
                 size=[1,1,1]):
        Fit.__init__(self, calc, par)

        self.els = els

        self.a0 = a0
        self.Ec = Ec
        
        self.SFE = SFE

        self.B = B
        self.C11 = C11
        self.C12 = C12
        self.C44 = C44
        self.Cp = Cp

        self.ecoh_ref = ecoh_ref

        self.w_a0 = sqrt(w_a0)/self.a0
        self.w_Ec = sqrt(w_Ec)/self.Ec

        if self.SFE is not None:
            self.w_SFE = sqrt(w_SFE)/self.SFE
        if self.B is not None:
            self.w_B = sqrt(w_B)/self.B
        if self.C11 is not None:
            self.w_C11 = sqrt(w_C11)/self.C11
        if self.C12 is not None:
            self.w_C12 = sqrt(w_C12)/self.C12
        if self.C44 is not None:
            self.w_C44 = sqrt(w_C44)/self.C44
        if self.Cp is not None:
            self.w_Cp = sqrt(w_Cp)/self.Cp

        self.size = size

        self.fmax = fmax
        self.eps = eps

        self.atoms = None

    def new_bulk(self):
        self.unitcell = self.crystal(
            self.els,
            latticeconstant  = self.a0,
            size             = [1, 1, 1]
            )
        self.atoms = self.unitcell.copy()
        self.atoms *= self.size
        self.atoms.translate([0.1, 0.1, 0.1])

    def set_calculator(self, calc):
        self.new_bulk()
        self.atoms.set_calculator(calc)
        ase.optimize.FIRE(
            ase.constraints.StrainFilter(self.atoms, mask=[1,1,1,0,0,0]),
            logfile=_logfile).run(fmax=self.fmax,steps=10000)
        a0 = self.get_lattice_constant()
        self.supercell = self.crystal(
            self.els,
            latticeconstant  = a0,
            size             = [3, 3, 6],
            directions=[[1,1,2],[-1,1,0],[-1,-1,1]],
            pbc=(1,1,0)
            )
        self.supercell.translate([0.1, 0.1, 0.1])
        self.supercell.set_calculator(calc)

    def get_lattice_constant(self):
        return np.sum(self.atoms.get_cell().diagonal())/np.sum(self.size)
      
    def get_SFE(self):
        E0 = self.supercell.get_potential_energy()/J*1e3
        S0 = self.supercell.get_cell()[0,0]*self.supercell.get_cell()[1,1]/m/m
        pos = self.supercell.get_positions()
        x,y=[],[]
        for i in range(10):
            pos1= np.copy(pos)
            pos1[:,0][pos[:,2]>self.supercell.get_cell()[2,2]/2-2]-=(self.supercell.get_cell()[0,0]/3)*0.08+(self.supercell.get_cell()[0,0]/3)*(1./50)*i
            self.supercell.set_positions(pos1)
            Es = self.supercell.get_potential_energy()/J*1e3
            x.append(0.08+(1./50)*i)
            y.append((Es/S0)-(E0/S0))
        GSF_fit = scipy.interpolate.InterpolatedUnivariateSpline(x,y)
        x_fit = np.linspace(0.08,0.08+(9./50),50)
        mins = argrelextrema(GSF_fit(x_fit),np.less)
        x_mins = x_fit[mins[0]]
        y_mins = GSF_fit(x_fit)[mins[0]]
        return y_mins[0]

    def get_C11(self):
        sxx0, syy0, szz0, syz0, szx0, sxy0  = self.atoms.get_stress()

        cell = self.atoms.get_cell()
        T = np.diag( [ self.eps, 0.0, 0.0 ] )
        self.atoms.set_cell( np.dot(np.eye(3)+T, cell), scale_atoms=True )
        sxx11, syy11, szz11, syz11, szx11, sxy11  = self.atoms.get_stress()
        self.atoms.set_cell(cell, scale_atoms=True)

        return (sxx11-sxx0)/self.eps

    def get_Cp(self):
        sxx0, syy0, szz0, syz0, szx0, sxy0  = self.atoms.get_stress()

        cell = self.atoms.get_cell()
        T = np.diag( [ self.eps, -self.eps, 0.0 ] )
        self.atoms.set_cell( np.dot(np.eye(3)+T, cell), scale_atoms=True )
        sxx12, syy12, szz12, syz12, szx12, sxy12  = self.atoms.get_stress()
        self.atoms.set_cell(cell, scale_atoms=True)

        return ((sxx12-sxx0)-(syy12-syy0))/(4*self.eps)

    def get_C44(self):
        sxx0, syy0, szz0, syz0, szx0, sxy0  = self.atoms.get_stress()

        cell = self.atoms.get_cell()
        T = np.array( [ [ 0.0, 0.5*self.eps, 0.5*self.eps ],
                        [ 0.5*self.eps, 0.0, 0.5*self.eps ],
                        [ 0.5*self.eps, 0.5*self.eps, 0.0 ] ] )
        self.atoms.set_cell( np.dot(np.eye(3)+T, cell), scale_atoms=True )
        sxx44, syy44, szz44, syz44, szx44, sxy44  = self.atoms.get_stress()
        self.atoms.set_cell(cell, scale_atoms=True)

        return (syz44+szx44+sxy44-syz0-szx0-sxy0)/(3*self.eps)
        
    def get_residuals(self, log=None):
        Ec = self.get_potential_energy()
        a0 = self.get_lattice_constant()

        if self.ecoh_ref is not None:
            syms = np.array(self.atoms.get_chemical_symbols())
            for el in set(syms):
                Ec -= (syms==el).sum()*self.ecoh_ref[el]

        Ec /= len(self.atoms)
        r_Ec = self.w_Ec*( Ec - self.Ec )
        r_a0 = self.w_a0*( a0 - self.a0 )

        if log is not None:
            print('# %20s Ec  = %20.10f eV/at.    (%20.10f eV/at.)    - %20.10f' \
                % ( '%s (%s)' % (self.unitcell.get_chemical_formula(),
                                 self.crystalstr), Ec, self.Ec, r_Ec ))
            print('# %20s a0  = %20.10f A     (%20.10f A)     - %20.10f' \
                % ( '', a0, self.a0, r_a0 ))

        r = [ r_Ec, r_a0 ]

        if self.SFE is not None:
            SFE = self.get_SFE()
        if self.B is not None or self.C11 is not None or self.C12 is not None:
            C11 = self.get_C11()
        if self.B is not None or self.Cp is not None or self.C12 is not None:
            Cp = self.get_Cp()
        if self.C44 is not None:
            C44 = self.get_C44()

        if self.SFE is not None:
            if SFE < 0:
                r_SFE = self.w_SFE*( SFE - self.SFE )*1000
            else:  
                r_SFE = self.w_SFE*( SFE - self.SFE )
            r += [ r_SFE ]
            if log is not None:
                print('# %20s SFE = %20.10f mJ/m**2   (%20.10f mJ/m**2)   - %20.10f' \
                    % ( '', SFE, self.SFE, r_SFE ))
        if self.B is not None:
            r_B = self.w_B*( (3*C11-4*Cp)/3 - self.B )
            r += [ r_B ]
            if log is not None:
                print('# %20s B   = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', (3*C11-4*Cp)/3/GPa, self.B/GPa, r_B ))
        if self.C11 is not None:
            r_C11 = self.w_C11*( C11 - self.C11 )
            r += [ r_C11 ]
            if log is not None:
                print('# %20s C11 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', C11/GPa, self.C11/GPa, r_C11 ))
        if self.C12 is not None:
            r_C12 = self.w_C12*( C11-2*Cp - self.C12 )
            r += [ r_C12 ]
            if log is not None:
                print('# %20s C12 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', (C11-2*Cp)/GPa, self.C12/GPa, r_C12 ))
        if self.C44 is not None:
            r_C44 = self.w_C44*( C44 - self.C44 )
            r += [ r_C44 ]
            if log is not None:
                print('# %20s C44 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', C44/GPa, self.C44/GPa, r_C44 ))
        if self.Cp is not None:
            r_Cp = self.w_Cp*( Cp - self.Cp )
            r += [ r_Cp ]
            if log is not None:
                print('# %20s Cp  = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', Cp/GPa, self.Cp/GPa, r_Cp ))
        
        return r
      
class FitTetragonalCrystal(Fit):

    __slots__ = [ 'a0','c0', 'calc', 'crystal', 'Ec', 'fmax', 'par', 'w_a0','w_c0', 'w_Ec' ]

    def __init__(self, calc, par, els,
                 Ec, a0, c0, c_a=None,
                 B=None, C11=None, C12=None,C13=None, C33=None, C44=None, C66=None,SFE=None,
                 w_Ec=1.0, w_a0=1.0,w_c0=1.0, w_c_a=1.0,
                 w_B=1.0, w_C11=1.0, w_C12=1.0,w_C13=1.0,w_C33=1.0,w_C44=1.0,w_C66=1.0, w_Cp=1.0,w_SFE=None,
                 fmax=1e-6, eps=0.001,
                 ecoh_ref=None,
                 size=[1,1,1]):
        Fit.__init__(self, calc, par)

        self.els = els

        self.a0 = a0
        self.c0 = c0
        self.Ec = Ec
        self.c_a = c_a

        self.SFE = SFE
        
        self.B = B
        self.C11 = C11
        self.C12 = C12
        self.C13 = C13
        self.C44 = C44
        self.C33 = C33
        self.C66 = C66


        self.ecoh_ref = ecoh_ref

        self.w_a0 = sqrt(w_a0)/self.a0
        self.w_c0 = sqrt(w_c0)/self.c0
        
        if self.Ec is not None:
            self.w_Ec = sqrt(w_Ec)/self.Ec
        if self.c_a is not None:
            self.w_c_a = sqrt(w_c_a)/self.c_a
        if self.SFE is not None:
            self.w_SFE = sqrt(w_SFE)/self.SFE
        if self.B is not None:
            self.w_B = sqrt(w_B)/self.B
        if self.C11 is not None:
            self.w_C11 = sqrt(w_C11)/self.C11
        if self.C12 is not None:
            self.w_C12 = sqrt(w_C12)/self.C12
        if self.C13 is not None:
            self.w_C13 = sqrt(w_C13)/self.C13
        if self.C33 is not None:
            self.w_C33 = sqrt(w_C33)/self.C33
        if self.C44 is not None:
            self.w_C44 = sqrt(w_C44)/self.C44
        if self.C66 is not None:
            self.w_C66 = sqrt(w_C66)/self.C66

        self.size = size

        self.fmax = fmax
        self.eps = eps

        self.atoms = None
    
    def new_bulk(self):
        self.unitcell = self.crystal(
            self.els,
            latticeconstant  = [self.a0,self.c0],
            size             = [1, 1, 1]
            )
        self.atoms = self.unitcell.copy()
        self.atoms *= self.size
        self.atoms.translate([0.1, 0.1, 0.1])

    def set_calculator(self, calc):
        self.new_bulk()
        self.atoms.set_calculator(calc)
        ase.optimize.FIRE(
            ase.constraints.StrainFilter(self.atoms, mask=[1,1,1,1,1,1]),
            logfile=_logfile).run(fmax=self.fmax,steps=10000)
        a0,c0 = self.get_lattice_constant()
        self.supercell = self.crystal(
            self.els,
            latticeconstant  = [a0,c0],
            size             = [3, 3, 6],
            directions=[[1,1,2],[-1,1,0],[-1,-1,1]],
            pbc=(1,1,0)
            )
        self.supercell.translate([0.1, 0.1, 0.1])
        self.supercell.set_calculator(calc)

    def get_SFE(self):
        E0 = self.supercell.get_potential_energy()/J*1e3
        S0 = self.supercell.get_cell()[0,0]*self.supercell.get_cell()[1,1]/m/m
        pos = self.supercell.get_positions()
        x,y=[],[]
        for i in range(10):
            pos1= np.copy(pos)
            pos1[:,0][pos[:,2]>self.supercell.get_cell()[2,2]/2-2]-=(self.supercell.get_cell()[0,0]/3)*0.05+(self.supercell.get_cell()[0,0]/3)*(1./50)*i
            self.supercell.set_positions(pos1)
            Es = self.supercell.get_potential_energy()/J*1e3
            x.append(0.05+(1./50)*i)
            y.append((Es/S0)-(E0/S0))
        GSF_fit = scipy.interpolate.InterpolatedUnivariateSpline(x,y)
        x_fit = np.linspace(0.05,0.05+(9./50),50)
        mins = argrelextrema(GSF_fit(x_fit),np.less)
        x_mins = x_fit[mins[0]]
        y_mins = GSF_fit(x_fit)[mins[0]]
        return y_mins[0]
      
    def get_lattice_constant(self):
        return np.sum(self.atoms.get_cell().diagonal()[:2])/np.sum(self.size[:2]),self.atoms.get_cell().diagonal()[2]/self.size[2]

    def get_C33(self):
        e0 = self.atoms.get_potential_energy()
        v0 = self.atoms.get_volume()
        
        cell = self.atoms.get_cell()
        T = np.array( [ [ 0.0, 0.0, 0.0 ],
                        [ 0.0, 0.0, 0.0 ],
                        [ 0.0, 0.0, self.eps ] ] )
        self.atoms.set_cell( np.dot(np.eye(3)+T, cell), scale_atoms=True )
        e = self.atoms.get_potential_energy()
        self.atoms.set_cell(cell, scale_atoms=True)
        
        
        return (2*(e-e0))/(v0*self.eps**2)
      
    def get_C44(self):
        e0 = self.atoms.get_potential_energy()
        v0 = self.atoms.get_volume()
        
        cell = self.atoms.get_cell()
        T = np.array( [ [ 0.0, 0.0, self.eps ],
                        [ 0.0, 0.0, self.eps ],
                        [ self.eps, self.eps, self.eps**2 ] ] )
        self.atoms.set_cell( np.dot(np.eye(3)+T, cell), scale_atoms=True )
        e = self.atoms.get_potential_energy()
        self.atoms.set_cell(cell, scale_atoms=True)
        
        
        return (e-e0)/(4*v0*self.eps**2)
      
    def get_C66(self):
        e0 = self.atoms.get_potential_energy()
        v0 = self.atoms.get_volume()
        
        cell = self.atoms.get_cell()
        T = np.array( [ [ (1.0+self.eps**2)**0.5, 0.0, 0.0 ],
                        [ 0.0, (1.0-self.eps**2)**0.5, 0.0 ],
                        [ 0.0, 0.0, 1.0 ] ] )
        self.atoms.set_cell( np.dot(np.eye(3)+T, cell), scale_atoms=True )
        e = self.atoms.get_potential_energy()
        self.atoms.set_cell(cell, scale_atoms=True)
        
        
        return (e-e0)/(2*v0*self.eps**2)
      
    def get_D1(self):
        e0 = self.atoms.get_potential_energy()
        v0 = self.atoms.get_volume()
        
        cell = self.atoms.get_cell()
        T = np.array( [ [ self.eps, 0.0, 0.0 ],
                        [ 0.0, self.eps, 0.0 ],
                        [ 0.0, 0.0, 0.0 ] ] )
        self.atoms.set_cell( np.dot(np.eye(3)+T, cell), scale_atoms=True )
        e = self.atoms.get_potential_energy()
        self.atoms.set_cell(cell, scale_atoms=True)
        
        
        return (e-e0)/(v0*self.eps**2)

    def get_D2(self):
        e0 = self.atoms.get_potential_energy()
        v0 = self.atoms.get_volume()
        
        cell = self.atoms.get_cell()
        T = np.array( [ [ 1.0+self.eps, 0.0, 0.0 ],
                        [ 0.0, 1.0+self.eps, 0.0 ],
                        [ 0.0, 0.0, 1.0/(1+self.eps)**2 ] ] )
        self.atoms.set_cell( np.dot(T, cell), scale_atoms=True )
        e = self.atoms.get_potential_energy()
        self.atoms.set_cell(cell, scale_atoms=True)
        
        
        return (e-e0)/(v0*self.eps**2)
      
    def get_D4(self):
        e0 = self.atoms.get_potential_energy()
        v0 = self.atoms.get_volume()
        
        cell = self.atoms.get_cell()
        T = np.array( [ [ ((1.0+self.eps)/(1.0-self.eps))**0.5, 0.0, 0.0 ],
                        [ 0.0, ((1.0-self.eps)/(1.0+self.eps))**0.5, 0.0 ],
                        [ 0.0, 0.0, 1.0 ] ] )
        self.atoms.set_cell( np.dot(T, cell), scale_atoms=True )
        e = self.atoms.get_potential_energy()
        self.atoms.set_cell(cell, scale_atoms=True)
        
        
        return (e-e0)/(v0*self.eps**2)
      
        
    def get_residuals(self, log=None):
        Ec = self.get_potential_energy()
        a0,c0 = self.get_lattice_constant()

        if self.ecoh_ref is not None:
            syms = np.array(self.atoms.get_chemical_symbols())
            for el in set(syms):
                Ec -= (syms==el).sum()*self.ecoh_ref[el]

        Ec /= len(self.atoms)

        r_a0 = self.w_a0*( a0 - self.a0 )
        r_c0 = self.w_c0*( c0 - self.c0 )

        r_Ec = self.w_Ec*( Ec - self.Ec )
        
        if log is not None:
            print('# %20s Ec  = %20.10f eV/at.    (%20.10f eV/at.)    - %20.10f' \
                % ( '%s (%s)' % (self.unitcell.get_chemical_formula(),
                                 self.crystalstr), Ec, self.Ec, r_Ec ))
            print('# %20s a0  = %20.10f A     (%20.10f A)     - %20.10f' \
                % ( '', a0, self.a0, r_a0 ))
            print('# %20s c0  = %20.10f A     (%20.10f A)     - %20.10f' \
                % ( '', c0, self.c0, r_c0 ))

        r = [ r_a0 ,r_c0, r_Ec]
        
        if self.SFE is not None:
            SFE = self.get_SFE()
        if self.B is not None or self.C11 is not None or self.C12 is not None or self.C13 is not None or self.C33 is not None:
            Czz = self.get_D2()
        if self.B is not None or self.C11 is not None or self.C12 is not None:
            Cp = self.get_D1()
        if self.B is not None or self.C11 is not None or self.C12 is not None:
            Cm = self.get_D4()
        if self.C33 is not None:
            C33 = self.get_C66()
        if self.C44 is not None:
            C44 = self.get_C66()
        if self.C66 is not None:
            C66 = self.get_C66()


        if self.SFE is not None:
            if SFE < 0:
                r_SFE = self.w_SFE*( SFE - self.SFE )*1000
            else:  
                r_SFE = self.w_SFE*( SFE - self.SFE )
            r += [ r_SFE ]
            if log is not None:
                print('# %20s SFE = %20.10f mJ/m**2   (%20.10f mJ/m**2)   - %20.10f' \
                    % ( '', SFE, self.SFE, r_SFE ))
        if self.c_a is not None:
            if (self.c_a < 1 and c0/a0 < 1) or (self.c_a > 1 and c0/a0 > 1) :
                r_c_a = self.w_c_a*( c0/a0 - self.c_a )
            else:
                r_c_a = self.w_c_a*( c0/a0 - self.c_a )*1000
            r += [ r_c_a ]
            if log is not None:
                print('# %20s c/a   = %20.10f   (%20.10f )   - %20.10f' \
                    % ( '', c0/a0 , self.c_a, r_c_a ))
        if self.B is not None:
            r_B = self.w_B*( (C33*Cp-2*C13**2)/Czz - self.B )
            r += [ r_B ]
            if log is not None:
                print('# %20s B   = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', (C33*Cp-2*C13**2)/Czz/GPa, self.B/GPa, r_B ))
        if self.C11 is not None:
            r_C11 = self.w_C11*( (Cp+Cm)/2 - self.C11 )
            r += [ r_C11 ]
            if log is not None:
                print('# %20s C11 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', (Cp+Cm)/2/GPa, self.C11/GPa, r_C11 ))
        if self.C12 is not None:
            r_C12 = self.w_C12*( (Cp-Cm)/2 - self.C12 )
            r += [ r_C12 ]
            if log is not None:
                print('# %20s C12 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', (Cp-Cm)/2/GPa, self.C12/GPa, r_C12 ))
        if self.C13 is not None:
            r_C13 = self.w_C13*( -(Czz-Cp-2*C33)/4 - self.C13 )
            r += [ r_C13 ]
            if log is not None:
                print('# %20s C13 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', -(Czz-Cp-2*C33)/4/GPa, self.C13/GPa, r_C13 ))
        if self.C33 is not None:
            r_C33 = self.w_C33*( C33 - self.C33 )
            r += [ r_C33 ]
            if log is not None:
                print('# %20s C33 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', C33/GPa, self.C33/GPa, r_C33 ))
        if self.C44 is not None:
            r_C44 = self.w_C44*( C44 - self.C44 )
            r += [ r_C44 ]
            if log is not None:
                print('# %20s C44 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', C44/GPa, self.C44/GPa, r_C44 ))
        if self.C66 is not None:
            r_C66 = self.w_C66*( C66 - self.C66 )
            r += [ r_C66 ]
            if log is not None:
                print('# %20s C66 = %20.10f GPa   (%20.10f GPa)   - %20.10f' \
                    % ( '', C66/GPa, self.C66/GPa, r_C66 ))

        
        return r

class FitHexagonalCrystal(Fit):

    __slots__ = [ 'a0', 'c0', 'calc', 'crystal', 'Ec', 'fmax', 'par', 'w_a0',
                  'w_Ec' ]

    def __init__(self, calc, par, els,
                 Ec, a0, c0,
                 w_Ec = 1.0, w_a0 = 1.0,
                 fmax = 0.01):
        Fit.__init__(self, calc, par)

        self.els = els

        self.Ec = Ec
        self.a0 = a0
        self.c0 = c0

        self.w_Ec = sqrt(w_Ec)/self.Ec
        self.w_a0 = sqrt(w_a0)/self.a0

        self.fmax = fmax

        self.atoms = None

    def new_bulk(self):
        self.atoms = self.crystal(
            self.els,
            latticeconstant  = [self.a0, self.c0],
            size             = [1, 1, 1]
            )
        self.atoms.translate([0.1, 0.1, 0.1])

    def set_calculator(self, calc):
        self.new_bulk()
        self.atoms.set_calculator(calc)
        ase.optimize.FIRE(
            ase.constraints.StrainFilter(self.atoms, mask=[1,1,0,0,0,0]),
            logfile=_logfile).run(fmax=self.fmax,steps=10000)

    def get_lattice_constant(self):
        cx, cy, cz = self.atoms.get_cell()
        
        return ( sqrt(np.dot(cx, cx)) + sqrt(np.dot(cy, cy)) )/2

    def get_residuals(self, log=None):
        Ec = self.get_potential_energy()/len(self.atoms)
        a0 = self.get_lattice_constant()

        r_Ec = self.w_Ec*( Ec + self.Ec )
        r_a0 = self.w_a0*( a0 - self.a0 )

        if log is not None:
            print('# %20s Ec  = %20.10f eV/at.    (%20.10f eV/at.)    - %20.10f' \
                % ( 'Crystal (%s)' % self.crystalstr, Ec, -self.Ec, r_Ec ))
            print('# %20s a0  = %20.10f A     (%20.10f A)     - %20.10f' \
                % ( '', a0, self.a0, r_a0 ))

        r = [ r_Ec, r_a0 ]

        return r

class FitSurface(Fit):

    __slots__ = [ 'a0', 'calc', 'crystal', 'Ec', 'fmax', 'par', 'w_a0', 'w_Ec' ]

    def __init__(self, calc, par, els, crystal,
                 Esurf,
                 w_Esurf = 1.0):
        self.Esurf = Esurf

        self.w_Esurf = sqrt(w_Esurf)

        self.els = els

        self.calc = calc
        self.par = par

        self.crystal = crystal

    def set_calculator(self, calc):
        self.atoms, self.ncells = \
            self.new_surface(self.crystal.get_lattice_constant())
        ase.io.write('%s.cfg' % self.surfstr, self.atoms)
        self.atoms.set_calculator(calc)

    def get_surface_energy(self):
        return ( self.atoms.get_potential_energy() -
                 self.crystal.get_cohesive_energy()*len(self.atoms) ) \
                 /(2*self.ncells)

    def get_residuals(self, log=None):
        Esurf = self.get_surface_energy()
        sx, sy, sz = self.atoms.get_cell().diagonal()
        tar_Esurf = self.Esurf*(sx*sy)/self.ncells
        r_Esurf = self.w_Esurf*( Esurf - tar_Esurf )/tar_Esurf

        if log is not None:
            print('# %20s Es  = %20.10f eV    (%20.10f eV)    - %20.10f' \
                % ( 'Surface(%s)' % self.surfstr, Esurf,
                    tar_Esurf, r_Esurf ))
            print('# %20s       %20.10f J/m^2 (%20.10f J/m^2)' \
                % ( '',
                    Esurf*self.ncells*Jm2/(sx*sy),
                    self.Esurf*Jm2 ))
        
        return [ r_Esurf ]

###

class FitSC(FitCubicCrystal):
    crystal = cubic.SimpleCubic
    crystalstr = 'sc'

class FitBCC(FitCubicCrystal):
    crystal = cubic.BodyCenteredCubic
    crystalstr = 'bcc'

class FitFCC(FitCubicCrystal):
    crystal = cubic.FaceCenteredCubic
    crystalstr = 'fcc'

class FitB2(FitCubicCrystal):
    crystal = compounds.B2
    crystalstr = 'B2'

class FitL1_0(FitTetragonalCrystal):
    crystal = compounds.L1_0
    crystalstr = 'L1_0'

class FitL1_2(FitCubicCrystal):
    crystal = compounds.L1_2
    crystalstr = 'L1_2'

class FitDiamond(FitCubicCrystal):
    crystal = cubic.Diamond
    crystalstr = 'dia'

class FitGraphite(FitHexagonalCrystal):
    crystal = hexagonal.Graphite
    crystalstr = 'gra'

class FitGraphene(FitHexagonalCrystal):
    crystal = hexagonal.Graphene
    crystalstr = 'grp'

class FitDiamond100(FitSurface):
    surfstr = 'dia-100'

    def new_surface(self, a0):
        a = cubic.Diamond(
            self.els,
            latticeconstant = a0,
            directions = [ [ 1,0,0 ],
                           [ 0,1,0 ],
                           [ 0,0,1 ] ],
            size = [ 1, 1, 5 ])
        a.translate([0.1,0.1,0.1])
        sx, sy, sz = a.get_cell().diagonal()
        a.set_cell([sx,sy,2*sz])

        return a, 2

class FitDiamond110(FitSurface):
    surfstr = 'dia-110'

    def new_surface(self, a0):
        a = cubic.Diamond(
            self.els,
            latticeconstant = a0,
            directions = [ [ 1, 0,0 ],
                           [ 0, 1,1 ],
                           [ 0,-1,1 ] ],
            size = [ 1, 1, 5 ])
        a.translate([0.1,0.1,0.1])
        sx, sy, sz = a.get_cell().diagonal()
        a.set_cell([sx,sy,2*sz])

        return a, 1

class FitDiamond111(FitSurface):
    surfstr = 'dia-111'
 
    def new_surface(self, a0):
        a = cubic.Diamond(
            self.els,
            latticeconstant = a0,
            directions = [ [ 2,-1,-1 ],
                           [ 0, 1,-1 ],
                           [ 1, 1, 1 ] ],
            size = [ 1, 1, 5 ])
        a.translate([0.1,0.1,0.1+a0/4])
        a.set_scaled_positions(a.get_scaled_positions())
        sx, sy, sz = a.get_cell().diagonal()
        a.set_cell([sx,sy,2*sz])

        return a, 2
