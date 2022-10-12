#
# Copyright 2019-2020 Johannes Hoermann (U. Freiburg)
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
""" Calculate ionic densities consistent with the Poisson-Boltzmann equation.

Copyright 2019, 2020 IMTEK Simulation
University of Freiburg

Authors:

  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
  Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import numpy as np
import scipy.constants as sc
import decimal

np.random.seed(74)

def ionic_strength(c,z):
    """Compute ionic strength from charges and concentrations

    Parameters
    ----------
    c : (M,) ndarray
        M bulk concentrations [concentration unit, i.e. mol m^-3]
    z : (M,) ndarray
        M number charges [number charge unit, i.e. 1]

    Returns
    -------
    I : float
        ionic strength ( 1/2 * sum(z_i^2*c_i) )
        [concentration unit, i.e. mol m^-3]
    """
    return 0.5*np.sum( np.square(z) * c )

def debye(c, z,
    T=298.15,
    relative_permittivity=79,
    vacuum_permittivity=sc.epsilon_0,
    R = sc.value('molar gas constant'),
    F=sc.value('Faraday constant') ):
    """Calculate the Debye length (in SI units per default).

    The Debye length indicates at which distance a charge will be screened off.

    Parameters
    ----------
    c : (M,) ndarray
        bulk concentrations of each ionic species [mol/m^3]
    z : (M,) ndarray
        charge of each ionic species [1]
    T : float
        temperature of the solution [K] (default: 298.15)
    relative_permittivity: float
        relative permittivity of the ionic solution [1] (default: 79)
    vacuum_permittivity: float
        vacuum permittivity [F m^-1] (default: 8.854187817620389e-12 )
    R : float
        molar gas constant [J mol^-1 K^-1] (default: 8.3144598)
    F : float
        Faraday constant [C mol^-1] (default: 96485.33289)

    Returns
    -------
    lambda_D : float
        Debye length, sqrt( epsR*eps*R*T/(2*F^2*I) ) [m]
    """
    I = ionic_strength(c,z)
    return np.sqrt(relative_permittivity*vacuum_permittivity*R*T/(2.0*F**2*I))


def gamma(u, T = 298.15):
    """Calculate term from Gouy-Chapmann theory.

    Parameters
    ----------
    u: float
        electrostatic potential at the metal/solution boundary in Volts, e.g. 0.05 [V]
    T: float
        temperature of the solution in Kelvin [K] (default: 298.15)

    Returns
    -------
    float
    """
    product = sc.value('Faraday constant') * u / (4 * sc.value('molar gas constant') * T)
    return np.tanh(product)

def potential(x, c, z, u, T=298.15, relative_permittivity=79):
    """The potential near a charged surface in an ionic solution.

    A single value is returned, which specifies the value of the potential at this distance.

    The decimal package is used for increased precision.
    If only normal float precision is used, the potential is a step function.
    Steps in the potential result in unphysical particle concentrations.

    Paramters
    ---------
    x : (N,) ndarray
        z-distance from the surface [m]
    c : (M,) ndarray
        bulk concentrations of each ionic species [mol/m^3]
    T : float
        temperature of the soultion [Kelvin] (default: 298.15)
    relative_permittivity:
        relative permittivity of the ionic solution [] (default: 79)

    Returns
    -------
    phi: (N,) ndarray
        Electrostatic potential [V]
    """
    # Increase the precision of the calculation to 30 digits to ensure a smooth potential
    #decimal.getcontext().prec = 30

    # Calculate the term in front of the log, containing a bunch of constants
    prefactor = 2 * sc.value('molar gas constant') * T / sc.value('Faraday constant')

    # For the later calculations we need the debye length
    debye_value =  debye(c, z, T, relative_permittivity)

    kappa = 1 / debye_value

    # We also need to evaluate the gamma function
    gamma_value =  gamma(u, T)

    # The e^{-kz} term
    exponential = np.exp(-kappa * x)

    # The fraction inside the log
    numerator = 1.0 + gamma_value * exponential
    divisor =   1.0 - gamma_value * exponential

    # This is the complete term for the potential
    phi = prefactor * np.log(numerator / divisor)

    # Convert to float again for better handling in plotting etc.
    # phi = float(phi)

    return phi


def concentration(x, c, z, u, T=298.15,
    relative_permittivity=79):
    """The concentration of ions near a charged surface.

    Calculates the molar concentration of ions of a species, at a distance x
    away from a substrate/solution interface. Potential difference u between
    substrate and bulk solution leads to non-neutrality close to the interface,
    with concentrations converging to their bulk values at high distances.

    Parameters
    ----------
    x : (N,) ndarray
        distance from the surface [m]
    c : (M,) ndarray
        bulk concentrations (i.e. far from the surface) [mol/m^-3]
    z : (M,) ndarray
        number charge of each ionic species [1]
    u : float
        eletrostatic potential at the metal/liquid interface against bulk [V]
    T : float
        temperature of the solution [K] (default: 298.15)
    relative_permittivity : float
        relative permittivity of the ionic solution, 80 for water [1]

    Returns:
    c : (M,N) ndarray
        molar concentrations of ion species [mol/m^3]
    """

    # Evaluate the potential at the current location
    potential_value = potential(x, c, z, u, T, relative_permittivity)
    phi_z = np.outer(potential_value,np.array(z))# N x M matrix (rows, cols)
    f = sc.value('Faraday constant') / (sc.value('molar gas constant') * T)
    # The concentration is an exponential function of the potential
    C = np.exp( - f * phi_z )

    # The concentration is scaled relative to the bulk concentration
    C *= c
    return C.T # M x N matrix (rows, cols)

def charge_density(x, c, z, u, T=298.15, relative_permittivity=79):
    """
    Charge density due to Poisson-Boltzmann distribution
    Parameters
    ----------
    x : (N,) ndarray
        distance from the surface [m]
    c : (M,) ndarray
        bulk concentrations (i.e. far from the surface) [mol/m^-3]
    z : (M,) ndarray
        number charge of each ionic species [1]
    u : float
        eletrostatic potential at the metal/liquid interface against bulk [V]
    T : float
        temperature of the solution [K] (default: 298.15)
    relative_permittivity : float
        relative permittivity of the ionic solution, 80 for water [1]

    Returns:
    c : (N,) ndarray
        charge density [C/m^3]
    """
    C = concentration(x,c,z,u,T,relative_permittivity)
    return sc.value("Faraday constant") * np.sum(C.T*z,axis=1)



def test():
    """Run docstring unittests"""
    import doctest
    doctest.testmod()


def main():
    """Do stuff."""

    print('Done.')


if __name__ == '__main__':
    # Run doctests
    test()
    # Execute everything else
    main()
