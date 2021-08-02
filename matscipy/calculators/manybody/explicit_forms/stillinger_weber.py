#
# Copyright 2021 Jan Griesser (U. Freiburg)
#           2021 Lars Pastewka (U. Freiburg)
#           2021 griesserj@griesserjs-MacBook-Pro.local
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

#
# Parameter sets
# The '__ref__' dictionary entry is the journal reference
#

Stillinger_Weber_PRB_31_5262_Si = {
    '__ref__':  'F. Stillinger and T. Weber, Phys. Rev. B 31, 5262 (1985)',
    'el':            'Si'            ,
    'epsilon':       2.1683          ,
    'sigma':         2.0951          ,
    'costheta0':     0.333333333333  ,
    'A':             7.049556277     ,
    'B':             0.6022245584    ,
    'p':             4               ,
    'a':             1.80            ,
    'lambda1':       21.0            ,
    'gamma':         1.20            
}

Holland_Marder_PRL_80_746_Si = {
    '__ref__':  'D. Holland and M. Marder, Phys. Rev. Lett. 80, 746 (1998)',
    'el':            'Si'            ,
    'epsilon':       2.1683          ,
    'sigma':         2.0951          ,
    'costheta0':     0.333333333333  ,
    'A':             7.049556277     ,
    'B':             0.6022245584    ,
    'p':             4               ,
    'a':             1.80            ,
    'lambda1':       42.0            ,
    'gamma':         1.20            
}

RLC_Vink_JNCS_282_746_Si = {
    '__ref__':  'RLC Vink et al., J. Non-Cryst. Solids 282 (2001)',
    'el':            'Si'            ,
    'epsilon':       1.64833         ,
    'sigma':         2.0951          ,
    'costheta0':     0.333333333333  ,
    'A':             7.049556277     ,
    'B':             0.6022245584    ,
    'p':             4               ,
    'a':             1.80            ,
    'lambda1':       31.5            ,
    'gamma':         1.20            
}

Russo_PRX_8_021040_Si = {
    '__ref__':  'J. Russo et. al. Phys. Rev. X 8, 021040 (2018)',
    'el':            'Si'            ,
    'epsilon':       2.1683          ,
    'sigma':         2.0951          ,
    'costheta0':     0.333333333333  ,
    'A':             7.049556277     ,
    'B':             0.6022245584    ,
    'p':             4               ,
    'a':             1.80            ,
    'lambda1':       18.75           ,
    'gamma':         1.20            
}

def ab(x):
    """
    Compute absolute value (norm) of an array of vectors
    """
    return np.linalg.norm(x, axis=1)

def StillingerWeber(parameters):
    """
    Implementation of the functional form of the Stillinger-Weber potential.

    Reference
    ------------
    F. Stillinger and T. Weber, Physical review B 31.8 5262 (1985)

    """

    el = parameters['el']
    epsilon = parameters['epsilon']
    sigma = parameters['sigma']
    costheta0 = parameters['costheta0']
    A = parameters['A']
    B = parameters['B']
    p = parameters['p']
    a = parameters['a']
    lambda1 = parameters['lambda1']
    gamma = parameters['gamma']

    fR = lambda r: A * epsilon * (B*np.power(sigma/r, p) - 1) * np.exp(sigma/(r-a*sigma))
    dfR = lambda r: - A*epsilon*B*p/r * np.power(sigma/r, p) * np.exp(sigma/(r-a*sigma)) - sigma/np.power(r-a*sigma, 2)*fR(r)
    ddfR = lambda r: A*B*p*epsilon/r * np.power(sigma/r, p) * np.exp(sigma/(r-a*sigma)) * (sigma/np.power(r-a*sigma, 2) + (p+1)/r) \
                     + 2 * sigma / np.power(r-a*sigma, 3) * fR(r) - sigma / np.power(r-a*sigma, 2) * dfR(r)

    fA = lambda r: np.exp(gamma*sigma/(r-a*sigma))
    dfA = lambda r: - gamma * sigma / np.power(r - a*sigma, 2) * fA(r)
    ddfA = lambda r: 2 * gamma * sigma / np.power(r-a*sigma, 3) * fA(r) - gamma * sigma / np.power(r-a*sigma, 2) * dfA(r)

    hf = lambda rik: np.where(ab(rik)<a*sigma, np.exp(gamma*sigma/(ab(rik)-a*sigma)), 0) 
    d2h = lambda rik: -gamma * sigma / np.power(ab(rik)-a*sigma, 2) * hf(rik)
    d22h = lambda rik: 2 * gamma * sigma / np.power(ab(rik)-a*sigma, 3) * hf(rik) - gamma * sigma / np.power(ab(rik)-a*sigma, 2) * d2h(rik)

    g = lambda cost: np.power(cost + costheta0, 2)
    dg = lambda cost: 2 * (cost + costheta0)
    ddg = lambda cost: 2*cost**0

    def F(r, xi, i, p):
        mask = (r < a*sigma)
        F_n = np.zeros_like(r)
        F_n[mask] = fR(r[mask]) + lambda1 * epsilon * fA(r[mask]) * xi[mask]
        return F_n
    def d1F(r, xi, i, p):
        mask = (r < a*sigma)
        d1F_n = np.zeros_like(r)
        d1F_n[mask] = dfR(r[mask]) + lambda1 * epsilon * xi[mask] * dfA(r[mask])
        return d1F_n
    def d2F(r, xi, i, p):
        mask = (r < a*sigma)
        d2F_n = np.zeros_like(r)
        d2F_n[mask] = lambda1 * epsilon * fA(r[mask])     
        return d2F_n
    def d11F(r, xi, i, p):
        mask = (r < a*sigma)
        d11F_n = np.zeros_like(r)
        d11F_n[mask] = ddfR(r[mask]) + lambda1 * epsilon * xi[mask] * ddfA(r[mask])   
        return d11F_n
    def d22F(r, xi, i, p):
        return np.zeros_like(r)
    def d12F(r, xi, i, p):
        mask = (r < a*sigma)
        d12F_n = np.zeros_like(r)
        d12F_n[mask] = lambda1 * epsilon * dfA(r[mask])
        return d12F_n

    G = lambda rij, rik, i, ij, ik: hf(rik) * g(costh(rij, rik))

    d1G = lambda rij, rik, i, ij, ik: (hf(rik) * Dg1(rij, rik).T).T
    d2G = lambda rij, rik, i, ij, ik: (Dh2(rik).T * g(costh(rij, rik)) + hf(rik) * Dg2(rij, rik).T).T

    Dh2 = lambda rik: (d2h(rik) * rik.T / ab(rik)).T 

    Dg1 = lambda rij, rik: (dg(costh(rij, rik)) * c1(rij, rik).T).T 
    Dg2 = lambda rij, rik: (dg(costh(rij, rik)) * c2(rij, rik).T).T

    d11G = lambda rij, rik, i, ij, ik: \
        ((hf(rik) * Dg11(rij, rik).T).T)

    Dg11 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c1(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc11(rij, rik).T).T

    d22G = lambda rij, rik, i, ij, ik: \
        Dg2(rij, rik).reshape(-1, 3, 1) * Dh2(rik).reshape(-1, 1, 3) + Dh2(rik).reshape(-1, 3, 1) * Dg2(rij, rik).reshape(-1, 1, 3) \
        + ((g(costh(rij, rik)) * Dh22(rij, rik).T).T + (hf(rik) * Dg22(rij, rik).T).T)

    Dg22 = lambda rij, rik: \
        (ddg(costh(rij, rik)) * (c2(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
        + dg(costh(rij, rik)) * dc22(rij, rik).T).T

    Dh22 = lambda rij, rik: \
        (d22h(rik) * (((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T \
         + d2h(rik) * ((np.eye(3) - ((rik.reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik)**2).T).T/ab(rik))).T

    d12G = lambda rij, rik, i, ij, ik: \
        Dg1(rij, rik).reshape(-1, 3, 1) * Dh2(rik).reshape(-1, 1, 3) + ((hf(rik) * Dg12(rij, rik).T).T)

    Dg12 = lambda rij, rik: \
         (ddg(costh(rij, rik)) * (c1(rij, rik).reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3)).T
         + dg(costh(rij, rik)) * dc12(rij, rik).T).T    

    # Helping functions 
    c1 = lambda rij, rik: ((rik.T/ab(rik) - rij.T/ab(rij) * costh(rij, rik)) / ab(rij)).T
    c2 = lambda rij, rik: ((rij.T/ab(rij) - rik.T/ab(rik) * costh(rij, rik)) / ab(rik)).T
    dc11 = lambda rij, rik: \
        ((- c1(rij, rik).reshape(-1, 3, 1) * rij.reshape(-1, 1, 3) \
         - rij.reshape(-1, 3, 1) * c1(rij, rik).reshape(-1, 1, 3) \
         - (costh(rij, rik) * (np.eye(3) -  ((rij.reshape(-1, 1, 3)*rij.reshape(-1, 3, 1)).T/ab(rij)**2).T).T).T \
        ).T/ab(rij)**2).T
    dc22 = lambda rij, rik: \
        ((- c2(rij, rik).reshape(-1, 3, 1) * rik.reshape(-1, 1, 3) \
         - rik.reshape(-1, 3, 1) * c2(rij, rik).reshape(-1, 1, 3) \
         - (costh(rij, rik) * (np.eye(3) -  ((rik.reshape(-1, 1, 3)*rik.reshape(-1, 3, 1)).T/ab(rik)**2).T).T).T \
        ).T/ab(rik)**2).T
    dc12 = lambda rij, rik: \
        (((np.eye(3) -  ((rij.reshape(-1, 1, 3)*rij.reshape(-1, 3, 1)).T/ab(rij)**2).T).T/ab(rij)
         - (c1(rij, rik).reshape(-1, 3, 1) * rik.reshape(-1, 1, 3)).T/ab(rik) \
        )/ab(rik)).T

    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))    

    return {
        'atom_type': lambda n: np.zeros_like(n),
        'pair_type': lambda i, j: np.zeros_like(i),
        'F': F,
        'G': G,
        'd1F': d1F,
        'd2F': d2F,
        'd11F': d11F,
        'd12F': d12F,
        'd22F': d22F,
        'd1G': d1G,
        'd2G': d2G,
        'd11G': d11G,
        'd22G': d22G,
        'd12G': d12G,
        'cutoff': a*sigma,
    }
