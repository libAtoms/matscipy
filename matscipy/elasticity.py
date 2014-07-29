# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) Lars Pastewka, Karlsruhe Institute of Technology
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

import numpy as np
from numpy.linalg import inv

###

# The indices of the full stiffness matrix of (orthorhombic) interest
Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

###

def Voigt_6_to_full_3x3(s):
    sxx, syy, szz, syz, sxz, sxy = s
    return np.array([[sxx,sxy,sxz],
                     [sxy,syy,syz],
                     [sxz,syz,szz]])


def full_3x3x3x3_to_Voigt_6x6(C):
    """
    Convert from the full 3x3x3x3 representation of the stiffness matrix
    to the representation in Voigt notation. Checks symmetry in that process.
    """

    tol = 1e-3

    C = np.asarray(C)
    Voigt = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            k, l = Voigt_notation[i]
            m, n = Voigt_notation[j]
            Voigt[i,j] = C[k,l,m,n]

            #print '---'
            #print k,l,m,n, C[k,l,m,n]
            #print m,n,k,l, C[m,n,k,l]
            #print l,k,m,n, C[l,k,m,n]
            #print k,l,n,m, C[k,l,n,m]
            #print m,n,l,k, C[m,n,l,k]
            #print n,m,k,l, C[n,m,k,l]
            #print l,k,n,m, C[l,k,n,m]
            #print n,m,l,k, C[n,m,l,k]
            #print '---'

            # Check symmetries
            assert abs(Voigt[i,j]-C[m,n,k,l]) < tol, \
                'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                .format(i, j, Voigt[i,j], m, n, k, l, C[m,n,k,l])
            assert abs(Voigt[i,j]-C[l,k,m,n]) < tol, \
                'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                .format(i, j, Voigt[i,j], k, l, m, n, C[l,k,m,n])
            assert abs(Voigt[i,j]-C[k,l,n,m]) < tol, \
                'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                .format(i, j, Voigt[i,j], k, l, n, m, C[k,l,n,m])
            assert abs(Voigt[i,j]-C[m,n,l,k]) < tol, \
                'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                .format(i, j, Voigt[i,j], m, n, l, k, C[m,n,l,k])
            assert abs(Voigt[i,j]-C[n,m,k,l]) < tol, \
                'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                .format(i, j, Voigt[i,j], n, m, k, l, C[n,m,k,l])
            assert abs(Voigt[i,j]-C[l,k,n,m]) < tol, \
                'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                .format(i, j, Voigt[i,j], l, k, n, m, C[l,k,n,m])
            assert abs(Voigt[i,j]-C[n,m,l,k]) < tol, \
                'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
                .format(i, j, Voigt[i,j], n, m, l, k, C[n,m,l,k])

    return Voigt


def Voigt_6x6_to_orthorhombic(C):
    """
    Convert the Voigt 6x6 representation into the orthorhombic elastic constants
    C11, C12 and C44.
    """

    C11 = np.array([C[0,0], C[1,1], C[2,2]])
    C12 = np.array([C[1,2], C[0,2], C[0,1]])
    C44 = np.array([C[3,3], C[4,4], C[5,5]])

    return C11, C12, C44

###

class CubicElasticModuli:
    tol = 1e-6

    def __init__(self, C11, C12, C44):
        """
        Initialize a cubic system with elastic constants C11, C12, C44
        """

        # la, mu, al are the three invariant elastic constants
        self.la = C12
        self.mu = C44
        self.al = C11 - self.la - 2*self.mu

        A = np.eye(3, dtype=float)

        # Compute initial stiffness matrix
        self.rotate(A)


    def rotate(self, A):
        """
        Compute the rotated stiffness matrix
        """

        A = np.asarray(A)

        # Is this a rotation matrix?
        if np.sometrue(np.abs(np.dot(np.array(A), np.transpose(np.array(A))) - 
                              np.eye(3, dtype=float)) > self.tol):
            raise RuntimeError('Matrix *A* does not describe a rotation.')

        C = [ ]
        for i, j in Voigt_notation:
            for k, l in Voigt_notation:
                h = 0.0
                if i == j and k == l:
                    h += self.la
                if i == k and j == l:
                    h += self.mu
                if i == l and j == k:
                    h += self.mu
                h += self.al*np.sum(A[i,:]*A[j,:]*A[k,:]*A[l,:])
                C += [ h ]

        self.C = np.asarray(C)
        self.C.shape = (6, 6)
        return self.C


    def _rotate_explicit(self, A):
        """
        Compute the rotated stiffness matrix by applying the rotation to the
        full stiffness matrix. This function is for debugging purposes only.
        """

        A = np.asarray(A)

        # Is this a rotation matrix?
        if np.sometrue(np.abs(np.dot(np.array(A), np.transpose(np.array(A))) - 
                              np.eye(3, dtype=float) ) > self.tol):
            raise RuntimeError('Matrix *A* does not describe a rotation.')

        C = np.zeros((3, 3, 3, 3), dtype=float)

        # Construct unrotated stiffness matrix
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        h = 0.0
                        if i == j and k == m:
                            h += self.la
                        if i == k and j == m:
                            h += self.mu
                        if i == m and j == k:
                            h += self.mu
                        if i == j and j == k and k == m:
                            h += self.al
                        C[i, j, k, m] = h

        # Rotate
        C = np.einsum('ia,jb,kc,ld,abcd->ijkl', A, A, A, A, C)

        self.C = full_3x3x3x3_to_Voigt_6x6(C)
        return self.C


    def stiffness(self):
        """
        Return the elastic constants
        """

        return self.C


    def compliance(self):
        """
        Return the compliance coefficients
        """

        return inv(self.C)

###
    
def measure_orthorhombic_elastic_moduli(a, delta=0.001, optimizer=None, 
                                        logfile=None, **kwargs):
    """
    Measure elastic constant for an orthorhombic unit cell

    Parameters:
    -----------
    a           ase.Atoms object
    optimizer   Optimizer to use for atomic position. Does not optimize atomic
                position if set to None.
    delta         Strain increment for analytical derivatives of stresses.
    """

    if optimizer is not None:
        optimizer(a, logfile=logfile).run(**kwargs)

    r0 = a.positions.copy()

    cell = a.cell
    s0 = a.get_stress()

    # C11
    C11 = [ ]
    for i in range(3):
        a.set_cell(cell, scale_atoms=True)
        a.set_positions(r0)
        
        D = np.eye(3)
        D[i, i] = 1.0+delta
        a.set_cell(np.dot(D, cell), scale_atoms=True)
        if optimizer is not None:
            optimizer(a, logfile=logfile).run(**kwargs)
        s = a.get_stress()
            
        C11 += [ (s[i]-s0[i])/delta ]

    volfac = 1.0/((1-delta**2)**(1./3))

    # C'
    Cp = [ ] 
    for i in range(3):
        a.set_cell(cell, scale_atoms=True)
        a.set_positions(r0)
        
        D = volfac*np.eye(3)
        j = (i+1)%3
        k = (i+2)%3
        D[j, j] *= 1+delta
        D[k, k] *= 1-delta
        a.set_cell(np.dot(D, cell), scale_atoms=True)
        if optimizer is not None:
            optimizer(a, logfile=logfile).run(**kwargs)
        s = a.get_stress()

        Cp += [ ((s[j]-s0[j])-(s[k]-s0[k]))/(4*delta) ]

    # C44
    C44 = [ ]
    for i in range(3):
        a.set_cell(cell, scale_atoms=True)
        a.set_positions(r0)

        D = volfac*np.eye(3)
        j = (i+1)%3
        k = (i+2)%3
        D[j, k] = volfac*delta
        D[k, j] = volfac*delta
        a.set_cell(np.dot(D, cell), scale_atoms=True)
        if optimizer is not None:
            optimizer(a, logfile=logfile).run(**kwargs)
        s = a.get_stress()

        C44 += [ (s[3+i]-s0[3+i])/(2*delta) ]

    a.set_cell(cell, scale_atoms=True)
    a.set_positions(r0)

    C11 = np.array(C11)
    Cp = np.array(Cp)
    C44 = np.array(C44)

    # Compute C12 from C11 and C'
    C12 = np.array([C11[1]+C11[2], C11[0]+C11[2], C11[0]+C11[1]])/2-2*Cp

    return C11, C12, C44

###

def measure_triclinic_elastic_moduli(a, delta=0.001, optimizer=None, 
                                     logfile=None, **kwargs):
    """
    Measure elastic constant for an orthorhombic unit cell

    Parameters:
    -----------
    a           ase.Atoms object
    optimizer   Optimizer to use for atomic position. Does not optimize atomic
                position if set to None.
    delta         Strain increment for analytical derivatives of stresses.
    """

    if optimizer is not None:
        optimizer(a, logfile=logfile).run(**kwargs)

    r0 = a.positions.copy()

    cell = a.cell.copy()
    volume = a.get_volume()

    C = np.zeros((3,3,3,3), dtype=float)
    for i in range(3):
        for j in range(3):
            a.set_cell(cell, scale_atoms=True)
            a.set_positions(r0)
        
            D = np.eye(3)
            D[i, j] += 0.5*delta
            D[j, i] += 0.5*delta
            a.set_cell(np.dot(D, cell.T).T, scale_atoms=True)
            if optimizer is not None:
                optimizer(a, logfile=logfile).run(**kwargs)
            sp = Voigt_6_to_full_3x3(a.get_stress()*a.get_volume())

            D = np.eye(3)
            D[i, j] -= 0.5*delta
            D[j, i] -= 0.5*delta
            a.set_cell(np.dot(D, cell.T).T, scale_atoms=True)
            if optimizer is not None:
                optimizer(a, logfile=logfile).run(**kwargs)
            sm = Voigt_6_to_full_3x3(a.get_stress()*a.get_volume())

            C[:,:,i,j] = (sp-sm)/(2*delta*volume)

    a.set_cell(cell, scale_atoms=True)
    a.set_positions(r0)

    return full_3x3x3x3_to_Voigt_6x6(C)



Cij_symmetry = {
   'cubic':           np.array([[1, 7, 7, 0, 0, 0],
                                [7, 1, 7, 0, 0, 0],
                                [7, 7, 1, 0, 0, 0],
                                [0, 0, 0, 4, 0, 0],
                                [0, 0, 0, 0, 4, 0],
                                [0, 0, 0, 0, 0, 4]]),

   'trigonal_high':   np.array([[1, 7, 8, 9, 10, 0],
                                [7, 1, 8, 0,-9, 0],
                                [8, 8, 3, 0, 0, 0],
                                [9, -9, 0, 4, 0, 0],
                                [10, 0, 0, 0, 4, 0],
                                [0, 0, 0, 0, 0, 6]]),

   'trigonal_low':    np.array([[1,  7,  8,  9,  10,  0 ],
                                [7,  1,  8, -9, -10,  0 ],
                                [8,  8,  3,  0,   0,  0 ],
                                [9, -9,  0,  4,   0, -10],
                                [10,-10, 0,  0,   4,  9 ],
                                [0,  0,  0, -10 , 9,  6 ]]),

   'tetragonal_high': np.array([[1, 7, 8, 0, 0, 0],
                                [7, 1, 8, 0, 0, 0],
                                [8, 8, 3, 0, 0, 0],
                                [0, 0, 0, 4, 0, 0],
                                [0, 0, 0, 0, 4, 0],
                                [0, 0, 0, 0, 0, 6]]),

   'tetragonal_low':  np.array([[1, 7, 8, 0, 0, 11],
                                [7, 1, 8, 0, 0, -11],
                                [8, 8, 3, 0, 0, 0],
                                [0, 0, 0, 4, 0, 0],
                                [0, 0, 0, 0, 4, 0],
                                [11, -11, 0, 0, 0, 6]]),

   'orthorhombic':    np.array([[ 1,  7,  8,  0,  0,  0],
                                [ 7,  2, 12,  0,  0,  0],
                                [ 8, 12,  3,  0,  0,  0],
                                [ 0,  0,  0,  4,  0,  0],
                                [ 0,  0,  0,  0,  5,  0],
                                [ 0,  0,  0,  0,  0,  6]]),

   'monoclinic':      np.array([[ 1,  7,  8,  0,  10,  0],
                                [ 7,  2, 12,  0, 14,  0],
                                [ 8, 12,  3,  0, 17,  0],
                                [ 0,  0,  0,  4,  0,  20],
                                [10, 14, 17,  0,  5,  0],
                                [ 0,  0,  0, 20,  0,  6]]),
    
    'triclinic':       np.array([[ 1,  7,  8,  9,  10, 11],
                                 [ 7,  2, 12,  13, 14, 15],
                                 [ 8, 12,  3,  16, 17, 18],
                                 [ 9, 13, 16,  4,  19, 20],
                                 [10, 14, 17, 19,  5,  21],
                                 [11, 15, 18, 20,  21, 6 ]]),
   }


strain_patterns = {

   'cubic': [
      # strain pattern e1+e4, yields C11, C21, C31 and C44, then C12 is average of C21 and C31
      [ np.array([1,0,0,1,0,0]), [(1,1), (2,1), (3,1), (4,4)]]
   ],

   'trigonal_high': [
      # strain pattern e3 yield C13, C23 and C33
      [ np.array([0,0,1,0,0,0]), [(1,3), (2,3), (3,3)]],

      # strain pattern e1+e4 yields C11 C21 C31 and C44
      [ np.array([1,0,0,1,0,0]), [(1,1), (2,1), (3,1), (4,4)]],

      # strain pattern e1 yields C11 C21 C31 C41 C51
      [ np.array([1,0,0,0,0,0]), [(1,1), (2,1), (3,1), (4,1), (5,1)]],

      # strain pattern e3+e4
      [ np.array([0,0,1,1,0,0]), [(3,3), (4,4)]]

   ],

   'trigonal_low': [
     # strain pattern e1, yields C11, C21, C31, C41, C51
     [ np.array([1,0,0,0,0,0]), [(1,1), (2,1), (3,1), (4,1), (5,1)]],

     # strain pattern e3 + e4, yields C33, C44
     [ np.array([0,0,1,1,0,0]), [(3,3), (4,4)] ],

     [ np.array([0,0,0,0,0,1]), [(6,6)] ]
   ],

   'tetragonal': [
     # strain pattern e1+e4
     [ np.array([1,0,0,1,0,0]), [(1,1), (2,1), (3,1), (6,1), (4,4)] ],

     # strain pattern e3+e6
     [ np.array([0,0,1,0,0,1]), [(3,3), (6,6)] ]
   ],

   'orthorhombic': [
      # strain pattern e1+e4
      [ np.array([1,0,0,1,0,0]), [(1,1), (2,1), (3,1), (4,4)] ],

      # strain pattern e2+e5
      [ np.array([0,1,0,0,1,0]), [(1,2), (2,2), (3,2), (5,5)] ],

      # strain pattern e3+e6
      [ np.array([0,0,1,0,0,1]), [(1,3), (2,3), (3,3), (6,6)] ]
   ],

   'monoclinic': [
      # strain pattern e1+e4
      [ np.array([1,0,0,1,0,0]), [(1,1), (2,1), (3,1), (4,4), (5,1), (6,4)] ],

      # strain pattern e3+e6
      [ np.array([0,0,1,0,0,1]), [(1,3), (2,3), (3,3), (5,3), (4,6), (6,6)] ],

      # strain pattern e2
      [ np.array([0,1,0,0,0,0]), [(1,2), (2,2), (3,2), (5,2)] ],

      # strain pattern e5
      [ np.array([0,0,0,0,1,0]), [(1,5), (2,5), (3,5), (5,5)] ]
   ],

   'triclinic': [
      [ np.array([1,0,0,0,0,0]), [(1,1), (2,1), (3,1), (4,1), (5,1), (6,1)]],
      [ np.array([0,1,0,0,0,0]), [(1,2), (2,2), (3,2), (4,2), (5,2), (6,2)]],
      [ np.array([0,0,1,0,0,0]), [(1,3), (2,3), (3,3), (4,3), (5,3), (6,3)]],
      [ np.array([0,0,0,1,0,0]), [(1,4), (2,4), (3,4), (4,4), (5,4), (6,4)]],
      [ np.array([0,0,0,0,1,0]), [(1,5), (2,5), (3,5), (4,5), (5,5), (6,5)]],
      [ np.array([0,0,0,0,0,1]), [(1,6), (2,6), (3,6), (4,6), (5,6), (6,6)]],
   ]

   }

Cij_symmetry['hexagonal'] = Cij_symmetry['trigonal_high']
Cij_symmetry[None] = Cij_symmetry['triclinic']

strain_patterns['hexagonal'] = strain_patterns['trigonal_high']
strain_patterns['tetragonal_high'] = strain_patterns['tetragonal_low'] = strain_patterns['tetragonal']
strain_patterns[None] = strain_patterns['triclinic']

def generate_strained_configs(at0, symmetry='triclinic', N_steps=5, delta=1e-2):
    """Generate a sequence of strained configurations"""

    if not symmetry in strain_patterns:
        raise ValueError('Unknown symmetry %s. Valid options are %s' % (symmetry, strain_patterns.keys()))

    for pindex, (pattern, fit_pairs) in enumerate(strain_patterns[symmetry]):
        for step in range(N_steps):
            strain = np.where(pattern == 1, delta*(step-(N_steps+1)/2.0), 0.0)
            at = at0.copy()
            T = strain_matrix(strain)
            at.set_lattice(np.dot(T,at.lattice), scale_positions=False)
            at.pos[:] = np.dot(T,at.pos)
            at.params['strain'] = T
            yield at


def calc_stress(configs, pot, relax=False, relax_tol=1e-3, relax_steps=100):
    """Given a sequence of configs, calculate stress on each one"""
    from quippy import GPA
    for at in configs:
        at2 = at.copy()
        at2.set_cutoff(pot.cutoff())
        at2.calc_connect()
        if relax:
            pot.minim(at2, 'cg', relax_tol, relax_steps, do_pos=True, do_lat=False)
        pot.calc(at2, virial=True)
        at2.params['stress'] = -at2.params['virial']*GPA/at2.cell_volume()
        yield at2


# Elastic constant calculation.

# Code adapted from elastics.py script, available from
# http://github.com/djw/elastic-constants
#
# Copyright (c) 2008, Dan Wilson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY DAN WILSON ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL DAN WILSON BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def fit_elastic_constants(configs, symmetry=None, N_steps=5, verbose=True, graphics=True):
    """Given a sequence of configs with strain and stress parameters, fit elastic constants C_ij"""

    def do_fit(index1, index2, stress, strain, patt):
        if verbose:
            print 'Fitting C_%d%d' % (index1, index2)
            print 'Strain %r' % strain[:,index2]
            print 'Stress %r' % stress[:,index1]

        cijFitted,intercept,r,tt,stderr = stats.linregress(strain[:,index2],stress[:,index1])

        if verbose:
            # print info about the fit
            print     'Cij (gradient)          :    ',cijFitted
            print     'Error in Cij            :    ', stderr
            if abs(r) > 0.9:
                print 'Correlation coefficient :    ',r
            else:
                print 'Correlation coefficient :    ',r, '     <----- WARNING'

        if graphics:
            import pylab
            # position this plot in a 6x6 grid
            sp = pylab.subplot(6,6,6*(index1-1)+index2)
            sp.set_axis_on()

            # change the labels on the axes
            xlabels = sp.get_xticklabels()
            pylab.setp(xlabels,'rotation',90,fontsize=7)
            ylabels = sp.get_yticklabels()
            pylab.setp(ylabels,fontsize=7)

            # colour the plot depending on the strain pattern
            colourDict = {1: '#BAD0EF', 2:'#FFCECE', 3:'#BDF4CB', 4:'#EEF093',5:'#FFA4FF',6:'#75ECFD'}
            sp.set_axis_bgcolor(colourDict[patt])

            # plot the data
            pylab.plot([strain[1,index2],strain[-1,index2]],[cijFitted*strain[1,index2]+intercept,cijFitted*strain[-1,index2]+intercept])
            pylab.plot(list(strain[:,index2]),list(stress[:,index1]),'ro')

        return cijFitted, stderr

    if not symmetry in strain_patterns:
        raise ValueError('Unknown symmetry %s. Valid options are %s' % (symmetry, strain_patterns.keys()))

    # There are 21 independent elastic constants
    Cijs = {}
    Cij_err = {}

    # Construct mapping from (i,j) to index into Cijs in range 1..21
    # (upper triangle only to start with)
    Cij_map = {}
    Cij_map_sym = {}
    for i in frange(6):
        for j in frange(i,6):
            Cij_map[(i,j)] = Cij_symmetry[None][i,j]
            Cij_map_sym[(i,j)] = Cij_symmetry[symmetry][i,j]

    # Reverse mapping, index 1..21 -> tuple (i,j) with i, j in range 1..6
    Cij_rev_map = dict(zip(Cij_map.values(), Cij_map.keys()))

    # Add the lower triangle to Cij_map, e.g. C21 = C12
    for (i1,i2) in Cij_map.keys():
        Cij_map[(i2,i1)] = Cij_map[(i1,i2)]
        Cij_map_sym[(i2,i1)] = Cij_map_sym[(i1,i2)]


    N_pattern = len(strain_patterns[symmetry])
    configs = iter(configs)

    strain = fzeros((N_pattern, N_steps, 6))
    stress = fzeros((N_pattern, N_steps, 6))

    if graphics:
        import pylab
        fig = pylab.figure(num=1, figsize=(9.5,8),facecolor='white')
        fig.clear()
        fig.subplots_adjust(left=0.07,right=0.97,top=0.97,bottom=0.07,wspace=0.5,hspace=0.5)

        for index1 in range(6):
            for index2 in range(6):
                # position this plot in a 6x6 grid
                sp = pylab.subplot(6,6,6*(index1)+index2+1)
                sp.set_axis_off()
                pylab.text(0.4,0.4, "n/a")

    # Fill in strain and stress arrays from config Atoms list
    for pindex, (pattern, fit_pairs) in fenumerate(strain_patterns[symmetry]):
        for step in frange(N_steps):
            at = configs.next()
            strain[pindex, step, :] = strain_vector(at.params['strain'])
            stress[pindex, step, :] = stress_vector(at.params['stress'])

    # Do the linear regression
    for pindex, (pattern, fit_pairs) in fenumerate(strain_patterns[symmetry]):
        for (index1, index2) in fit_pairs:
            fitted, err = do_fit(index1, index2, stress[pindex,:,:], strain[pindex,:,:], pindex)

            index = abs(Cij_map_sym[(index1, index2)])

            if not index in Cijs:
                if verbose:
                    print 'Setting C%d%d (%d) to %f +/- %f' % (index1, index2, index, fitted, err)
                Cijs[index] = [fitted]
                Cij_err[index] = [err]
            else:
                if verbose:
                    print 'Updating C%d%d (%d) with value %f +/- %f' % (index1, index2, index, fitted, err)
                Cijs[index].append(fitted)
                Cij_err[index].append(err)
            if verbose: print '\n'


    C = fzeros((6,6))
    C_err = fzeros((6,6))
    C_labels = fzeros((6,6),dtype='S4')
    C_labels[:] = '    '

    # Convert lists to mean
    for k in Cijs:
        Cijs[k] = np.mean(Cijs[k])

    # Combine statistical errors
    for k, v in Cij_err.iteritems():
        Cij_err[k] = np.sqrt(np.sum(np.array(v)**2))/np.sqrt(len(v))

    if symmetry.startswith('trigonal'):
        # Special case for trigonal lattice: C66 = (C11 - C12)/2
        Cijs[Cij_map[(6,6)]] = 0.5*(Cijs[Cij_map[(1,1)]]-Cijs[Cij_map[(1,2)]])
        Cij_err[Cij_map[(6,6)]] = np.sqrt(Cij_err[Cij_map[(1,1)]]**2 + Cij_err[Cij_map[(1,2)]]**2)

    # Generate the 6x6 matrix of elastic constants
    # - negative values signify a symmetry relation
    for i in frange(6):
        for j in frange(6):
            index = Cij_symmetry[symmetry][i,j]
            if index > 0:
                C[i,j] = Cijs[index]
                C_err[i,j] = Cij_err[index]
                C_labels[i,j] = ' C%d%d' % Cij_rev_map[index]
                C_err[i,j] = Cij_err[index]
            elif index < 0:
                C[i,j] = -Cijs[-index]
                C_err[i,j] = Cij_err[-index]
                C_labels[i,j] = '-C%d%d' % Cij_rev_map[-index]

    if verbose:
        print np.array2string(C_labels).replace("'","")
        print '\n = \n'
        print np.array2string(C, suppress_small=True, precision=2)
        print

        # Summarise the independent components of C_ij matrix
        printed = {}
        for i in frange(6):
            for j in frange(6):
                index = Cij_symmetry[symmetry][i,j]
                if index <= 0 or index in printed: continue
                print 'C_%d%d = %-4.2f +/- %-4.2f GPa' % (i, j, C[i,j], C_err[i,j])
                printed[index] = 1

    return C, C_err


def elastic_constants(at, sym='cubic', relax=True, verbose=True, graphics=True):
    """
    Compute elastic constants matrix C_ij using crystal symmetry

    Returns the 6x6 matrix :math:`C_{ij}` of configuration `at`
    with attached calculator assumming symmetry `sym` (default "cubic").
    """

    strained_configs = generate_strained_configs(at, sym)
    stressed_configs = calc_stress(strained_configs, relax=relax)
    C, C_err = fit_elastic_constants(stressed_configs, sym, verbose=verbose, graphics=graphics)

    return C
