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

class CubicElasticModuli:
    tol = 1e-6

    def __init__(self, C11, C12, C44):
        """
        Initialize a cubic system with elastic constants C11, C12, C44
        """

        self.la = C12
        self.mu = C44
        self.al = C11 - self.la - 2*self.mu

        A = np.eye(3, dtype=float)

        # Compute initial compliance matrix
        self.rotate(A)


    def rotate(self, A):
        """
        Compute the rotated compliance matrix
        """

        # Is this a rotation matrix?
        if np.sometrue(np.abs(np.dot(np.array(A), np.transpose(np.array(A))) - 
                              np.eye(3, dtype=float)) > self.tol ):
            raise RuntimeError('A does not describe a rotation.')

        C = np.zeros((3, 3, 3, 3), dtype=float)

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
                        for o in range(3):
                            h += self.al * A[i, o] * A[j, o] * A[k, o] * A[m, o]
                            #h += self.al * A[o, i] * A[o, j] * A[o, k] * A[o, m]
                        C[i, j, k, m] = h

        self.C = C

        return C


    def rotate2(self, A):
        """
        Compute the rotated compliance matrix
        """

        # Is this a rotation matrix?
        if np.sometrue(np.abs(np.dot(np.array(A), np.transpose(array(A))) - 
                              np.eye(3, dtype=float) ) > self.tol):
            raise RuntimeError('A does not describe a rotation.')

        C = np.zeros((3, 3, 3, 3), dtype=float)

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

        D = zeros( ( 3, 3, 3, 3 ), dtype=float )

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        h = 0.0
                        for a in range(3):
                            for b in range(3):
                                for c in range(3):
                                    for d in range(3):
                                        h += A[i, a]*A[j, b]*C[a, b, c, d]* \
                                             A[k, c]*A[l, d]
                        D[i, j, k, l] = h

        self.C = D

        return D


    def compliance(self):
        """
        Return the elastic constants - checks whether this is still cubic
        """

        t = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        C6 = np.zeros((6, 6), dtype=float)

        for a in range(6):
            for b in range(6):
                i, j = t[a]
                k, l = t[b]

                C6[a, b] = self.C[i, j, k, l]

        if np.sometrue(np.abs(C6 - np.transpose(C6)) > self.tol):
            raise RuntimeError('C6 not symmetric.')

        return C6


    def compliances_old(self):
        """
        Return the elastic constants - checks whether this is still cubic
        """        
        C11 = ( self.C[0, 0, 0, 0], self.C[1, 1, 1, 1], self.C[2, 2, 2, 2] )
        C12 = ( self.C[0, 0, 1, 1], self.C[0, 0, 2, 2], self.C[1, 1, 2, 2] )
        C44 = ( self.C[1, 2, 1, 2], self.C[0, 2, 0, 2], self.C[0, 1, 0, 1] )

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        if i == j == k == m:
                            if abs(self.C[i, j, k, m] - C11[i]) > self.tol:
                                raise RuntimeError('C[%i,%i,%i,%i] != C11[%i] (%f != %f)' % (i, j, k, m, i, self.C[i, j, k, m], C11[i]))
                        elif i == j and k == m:
                            if abs(self.C[i, j, k, m] - C12[i+k-1]) > self.tol:
                                raise RuntimeError('C[%i,%i,%i,%i] != C12[%i] (%f != %f)' % (i, j, k, m, i-k-1, self.C[i, j, k, m], C12[i+k-1]))
                        elif (i == k and j == m) or (i == m and j == k):
                            if abs(self.C[i, j, k, m] - C44[3-i-j]) > self.tol:
                                raise RuntimeError('C[%i,%i,%i,%i] != C44[%i] (%f != %f)' % (i, j, k, m, i+j-1, self.C[i, j, k, m], C44[i+j-1]))
                        else:
                            if abs(self.C[i, j, k, m] - 0.0) > self.tol:
                                raise RuntimError('C[%i,%i,%i,%i] = %f != 0' % (i, j, k, m, self.C[i, j, k, m]))

        return (C11, C12, C44)



    def stiffness(self):
        """
        Return the elastic constants - checks whether this is still cubic
        """

        C6 = self.compliance()

        S6 = inv(C6)

        return S6
