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

import math

import numpy as np
try:
    from scipy.optimize import brentq, leastsq
except:
    print 'Warning: no scipy'

from matscipy.elasticity import CubicElasticModuli

###

class RectilinearAnisotropicCrack:
    """
    Near field solution for a crack in a rectilinear anisotropic elastic medium.
    See:
    G. C. Sih, P. C. Paris and G. R. Irwin, Int. J. Frac. Mech. 1, 189 (1965)
    """
    def __init__(self):
        self.a11 = None
        self.a22 = None
        self.a12 = None
        self.a16 = None
        self.a26 = None
        self.a66 = None


    def set_plane_stress(self, a11, a22, a12, a16, a26, a66):
        self.a11 = a11
        self.a22 = a22
        self.a12 = a12
        self.a16 = a16
        self.a26 = a26
        self.a66 = a66

        self._init_crack()


    def set_plane_strain(self, b11, b22, b33, b12, b13, b23, b16, b26, b36, 
                         b66):
        self.a11 = b11 - (b13*b13)/b33
        self.a22 = b22 - (b23*b23)/b33
        self.a12 = b12 - (b13*b23)/b33
        self.a16 = b16 - (b13*b36)/b33
        self.a26 = b26 - (b23*b36)/b33
        self.a66 = b66

        self._init_crack()


    def _init_crack(self):
        """
        Initialize dependend parameters.
        """
        p = np.poly1d( [ self.a11, -2*self.a16, 2*self.a12+self.a66,
                         -2*self.a26, self.a22 ] )

        mu1, mu2, mu3, mu4 = p.r

        if mu1 != mu2.conjugate():
            raise RuntimeError('Roots not in pairs.')

        if mu3 != mu4.conjugate():
            raise RuntimeError('Roots not in pairs.')

        self.mu1 = mu1
        self.mu2 = mu3

        self.p1  = self.a11*(self.mu1**2) + self.a12 - self.a16*self.mu1
        self.p2  = self.a11*(self.mu2**2) + self.a12 - self.a16*self.mu2

        self.q1  = self.a12*self.mu1 + self.a22/self.mu1 - self.a26
        self.q2  = self.a12*self.mu2 + self.a22/self.mu2 - self.a26

        self.h1 = 1/(self.mu1 - self.mu2)
        self.h2 = self.mu1 * self.p2
        self.h3 = self.mu2 * self.p1
        self.h4 = self.mu1 * self.q2
        self.h5 = self.mu2 * self.q1


    def displacements(self, r, theta, k):
        """
        Displacement field in mode I fracture.
        """

        h1 = k * np.sqrt(2*r)

        h2 = ( np.cos(theta) + self.mu2*np.sin(theta) )**0.5
        h3 = ( np.cos(theta) + self.mu1*np.sin(theta) )**0.5

        u = h1 * ( self.h1 * ( self.h2 * h2 - self.h3 * h3 ) ).real
        v = h1 * ( self.h1 * ( self.h4 * h2 - self.h5 * h3 ) ).real

        return u, v


    def _f(self, theta, v):
        h2 = ( cos(theta) + self.mu2*sin(theta) )**0.5
        h3 = ( cos(theta) + self.mu1*sin(theta) )**0.5

        return v - ( self.h2 * h2 - self.h3 * h3 ).real/ \
            ( self.h4 * h2 - self.h5 * h3 ).real


    def rtheta(self, u, v, k):
        """
        Invert displacement field in mode I fracture, i.e. compute r and theta
        from displacements.
        """

        # u/v = (self.h2*h2 - self.h3*h3)/(self.h4*h2-self.h5*h3)
        theta = brentq(self._f, -pi, pi, args=(u/v))

        h1 = k * sqrt(2*r)

        h2 = ( cos(theta) + self.mu2*sin(theta) )**0.5
        h3 = ( cos(theta) + self.mu1*sin(theta) )**0.5

        sqrt_2_r = ( self.h1 * ( self.h2 * h2 - self.h3 * h3 ) ).real/k
        r1 = sqrt_2_r**2/2
        sqrt_2_r = ( self.h1 * ( self.h4 * h2 - self.h5 * h3 ) ).real/k
        r2 = sqrt_2_r**2/2

        print r1, r2, theta

        return ( (r1+r2)/2, theta )


    def k1g(self, surface_energy):
        """
        K1G, Griffith critical stress intensity in mode I fracture
        """

        return math.sqrt(-4*surface_energy / \
                         (math.pi*self.a22* \
                          ((self.mu1+self.mu2)/(self.mu1*self.mu2)).imag))


    def k1gsqG(self):
        return -2/(self.a22*((self.mu1+self.mu2)/(self.mu1*self.mu2)).imag)

###

class CubicCrystalCrack:
    """
    Crack in a cubic crystal.
    """

    PLANE_STRESS = 0
    PLANE_STRAIN = 1

    def __init__(self, C11, C12, C44, crack_surface, crack_front,
                 mode = PLANE_STRAIN):
        """
        Initialize a crack in a cubic crystal with elastic constants C11, C12
        and C44. The crack surface is given by crack_surface, the cracks runs
        in the plane given by crack_front.
        """

        self.E = CubicElasticModuli(C11, C12, C44)

        # x (third_dir) - direction in which the crack is running
        # y (crack_surface) - free surface that forms due to the crack
        # z (crack_front) - direction of the crack front
        third_dir = np.cross(crack_surface, crack_front)

        self.third_dir = np.array(third_dir) / np.sqrt(np.dot(third_dir,
                                                              third_dir))
        self.crack_surface = np.array(crack_surface) / \
            np.sqrt(np.dot(crack_surface, crack_surface))
        self.crack_front = np.array(crack_front) / \
            np.sqrt(np.dot(crack_front, crack_front))

        A = np.array([self.third_dir, self.crack_surface, self.crack_front])

        self.E.rotate(A)

        self.crack = RectilinearAnisotropicCrack()

        C6 = self.E.stiffness()

        if mode == self.PLANE_STRESS:
            self.crack.set_plane_stress(C6[0, 0], C6[1, 1], C6[0, 1],
                                        C6[0, 5], C6[1, 5], C6[5, 5])
        elif mode == self.PLANE_STRAIN:
            self.crack.set_plane_strain(C6[0, 0], C6[1, 1], C6[2, 2],
                                        C6[0, 1], C6[0, 2], C6[1, 2],
                                        C6[0, 5], C6[1, 5], C6[2, 5],
                                        C6[5, 5])


    def k1g(self, surface_energy):
        """
        K1G, Griffith critical stress intensity in mode I fracture
        """
        return self.crack.k1g(surface_energy)

    
    def k1gsqG(self):
        return self.crack.k1gsqG()


    def displacements_from_cylinder_coordinates(self, r, theta, k):
        """
        Displacement field in mode I fracture
        """
        return self.crack.displacements(r, theta, k)


    def displacements_from_cartesian_coordinates(self, dx, dz, k):
        """
        Displacement field in mode I fracture
        """
        abs_dr = np.sqrt(dx*dx+dz*dz)
        theta = np.arctan2(dz, dx)
        return self.displacements_from_cylinder_coordinates(abs_dr, theta, k)


    def displacements(self, ref_pos, r0, k):
        """
        Returns the displacement field for a list of positions
        """
        dx, dy, dz = (ref_pos - r0.reshape(-1,3)).T
        ux, uy = self.displacements_from_cartesian_coordinates(dx, dz, k)
        return np.transpose([ux, np.zeros_like(ux), uy])


    def displacement_residuals(self, pos, ref_pos, r0, k):
        """
        Return actual displacement field minus ideal displacement field.
        """
        u1 = pos - ref_pos
        u2 = self.displacements(ref_pos, r0, k)
        return u1 - u2


    def _residual(self, r0, pos, ref_pos, k, mask):
        x0, z0 = r0
        r0 = np.array([x0, 0.0, z0])
        du = self.displacement_residuals(pos, ref_pos, r0, k)
        return (du[mask]*du[mask]).sum(axis=1)


    def crack_tip_position(self, pos, ref_pos, r0, k, mask=None):
        """
        Return an estimate of the real crack tip position assuming the stress
        intensity factor is k. mask marks the atoms to use for this calculation.
        r0 is the initial guess for the crack tip position.
        """
        if mask is None:
            mask = np.ones(len(a), dtype=bool)
        x0, y0, z0 = r0
        r0 = np.array([x0, z0])
        ( x0, z0 ), ier = leastsq(self._residual, r0,
                                  args=(pos, ref_pos, k, mask))
        if ier not in [ 1, 2, 3, 4 ]:
            raise RuntimeError('Could not find crack tip')
        return x0, z0


    def _residual_z(self, z0, x0, pos, ref_pos, k, mask):
        r0 = np.array([x0, 0.0, z0])
        du = self.displacement_residuals(pos, ref_pos, r0, k)
        return (du[mask]*du[mask]).sum(axis=1)


    def crack_tip_position_z(self, pos, ref_pos, r0, k, mask=None):
        """
        Return an estimate of the real crack tip position assuming the stress
        intensity factor is k. mask marks the atoms to use for this calculation.
        r0 is the initial guess for the crack tip position.
        """
        if mask is None:
            mask = np.ones(len(a), dtype=bool)
        x0, y0, z0 = r0
        ( z0, ), ier = leastsq(self._residual_z, z0,
                          args=(x0, pos, ref_pos, k, mask))
        if ier not in [ 1, 2, 3, 4 ]:
            raise RuntimeError('Could not find crack tip')
        return z0


    def scale_displacements(self, pos, ref_pos, old_k, new_k):
        """
        Rescale atomic positions from stress intensity factor old_k to the new
        stress intensity factor new_k. This is useful for extrapolation of 
        relaxed positions.
        """
        return ref_pos + new_k/old_k*(pos-ref_pos)

