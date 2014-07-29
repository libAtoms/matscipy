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
        third_dir = np.array(third_dir) / np.sqrt(np.dot(third_dir,
                                                         third_dir))
        crack_surface = np.array(crack_surface) / \
            np.sqrt(np.dot(crack_surface, crack_surface))
        crack_front = np.array(crack_front) / \
            np.sqrt(np.dot(crack_front, crack_front))

        A = np.array([third_dir, crack_surface, crack_front])
        if np.linalg.det(A) < 0:
            third_dir = -third_dir
        A = np.array([third_dir, crack_surface, crack_front])

        self.E.rotate(A)

        self.crack = RectilinearAnisotropicCrack()

        S6 = self.E.compliance()

        if mode == self.PLANE_STRESS:
            self.crack.set_plane_stress(S6[0, 0], S6[1, 1], S6[0, 1],
                                        S6[0, 5], S6[1, 5], S6[5, 5])
        elif mode == self.PLANE_STRAIN:
            self.crack.set_plane_strain(S6[0, 0], S6[1, 1], S6[2, 2],
                                        S6[0, 1], S6[0, 2], S6[1, 2],
                                        S6[0, 5], S6[1, 5], S6[2, 5],
                                        S6[5, 5])


    def k1g(self, surface_energy):
        """
        Compute Griffith critical stress intensity in mode I fracture.

        Parameters
        ----------
        surface_energy : float
            Surface energy of the respective crystal surface.

        Returns
        -------
        k1g : float
            Stress intensity factor.
        """
        return self.crack.k1g(surface_energy)

    
    def k1gsqG(self):
        return self.crack.k1gsqG()


    def displacements_from_cylinder_coordinates(self, r, theta, k):
        """
        Displacement field in mode I fracture from cylindrical coordinates.
        """
        return self.crack.displacements(r, theta, k)


    def displacements_from_cartesian_coordinates(self, dx, dz, k):
        """
        Displacement field in mode I fracture from cartesian coordinates.
        """
        abs_dr = np.sqrt(dx*dx+dz*dz)
        theta = np.arctan2(dz, dx)
        return self.displacements_from_cylinder_coordinates(abs_dr, theta, k)


    def displacements(self, ref_x, ref_y, x0, y0, k):
        """
        Displacement field for a list of cartesian positions.

        Parameters
        ----------
        ref_x : array_like
            x-positions of the reference crystal.
        ref_y : array_like
            y-positions of the reference crystal.
        x0 : float
            x-coordinate of the crack tip.
        y0 : float
            y-coordinate of the crack tip.
        k : float
            Stress intensity factor.
        
        Returns
        -------
        ux : array_like
            x-displacements.
        uy : array_like
            y-displacements.
        """
        dx = ref_x - x0
        dy = ref_y - y0
        ux, uy = self.displacements_from_cartesian_coordinates(dx, dy, k)
        return ux, uy


    def displacement_residuals(self, x, y, ref_x, ref_y, x0, y0, k):
        """
        Return actual displacement field minus ideal displacement field.
        """
        u1x = x - ref_x
        u1y = y - ref_y
        u2x, u2y = self.displacements(ref_x, ref_y, x0, y0, k)
        return u1x - u2x, u1y - u2y


    def _residual(self, r0, x, y, ref_x, ref_y, k, mask):
        x0, y0 = r0
        dux, duy = self.displacement_residuals(x, y, ref_x, ref_y, x0, y0, k)
        return dux[mask]*dux[mask]+duy[mask]*duy[mask]


    def crack_tip_position(self, x, y, ref_x, ref_y, x0, y0, k, mask=None):
        """
        Return an estimate of the real crack tip position assuming the stress
        intensity factor is k.

        Parameters
        ----------
        x : array_like
            x-positions of the atomic system containing the crack.
        y : array_like
            y-positions of the atomic system containing the crack.
        ref_x : array_like
            x-positions of the reference crystal.
        ref_y : array_like
            y-positions of the reference crystal.
        x0 : float
            Initial guess for the x-coordinate of the crack tip.
        y0 : float
            Initial guess for the y-coordinate of the crack tip.
        k : float
            Stress intensity factor.
        mask : array_like, optional
            Marks the atoms to use for this calculation.

        Returns
        -------
        x0 : float
            x-coordinate of the crack tip.
        y0 : float
            y-coordinate of the crack tip.
        """
        if mask is None:
            mask = np.ones(len(a), dtype=bool)
        ( x0, y0 ), ier = leastsq(self._residual, ( x0, y0 ),
                                  args=(x, y, ref_x, ref_y, k, mask))
        if ier not in [ 1, 2, 3, 4 ]:
            raise RuntimeError('Could not find crack tip')
        return x0, y0


    def _residual_z(self, y0, x0, x, y, ref_x, ref_y, k, mask):
        dux, duy = self.displacement_residuals(x, y, ref_x, ref_y, x0, y0, k)
        return dux[mask]*dux[mask]+duy[mask]*duy[mask]


    def crack_tip_position_y(self, x, y, ref_x, ref_y, x0, y0, k, mask=None):
        """
        Return an estimate of the y-coordinate of the real crack tip position 
        assuming the stress intensity factor is k.

        Parameters
        ----------
        x : array_like
            x-positions of the atomic system containing the crack.
        y : array_like
            y-positions of the atomic system containing the crack.
        ref_x : array_like
            x-positions of the reference crystal.
        ref_y : array_like
            y-positions of the reference crystal.
        x0 : float
            Initial guess for the x-coordinate of the crack tip.
        y0 : float
            Initial guess for the y-coordinate of the crack tip.
        k : float
            Stress intensity factor.
        mask : array_like, optional
            Marks the atoms to use for this calculation.

        Returns
        -------
        y0 : float
            y-coordinate of the crack tip
        """
        if mask is None:
            mask = np.ones(len(a), dtype=bool)
        ( y0, ), ier = leastsq(self._residual_z, y0,
                          args=(x0, x, y, ref_x, ref_y, k, mask))
        if ier not in [ 1, 2, 3, 4 ]:
            raise RuntimeError('Could not find crack tip')
        return y0


    def scale_displacements(self, x, y, ref_x, ref_y, old_k, new_k):
        """
        Rescale atomic positions from stress intensity factor old_k to the new
        stress intensity factor new_k. This is useful for extrapolation of 
        relaxed positions.

        Parameters
        ----------
        x : array_like
            x-positions of the atomic system containing the crack.
        y : array_like
            y-positions of the atomic system containing the crack.
        ref_x : array_like
            x-positions of the reference crystal.
        ref_y : array_like
            y-positions of the reference crystal.
        old_k : float
            Current stress intensity factor.
        new_k : float
            Stress intensity factor for the output positions.

        Returns
        -------
        x : array_like
            New x-positions of the atomic system containing the crack.
        y : array_like
            New y-positions of the atomic system containing the crack.
        """
        return ref_x + new_k/old_k*(x-ref_x), ref_y + new_k/old_k*(y-ref_y)

