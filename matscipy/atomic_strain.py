#
# Copyright 2014, 2020 Lars Pastewka (U. Freiburg)
#           2015, 2017 James Kermode (Warwick U.)
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
Compute deformation gradient tensor and D^2_min measure for non-affine
displacements.
See: Falk, Langer, Phys. Rev. E 57, 7192 (1998)
"""

import numpy as np
import scipy

from matscipy.neighbours import mic, neighbour_list

###

def get_XIJ(nat, i_now, dr_now, dr_old):
    """
    Calculates the X_{ij} matrix
    """
    # Do an element-wise outer product
    dr_dr = dr_now.reshape(-1,3,1)*dr_old.reshape(-1,1,3)

    xij = np.zeros([nat,3,3])
    for i in range(3):
        for j in range(3):
            # For each atom, sum over all neighbors
            xij[:,i,j] = np.bincount(i_now, weights=dr_dr[:,i,j])

    return xij


def get_YIJ(nat, i_now, dr_old):
    """
    Calculates the Y_{ij} matrix
    """
    # Just do an element-wise outer product
    dr_dr = dr_old.reshape(-1,3,1)*dr_old.reshape(-1,1,3)

    yij = np.zeros([nat,3,3])
    for i in range(3):
        for j in range(3):
            # For each atom, sum over all neighbors
            yij[:,i,j] = np.bincount(i_now, weights=dr_dr[:,i,j])

    return yij


def array_inverse(A):
    """
    Compute inverse for each matrix in a list of matrices.
    This is faster than calling numpy.linalg.inv for each matrix.
    """
    A = np.ascontiguousarray(A, dtype=float)
    n_eq = A.shape[1]
    identity = np.eye(n_eq)
    def lapack_inverse(a):
        b = np.copy(identity)
        lu, piv, x, info = scipy.linalg.lapack.dgesv(a, b)
        if info > 0:
            raise np.linalg.LinAlgError('Singular matrix')
        return x

    return np.array([lapack_inverse(a) for a in A])


def get_delta_plus_epsilon_dgesv(nat, i_now, dr_now, dr_old):
    """
    Calculate delta_ij+epsilon_ij, i.e. the deformation gradient matrix
    """
    XIJ = get_XIJ(nat, i_now, dr_now, dr_old)
    YIJ = get_YIJ(nat, i_now, dr_old)

    YIJ_invert = array_inverse(YIJ)

    # Perform sum_k X_ik Y_jk^-1
    epsilon = np.sum(XIJ.reshape(-1,3,1,3)*YIJ_invert.reshape(-1,1,3,3), axis=3)

    return epsilon


def get_delta_plus_epsilon(nat, i_now, dr_now, dr_old):
    """
    Calculate delta_ij+epsilon_ij, i.e. the deformation gradient matrix
    """
    epsilon = []
    for i in range(nat):
        mask = i_now==i
        x, residuals, rank, s = np.linalg.lstsq(dr_old[mask], dr_now[mask])
        epsilon += [ x.T ]

    return np.array(epsilon)


def get_D_square_min(atoms_now, atoms_old, i_now, j_now, delta_plus_epsilon=None):
    """
    Calculate the D^2_min norm of Falk and Langer
    """
    nat = len(atoms_now)
    assert len(atoms_now) == len(atoms_old)

    pos_now = atoms_now.positions
    pos_old = atoms_old.positions

    # Compute current and old distance vectors. Note that current distance
    # vectors cannot be taken from the neighbor calculation, because neighbors
    # are calculated from the sheared cell while these distance need to come
    # from the unsheared cell. Taking the distance from the unsheared cell
    # make periodic boundary conditions (and flipping of cell) a lot easier.
    dr_now = mic(pos_now[i_now] - pos_now[j_now], atoms_now.cell)
    dr_old = mic(pos_old[i_now] - pos_old[j_now], atoms_old.cell)

    # Sanity check: Shape needs to be identical!
    assert dr_now.shape == dr_old.shape

    if delta_plus_epsilon is None:
        # Get minimum strain tensor
        delta_plus_epsilon = get_delta_plus_epsilon(nat, i_now, dr_now, dr_old)

    # Spread epsilon out for each neighbor index
    delta_plus_epsilon_n = delta_plus_epsilon[i_now]

    # Compute D^2_min (residual of the least squares fit)
    residual_n = np.sum(
        (
        dr_now-
        np.sum(delta_plus_epsilon_n.reshape(-1,3,3)*dr_old.reshape(-1,1,3),
               axis=2)
        )**2,
        axis=1)

    # For each atom, sum over all neighbors
    residual = np.bincount(i_now, weights=residual_n)

    return delta_plus_epsilon, residual


def atomic_strain(atoms_now, atoms_old, cutoff=None, neighbours=None):
    """
    Calculate deformation gradient tensor and D^2_min measure for non-affine
    displacements.
    See: Falk, Langer, Phys. Rev. B 57, 7192 (1998)

    Parameters:
    -----------
    atoms_now : ase.Atoms
        Current atomic configuration
    atoms_old : ase.Atoms
        Reference atomic configuration
    cutoff : float
        Neighbor list cutoff.
    neighbours : ( array_like, array_like )
        Neighbor list. Automatically computed if not provided.

    Returns:
    --------
    delta_plus_epsilon : array
        3x3 deformation gradient tensor for each atom.
    residual : array
        D^2_min norm for each atom
    """

    if neighbours is None:
        if cutoff is None:
            raise ValueError('Please provide either neighbor list or neighbor '
                             'list cutoff.')

        # Get neighbours
        i_now, j_now = neighbour_list("ij", atoms_now, cutoff)
    elif cutoff is not None:
        raise ValueError('Please provide either neighbor list or neighbor '
                         'list cutoff, not both.')
    else:
        i_now, j_now = neighbours

    # Get strain gradient tensor and D square values
    delta_plus_epsilon, residual = get_D_square_min(atoms_now, atoms_old, i_now,
                                                    j_now)

    return delta_plus_epsilon, residual
