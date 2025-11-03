#
# Copyright 2025 Lars Pastewka (U. Freiburg)
#           2025 James Kermode (Warwick U.)
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

import unittest
import numpy as np

from matscipy.fracture_mechanics.crack import RectilinearAnisotropicCrack


class TestRectilinearAnisotropicCrack(unittest.TestCase):
    """Test the RectilinearAnisotropicCrack class."""

    def test_plane_strain_reduction_formula(self):
        """
        Test that the plane strain reduction formula is applied consistently
        to all elastic constants.

        The plane strain reduction from 3D compliance matrix (b_ij) to
        2D compliance matrix (a_ij) follows the formula:
            a_ij = b_ij - (b_i3 * b_j3) / b33

        This test verifies that all components, including a66, follow this
        pattern. See Andric & Curtin (2019), Eq. 21:
        P Andric and W A Curtin 2019 Modelling Simul. Mater. Sci. Eng. 27 013001
        https://iopscience.iop.org/article/10.1088/1361-651X/aae40c/meta

        Related to GitHub issue #275.
        """
        crack = RectilinearAnisotropicCrack()

        # Use arbitrary compliance values where b36 != 0
        # These don't need to represent a physical material, just test the formula
        b11, b22, b33 = 1.0, 2.0, 3.0
        b12, b13, b23 = 0.5, 0.3, 0.4
        b16, b26, b36 = 0.2, 0.25, 0.35  # Non-zero b36 is critical for this test
        b66 = 1.5

        crack.set_plane_strain(b11, b22, b33, b12, b13, b23,
                               b16, b26, b36, b66)

        # Verify the reduction formula is applied consistently to all components
        np.testing.assert_allclose(crack.a11, b11 - (b13 * b13) / b33,
                                   err_msg="a11 reduction formula incorrect")
        np.testing.assert_allclose(crack.a22, b22 - (b23 * b23) / b33,
                                   err_msg="a22 reduction formula incorrect")
        np.testing.assert_allclose(crack.a12, b12 - (b13 * b23) / b33,
                                   err_msg="a12 reduction formula incorrect")
        np.testing.assert_allclose(crack.a16, b16 - (b13 * b36) / b33,
                                   err_msg="a16 reduction formula incorrect")
        np.testing.assert_allclose(crack.a26, b26 - (b23 * b36) / b33,
                                   err_msg="a26 reduction formula incorrect")

        # This is the critical test - a66 should also follow the pattern
        # Following the pattern: a66 = b66 - (b36 * b36) / b33
        # (Note: b63 = b36 due to matrix symmetry)
        np.testing.assert_allclose(crack.a66, b66 - (b36 * b36) / b33,
                                   err_msg="a66 reduction formula incorrect - should be b66 - (b36*b36)/b33")

    def test_plane_strain_zero_coupling(self):
        """
        Test plane strain reduction when b36 = 0 (no coupling).

        When b36 = 0, the a66 = b66 - (b36*b36)/b33 simplifies to a66 = b66,
        so both the buggy and correct implementations give the same result.
        This test ensures we don't break this special case.
        """
        crack = RectilinearAnisotropicCrack()

        b11, b22, b33 = 1.0, 2.0, 3.0
        b12, b13, b23 = 0.5, 0.3, 0.4
        b16, b26, b36 = 0.2, 0.25, 0.0  # b36 = 0
        b66 = 1.5

        crack.set_plane_strain(b11, b22, b33, b12, b13, b23,
                               b16, b26, b36, b66)

        # When b36 = 0, a66 should equal b66
        np.testing.assert_allclose(crack.a66, b66)

    def test_plane_stress_unchanged(self):
        """
        Test that plane stress formula remains unchanged.

        Plane stress directly uses the compliance components without reduction.
        """
        crack = RectilinearAnisotropicCrack()

        a11, a22, a12 = 1.0, 2.0, 0.5
        a16, a26, a66 = 0.2, 0.25, 1.5

        crack.set_plane_stress(a11, a22, a12, a16, a26, a66)

        # Plane stress should directly assign values
        np.testing.assert_allclose(crack.a11, a11)
        np.testing.assert_allclose(crack.a22, a22)
        np.testing.assert_allclose(crack.a12, a12)
        np.testing.assert_allclose(crack.a16, a16)
        np.testing.assert_allclose(crack.a26, a26)
        np.testing.assert_allclose(crack.a66, a66)


if __name__ == '__main__':
    unittest.main()
