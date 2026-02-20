#
# Copyright 2026 James Kermode (Warwick U.)
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
Tests for Rice-Thompson dislocation emission criterion methods
on RectilinearAnisotropicCrack and CubicCrystalCrack.

References:
    J. R. Rice, JMPS (1992)
    G. E. Beltz and J. R. Rice, JMPS (1994)
    P. Andric and W. A. Curtin, JMPS (2017)
"""

import unittest
import numpy as np

from matscipy.fracture_mechanics.crack import (
    RectilinearAnisotropicCrack,
    CubicCrystalCrack,
)


class TestRiceEmissionEnergies(unittest.TestCase):
    """Test the g1e and g1e_fcc energy release rate methods."""

    def setUp(self):
        self.crack = RectilinearAnisotropicCrack()

    def test_g1e_returns_max_gamma(self):
        """g1e should simply return the unstable stacking fault energy."""
        for gamma in [0.1, 0.5, 1.0, 2.5]:
            self.assertAlmostEqual(self.crack.g1e(gamma), gamma)

    def test_g1e_fcc_low_surface_energy(self):
        """When surface_energy <= 3.45 * max_gamma, g1e_fcc = max_gamma."""
        max_gamma = 1.0
        # At the boundary: surface_energy = 3.45 * max_gamma
        self.assertAlmostEqual(self.crack.g1e_fcc(3.45, max_gamma), max_gamma)
        # Below the boundary
        self.assertAlmostEqual(self.crack.g1e_fcc(3.0, max_gamma), max_gamma)
        self.assertAlmostEqual(self.crack.g1e_fcc(1.0, max_gamma), max_gamma)

    def test_g1e_fcc_high_surface_energy(self):
        """When surface_energy > 3.45 * max_gamma,
        g1e_fcc = 0.145 * surface_energy + 0.5 * max_gamma (Andric & Curtin)."""
        max_gamma = 1.0
        surface_energy = 4.0  # > 3.45
        expected = 0.145 * surface_energy + 0.5 * max_gamma
        self.assertAlmostEqual(
            self.crack.g1e_fcc(surface_energy, max_gamma), expected
        )

    def test_g1e_fcc_continuity_at_boundary(self):
        """Check that the two branches are close at the transition point."""
        max_gamma = 1.0
        # Just below boundary
        se_below = 3.45 * max_gamma
        g_below = self.crack.g1e_fcc(se_below, max_gamma)
        # Just above boundary
        se_above = 3.45 * max_gamma + 0.01
        g_above = self.crack.g1e_fcc(se_above, max_gamma)
        # The Andric-Curtin formula is designed to be approximately continuous
        self.assertAlmostEqual(g_below, g_above, places=2)


class TestRiceEmissionK1e(unittest.TestCase):
    """Test k1e_iso and k1e_aniso for computing critical stress intensity
    factor for dislocation emission."""

    def _make_isotropic_crack(self, E, nu):
        """Create a CubicCrystalCrack for an isotropic material."""
        K = E / (3.0 * (1 - 2 * nu))
        C44 = E / (2.0 * (1 + nu))
        C11 = K + 4.0 * C44 / 3.0
        C12 = K - 2.0 * C44 / 3.0
        return CubicCrystalCrack([1, 0, 0], [0, 1, 0], C11, C12, C44)

    def test_k1e_iso_positive(self):
        """K1e from isotropic formula should be positive and finite."""
        crack = self._make_isotropic_crack(E=100.0, nu=0.3)
        max_gamma = 1.0
        phi = 0.0  # pure edge
        theta = np.pi / 4  # 45 degree slip plane
        k1e = crack.crack.k1e_iso(max_gamma, phi, theta)
        self.assertTrue(np.isfinite(k1e))
        self.assertGreater(k1e, 0.0)

    def test_k1e_aniso_positive(self):
        """K1e from anisotropic formula should be positive and finite."""
        crack = self._make_isotropic_crack(E=100.0, nu=0.3)
        max_gamma = 1.0
        phi = 0.0
        theta = np.pi / 4
        k1e = crack.crack.k1e_aniso(max_gamma, phi, theta)
        self.assertTrue(np.isfinite(k1e))
        self.assertGreater(k1e, 0.0)

    def test_k1e_increases_with_gamma(self):
        """K1e should increase with increasing unstable stacking fault energy."""
        crack = self._make_isotropic_crack(E=100.0, nu=0.3)
        phi = 0.0
        theta = np.pi / 4
        k1e_low = crack.crack.k1e_iso(0.5, phi, theta)
        k1e_high = crack.crack.k1e_iso(2.0, phi, theta)
        self.assertGreater(k1e_high, k1e_low)

    def test_k1e_iso_phi_zero_vs_nonzero(self):
        """Non-zero phi (screw component) should increase K1e for nu < 1."""
        crack = self._make_isotropic_crack(E=100.0, nu=0.3)
        max_gamma = 1.0
        theta = np.pi / 4
        k1e_edge = crack.crack.k1e_iso(max_gamma, 0.0, theta)
        k1e_mixed = crack.crack.k1e_iso(max_gamma, np.pi / 6, theta)
        # For nu < 1, the (1 + (1-nu)*tan^2(phi)) factor means
        # non-zero phi increases G1e and hence K1e
        self.assertGreater(k1e_mixed, k1e_edge)

    def test_k1e_aniso_cubic_material(self):
        """Test k1e_aniso with an anisotropic cubic material (alpha-Fe)."""
        # BCC iron elastic constants in eV/A^3
        # C11=1.43, C12=0.85, C44=0.71 (approximate, in eV/A^3)
        C11, C12, C44 = 1.43, 0.85, 0.71
        crack = CubicCrystalCrack([1, 1, 0], [0, 0, 1], C11, C12, C44)
        max_gamma = 0.1  # approximate unstable stacking fault energy
        phi = 0.0
        theta = np.radians(54.7)  # {110} slip inclined to {001} crack plane
        k1e = crack.crack.k1e_aniso(max_gamma, phi, theta)
        self.assertTrue(np.isfinite(k1e))
        self.assertGreater(k1e, 0.0)

    def test_k1e_iso_scales_as_sqrt_gamma(self):
        """K1e should scale as sqrt(max_gamma) for fixed geometry."""
        crack = self._make_isotropic_crack(E=100.0, nu=0.3)
        phi = 0.0
        theta = np.pi / 4
        k1e_1 = crack.crack.k1e_iso(1.0, phi, theta)
        k1e_4 = crack.crack.k1e_iso(4.0, phi, theta)
        # K1e ~ sqrt(gamma), so K1e(4) / K1e(1) should be 2
        self.assertAlmostEqual(k1e_4 / k1e_1, 2.0, places=5)

    def test_k1e_aniso_scales_as_sqrt_gamma(self):
        """K1e_aniso should also scale as sqrt(max_gamma)."""
        crack = self._make_isotropic_crack(E=100.0, nu=0.3)
        phi = 0.0
        theta = np.pi / 4
        k1e_1 = crack.crack.k1e_aniso(1.0, phi, theta)
        k1e_4 = crack.crack.k1e_aniso(4.0, phi, theta)
        self.assertAlmostEqual(k1e_4 / k1e_1, 2.0, places=5)

    def test_k1g_scales_as_sqrt_surface_energy(self):
        """Sanity check: k1g should scale as sqrt(surface_energy)."""
        crack = self._make_isotropic_crack(E=100.0, nu=0.3)
        k1g_1 = crack.k1g(1.0)
        k1g_4 = crack.k1g(4.0)
        self.assertAlmostEqual(k1g_4 / k1g_1, 2.0, places=5)


if __name__ == '__main__':
    unittest.main()
