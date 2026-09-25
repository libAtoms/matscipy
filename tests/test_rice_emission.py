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
    barnett_lothe_L,
    hill_poisson_ratio,
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
    """Test k1e_iso and k1e_aniso, the critical stress intensity factors for
    dislocation emission, against Rice's isotropic result and k1g."""

    E, nu = 100.0, 0.3  # GPa
    angles = [(theta, phi) for theta in (np.pi / 6, np.pi / 4, np.radians(70.5))
              for phi in (0.0, np.pi / 6)]

    def _make_isotropic_crack(self, E, nu):
        """Create a CubicCrystalCrack for an isotropic material."""
        K = E / (3.0 * (1 - 2 * nu))
        C44 = E / (2.0 * (1 + nu))
        C11 = K + 4.0 * C44 / 3.0
        C12 = K - 2.0 * C44 / 3.0
        return CubicCrystalCrack([1, 0, 0], [0, 1, 0], C11, C12, C44)

    def test_hill_poisson_ratio(self):
        crack = self._make_isotropic_crack(self.E, self.nu)
        self.assertAlmostEqual(hill_poisson_ratio(crack.C), self.nu, places=10)
        # orientation independent for an anisotropic crystal
        nus = [hill_poisson_ratio(CubicCrystalCrack(s, f, 243., 145., 116.).C)
               for s, f in (([1, 1, 0], [0, 0, 1]), ([1, 1, 1], [1, -1, 0]))]
        self.assertAlmostEqual(nus[0], nus[1], places=10)

    def test_barnett_lothe_L_isotropic(self):
        """Energy tensor of an isotropic material: mu/(1-nu) for edge, mu for screw."""
        crack = self._make_isotropic_crack(self.E, self.nu)
        mu = self.E / (2 * (1 + self.nu))
        np.testing.assert_allclose(barnett_lothe_L(crack.C),
                                   np.diag([mu / (1 - self.nu)] * 2 + [mu]), rtol=1e-12, atol=1e-10)

    def test_k1e_iso_vs_griffith(self):
        """K1e/K1c must equal sqrt(G_Ie / G_c) with Rice's G_Ie and G_c = 2 gamma_s,
        i.e. k1e_iso is in the same units as k1g."""
        crack = self._make_isotropic_crack(self.E, self.nu)
        gamma = 1.0
        for theta, phi in self.angles:
            G_Ie = 8 * gamma * (1 + (1 - self.nu) * np.tan(phi)**2) / (
                (1 + np.cos(theta)) * np.sin(theta)**2)
            self.assertAlmostEqual(crack.k1e_iso(gamma, phi, theta) / crack.k1g(gamma),
                                   np.sqrt(G_Ie / (2 * gamma)), places=8)

    def test_k1e_iso_plane_strain_isotropic(self):
        """For an isotropic material K = sqrt(G E / (1 - nu^2))."""
        crack = self._make_isotropic_crack(self.E, self.nu)
        theta, phi = np.pi / 4, 0.0
        G_Ie = 8 / ((1 + np.cos(theta)) * np.sin(theta)**2)
        self.assertAlmostEqual(crack.k1e_iso(1.0, phi, theta),
                               np.sqrt(G_Ie * self.E / (1 - self.nu**2)), places=6)

    def test_k1e_aniso_isotropic_limit(self):
        """Beltz-Rice anisotropic result reduces to Rice's isotropic one,
        including for slip directions with a screw component (phi != 0)."""
        crack = self._make_isotropic_crack(self.E, self.nu)
        for theta, phi in self.angles:
            self.assertAlmostEqual(crack.k1e_aniso(1.0, phi, theta) /
                                   crack.k1e_iso(1.0, phi, theta), 1.0, places=6)

    def test_k1e_scales_as_sqrt_gamma(self):
        crack = CubicCrystalCrack([1, 1, 0], [0, 0, 1], 243., 145., 116.)
        theta = np.radians(54.7)
        for k1e in (crack.k1e_iso, crack.k1e_aniso):
            self.assertAlmostEqual(k1e(4.0, 0.0, theta) / k1e(1.0, 0.0, theta), 2.0,
                                   places=8)

    def test_k1e_aniso_depends_on_orientation(self):
        """For an anisotropic (Fe-like, GPa) crystal the Beltz-Rice K1e differs
        between crack systems, while the isotropic estimate does not change
        relative to k1g."""
        theta = np.radians(54.7)
        ratios_aniso, ratios_iso = [], []
        for surface, front in (([1, 1, 0], [0, 0, 1]), ([0, 0, 1], [1, 1, 0])):
            crack = CubicCrystalCrack(surface, front, 243., 145., 116.)
            ratios_aniso.append(crack.k1e_aniso(1.0, 0.0, theta) / crack.k1g(1.0))
            ratios_iso.append(crack.k1e_iso(1.0, 0.0, theta) / crack.k1g(1.0))
        # reference values from the Stroh eigenvector route (agrees with the
        # Barnett-Lothe integral to ~1e-15 where the eigenproblem is well conditioned)
        np.testing.assert_allclose(ratios_aniso, [1.7942, 2.2715], atol=1e-4)
        self.assertAlmostEqual(ratios_iso[0], ratios_iso[1], places=8)

    def test_explicit_nu(self):
        crack = self._make_isotropic_crack(self.E, self.nu)
        self.assertAlmostEqual(crack.k1e_iso(1.0, 0.3, 0.8, nu=self.nu),
                               crack.k1e_iso(1.0, 0.3, 0.8), places=10)

    def test_missing_elastic_constants(self):
        crack = RectilinearAnisotropicCrack()
        with self.assertRaises(ValueError):
            crack.k1e_aniso(1.0, 0.0, np.pi / 4)
        with self.assertRaises(ValueError):
            crack.k1e_iso(1.0, 0.0, np.pi / 4)


if __name__ == '__main__':
    unittest.main()
