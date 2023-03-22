#
# Copyright 2020-2021 Lars Pastewka (U. Freiburg)
#           2020 Wolfram G. Nöhring (U. Freiburg)
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

# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
#                  Adrien Gola, Karlsruhe Institute of Technology
#                  Wolfram Nöhring, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================

import unittest

import numpy as np

import os

from matscipy.calculators.eam import io, average_atom

import matscipytest



###

class TestEAMAverageAtom(matscipytest.MatSciPyTestCase):

    tol = 1e-6
    
    def test_average_atom_Ni_Al(self):
        """Create A-atom potential for a Ni-Al eam/alloy potential and 15% Al

        The input potential is from Ref. [1]_ and was downloaded from
        the NIST database [2]_. The generated The generated A-atom
        potential is compared to a reference A-atom potential, which
        was created with an independent implementation. This is not
        a very strong test, but should capture some regressions.

        References
        ----------
        [1] G.P. Purja Pun, and Y. Mishin (2009), "Development of an
            interatomic potential for the Ni-Al system", Philosophical
            Magazine, 89(34-36), 3245-3267. DOI: 10.1080/14786430903258184.
        [2] https://www.ctcms.nist.gov/potentials/Download/2009--Purja-Pun-G-P-Mishin-Y--Ni-Al/2/Mishin-Ni-Al-2009.eam.alloy
        """
        input_table = "Mishin-Ni-Al-2009.eam.alloy"
        reference_table = "Mishin-Ni-Al-2009_reference_A-atom_Ni85Al15.eam.alloy"
        concentrations = np.array((0.85, 0.15))
        source, parameters, F, f, rep = io.read_eam(input_table)
        (new_parameters, new_F, new_f, new_rep) = average_atom.average_potential(
            concentrations, parameters, F, f, rep
        )
        ref_source, ref_parameters, ref_F, ref_f, ref_rep = io.read_eam(reference_table)
        diff_F = np.linalg.norm(ref_F - new_F)
        diff_f = np.linalg.norm(ref_f - new_f)
        diff_rep = np.linalg.norm(ref_rep - new_rep)
        print(diff_F, diff_f, diff_rep)
        self.assertTrue(diff_F < self.tol)
        self.assertTrue(diff_f < self.tol)
        self.assertTrue(diff_rep < self.tol)

    def test_average_atom_Fe_Cu_Ni(self):
        """Create A-atom potential for a Fe-Cu-Ni eam/alloy potential at equicomposition

        The input potential is from Ref. [1]_ and was downloaded from
        the NIST database [2]_. The generated The generated A-atom
        potential is compared to a reference A-atom potential, which
        was created with an independent implementation. This is not
        a very strong test, but should capture some regressions.

        References
        ----------
        [1] G. Bonny, R.C. Pasianot, N. Castin, and L. Malerba (2009), 
            "Ternary Fe-Cu-Ni many-body potential to model reactor 
            pressure vessel steels: First validation by simulated 
            thermal annealing", Philosophical Magazine, 89(34-36), 
            3531-3546. DOI: 10.1080/14786430903299824.
        [2] https://www.ctcms.nist.gov/potentials/Download/2009--Bonny-G-Pasianot-R-C-Castin-N-Malerba-L--Fe-Cu-Ni/1/FeCuNi.eam.alloy
        """
        input_table = "FeCuNi.eam.alloy"
        reference_table = "FeCuNi_reference_A-atom_Fe33Cu33Ni33.eam.alloy"
        concentrations = np.array((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
        source, parameters, F, f, rep = io.read_eam(input_table)
        (new_parameters, new_F, new_f, new_rep) = average_atom.average_potential(
            concentrations, parameters, F, f, rep
        )
        ref_source, ref_parameters, ref_F, ref_f, ref_rep = io.read_eam(reference_table)
        diff_F = np.linalg.norm(ref_F - new_F)
        diff_f = np.linalg.norm(ref_f - new_f)
        diff_rep = np.linalg.norm(ref_rep - new_rep)
        print(diff_F, diff_f, diff_rep)
        self.assertTrue(diff_F < self.tol)
        self.assertTrue(diff_f < self.tol)
        self.assertTrue(diff_rep < self.tol)

###

if __name__ == '__main__':
    unittest.main()
