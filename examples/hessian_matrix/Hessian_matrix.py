#
# Copyright 2021 Lars Pastewka (U. Freiburg)
#           2019 Jan Griesser (U. Freiburg)
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
from scipy.sparse import csr_matrix
from ase.io import NetCDFTrajectory
from ase.io import read
import matscipy.calculators.pair_potential as calculator


# Load atomic configuration
MinStructure = NetCDFTrajectory(
    "structure/PBC.KA.N256.Min.nc", "r", keep_open=True)

# Paramters for a binary Kob-Anderson glass.
# Kob, Walter, and Hans C. Andersen. Phys. Rev. E 51.5 (1995): 4626.
# The interaction is modeled via a quadratic shifted Lennard-Jones potential.
parameters = {(1, 1): calculator.LennardJonesQuadratic(1, 1, 3), (1, 2): calculator.LennardJonesQuadratic(
    1.5, 0.8, 2.4), (2, 2): calculator.LennardJonesQuadratic(0.5, 0.88, 2.64)}
a = calculator.PairPotential(parameters)

# Exemplary code for the calculation of the full hessian matrix.
# Sparse
H_sparse = a.calculate_hessian_matrix(
    MinStructure[len(MinStructure)-1], "sparse")

# Dense
H_dense = a.calculate_hessian_matrix(
    MinStructure[len(MinStructure)-1], "dense")

# Compute the only rows of the Hessian matrix which correspong to atom Ids = 5,6
H_sparse1 = a.calculate_hessian_matrix(
    MinStructure[len(MinStructure)-1], "sparse", limits=(5, 7))
