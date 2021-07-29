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
