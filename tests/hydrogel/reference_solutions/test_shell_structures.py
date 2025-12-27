



import numpy as np
import pytest
from matscipy.calculators.hydrogel.reference_solutions.lattice_shell_structures import GraphiteShellStructure


class TestGraphite():
    @pytest.fixture(scope="class")
    def graphite_shellstructure(self):
        """Fixture to create GraphiteShellStructure instance once per test class."""
        return GraphiteShellStructure(cutoff=4.0)

    def test_Z_first_shells(self, graphite_shellstructure):
        expected_Z = [3, 6, 3, 6, 6 ]
        for i, Z in enumerate(expected_Z):
            assert graphite_shellstructure.Z[i] == Z, f"Expected Z={Z} for shell {i+1}, got {graphite_shellstructure.Z[i]}"

    def test_a_first_shells(self, graphite_shellstructure): 
        expected_a = [1.0, np.sqrt(3), 2.0, np.sqrt(7), 3.0 ]
        for i, a in enumerate(expected_a):
            assert np.isclose(graphite_shellstructure.a[i], a), f"Expected a={a} for shell {i+1}, got {graphite_shellstructure.a[i]}"