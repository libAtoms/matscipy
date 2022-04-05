import numpy as np

from abc import ABC, abstractmethod
from ...calculators.calculator import MatscipyCalculator
from ...neighbours import (
    CutoffNeighbourhood,
    Neighbourhood,
)


class Manybody(MatscipyCalculator):
    """Generic two- and three- body interaction calculator."""

    implemented_properties = [
        'free_energy',
        'energy',
        'stress',
        'forces',
        'hessian',
        'nonaffine_forces',
        'birch_coefficients',
    ]

    class Phi(ABC):
        """Define the manybody interaction with pair term ɸ(rᵢⱼ², ξᵢⱼ)."""

        @abstractmethod
        def __call__(self, rsq_p, xi_p, atom_type_p, pair_type_p):
            """Return ɸ(rᵢⱼ², ξᵢⱼ)."""

        @abstractmethod
        def gradient(self, rsq_p, xi_p, atom_type_p, pair_type_p):
            """Return [∂₁ɸ(rᵢⱼ², ξᵢⱼ), ∂₂ɸ(rᵢⱼ², ξᵢⱼ)]."""

        @abstractmethod
        def hessian(self, rsq_p, xi_p, atom_type_p, pair_type_p):
            """Return [∂₁₁ɸ(rᵢⱼ², ξᵢⱼ), ∂₂₂ɸ(rᵢⱼ², ξᵢⱼ), ∂₁₂ɸ(rᵢⱼ², ξᵢⱼ)]."""

    class Theta:
        """Define the three-body term Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)."""

        @abstractmethod
        def __call__(self, R1_p, R2_p, R3_p, atom_type_p, pair_type_p):
            """Return Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)."""

        @abstractmethod
        def gradient(self, R1_p, R2_p, R3_p, atom_type_p, pair_type_p):
            """
            Return [∂₁Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₂Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)].
            """

        @abstractmethod
        def hessian(self, R1_p, R2_p, R3_p, atom_type_p, pair_type_p):
            """
            Return [∂₁₁Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₂₂Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₃₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₂₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₁₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₁₂Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)].
            """

    def __init__(self,
                 phi: Phi,
                 theta: Theta,
                 neighbourhood: Neighbourhood = None):
        """Construct with potentials ɸ(rᵢⱼ², ξᵢⱼ) and Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)."""
        super().__init__()
        self.phi = phi
        self.theta = theta

        self.neighbourhood = CutoffNeighbourhood

    def calculate(self, atoms, properties, system_changes):
        """Calculate properties on atoms."""
        super().calculate(atoms, properties, system_changes)

        # ... do stuff ...
        r_pc = ...
        f_pc = ...
        epot = ...
        f_nc = ...

        # Compute virial
        virial_v = self._virial(r_pc, f_pc)

        # Update results
        self.results.update(
            {
                "energy": epot,
                "free_energy": epot,
                "stress": virial_v / atoms.get_volume(),
                "forces": f_nc,
            }
        )
