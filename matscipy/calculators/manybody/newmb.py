"""Manybody calculator definition."""

import numpy as np

from abc import ABC, abstractmethod
from typing import Mapping
from ...calculators.calculator import MatscipyCalculator
from ...neighbours import Neighbourhood
from ...numpy_tricks import mabincount


# Broacast slices
_c = np.s_[..., np.newaxis]
_cc = np.s_[..., np.newaxis, np.newaxis]


def ein(*args):
    """Optimized einsum."""
    return np.einsum(*args, optimize=True)


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
        def __call__(self, rsq_p, xi_p):
            """Return ɸ(rᵢⱼ², ξᵢⱼ)."""

        @abstractmethod
        def gradient(self, rsq_p, xi_p):
            """Return [∂₁ɸ(rᵢⱼ², ξᵢⱼ), ∂₂ɸ(rᵢⱼ², ξᵢⱼ)]."""

        @abstractmethod
        def hessian(self, rsq_p, xi_p):
            """Return [∂₁₁ɸ(rᵢⱼ², ξᵢⱼ), ∂₂₂ɸ(rᵢⱼ², ξᵢⱼ), ∂₁₂ɸ(rᵢⱼ², ξᵢⱼ)]."""

    class Theta:
        """Define the three-body term Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)."""

        @abstractmethod
        def __call__(self, R1_p, R2_p, R3_p):
            """Return Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)."""

        @abstractmethod
        def gradient(self, R1_p, R2_p, R3_p):
            """
            Return [∂₁Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₂Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)].
            """

        @abstractmethod
        def hessian(self, R1_p, R2_p, R3_p):
            """
            Return [∂₁₁Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₂₂Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₃₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₂₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₁₃Θ(rᵢⱼ², rᵢₖ², rⱼₖ²),
                    ∂₁₂Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)].
            """

    def __init__(self,
                 phi: Mapping[int, Phi],
                 theta: Mapping[int, Theta],
                 neighbourhood: Neighbourhood):
        """Construct with potentials ɸ(rᵢⱼ², ξᵢⱼ) and Θ(rᵢⱼ², rᵢₖ², rⱼₖ²)."""
        super().__init__()
        self.phi = phi
        self.theta = theta
        self.neighbourhood = neighbourhood

    @staticmethod
    def _assemble_triplet_to_pair(ij_t, values_t, nb_pairs):
        return mabincount(ij_t, values_t, minlength=nb_pairs)

    @staticmethod
    def _assemble_pair_to_atom(i_p, values_p, nb_atoms):
        return mabincount(i_p, values_p, minlength=nb_atoms)

    @staticmethod
    def _assemble_triplet_to_atom(i_t, values_t, nb_atoms):
        return mabincount(i_t, values_t, minlength=nb_atoms)

    def calculate(self, atoms, properties, system_changes):
        """Calculate properties on atoms."""
        super().calculate(atoms, properties, system_changes)

        # Topology information
        i_p, j_p, r_pc = self.neighbourhood.get_pairs(atoms, 'ijD')
        ij_t, ik_t, jk_t, r_tqc = self.neighbourhood.get_triplets(atoms, 'ijkD')
        n_p, n_t = len(i_p), len(i_p[ij_t])
        n = len(atoms)

        # Pair and triplet types
        t_p = self.neighbourhood.pair_type(
            *(atoms.numbers[i] for i in (i_p, j_p))
        )
        t_t = self.neighbourhood.triplet_type(
            *(atoms.numbers[i] for i in (i_p[ij_t], j_p[ij_t], j_p[ik_t]))
        )

        # Squared distances
        rsq_p = np.sum(r_pc**2, axis=-1)
        rsq_tq = np.sum(r_tqc**2, axis=-1)

        # Three-body potential data
        theta_t = np.zeros(n_t)
        dtheta_qt = np.zeros((3, n_t))

        for t in np.unique(t_t):
            m = t_t == t  # type mask
            R = rsq_tq[m].T  # distances squared

            # Computing energy and gradient
            theta_t[m] = self.theta[t](*R)
            dtheta_qt[:, m] = self.theta[t].gradient(*R)

        # Aggregating xi
        xi_p = self._assemble_triplet_to_pair(ij_t, theta_t, n_p)

        # Pair potential data
        phi_p = np.zeros(n_p)
        dphi_cp = np.zeros((2, n_p))

        for t in np.unique(t_p):
            m = t_p == t  # type mask

            phi_p[m] = self.phi[t](rsq_p[m], xi_p[m])
            dphi_cp[:, m] = self.phi[t].gradient(rsq_p[m], xi_p[m])

        # Energy
        epot = 0.5 * phi_p.sum()

        # Forces
        dpdxi = dphi_cp[1] 
        dpdxi_dtdRX_rX = dpdxi[ij_t][_cc] * ein('qt,tqc->tqc', dtheta_qt, r_tqc)  # compute dp/dxi * dΘ/dRX * rX
        dpdR_r = dphi_cp[0][_c] * r_pc  # compute dɸ/dR * r

        f_nc = self._assemble_triplet_to_atom(i_p[ij_t], dpdxi_dtdRX_rX[:, 0], n) - self._assemble_triplet_to_atom(j_p[ij_t], dpdxi_dtdRX_rX[:, 0], n) \
             + self._assemble_triplet_to_atom(i_p[ik_t], dpdxi_dtdRX_rX[:, 1], n) - self._assemble_triplet_to_atom(j_p[ik_t], dpdxi_dtdRX_rX[:, 1], n) \
             + self._assemble_triplet_to_atom(j_p[ij_t], dpdxi_dtdRX_rX[:, 2], n) - self._assemble_triplet_to_atom(j_p[ik_t], dpdxi_dtdRX_rX[:, 2], n) 

        f_nc += self._assemble_pair_to_atom(i_p, dpdR_r, n) - self._assemble_pair_to_atom(j_p, dpdR_r, n)

        # Stresses
        dtdRX_rXrX = ein('tXi,tXj->tij', dpdxi_dtdRX_rX, r_tqc)
        dpdR_rr = ein('pi,pj->pij', dpdR_r, r_pc)
        
        s_cc = np.sum(dtdRX_rXrX, axis=0)
        s_cc += np.sum(dpdR_rr, axis=0)
        s_cc *= 1 / atoms.get_volume()

        # Update results
        self.results.update(
            {
                "energy": epot,
                "free_energy": epot,
                "stress": s_cc,
                "forces": f_nc,
            }
        )
