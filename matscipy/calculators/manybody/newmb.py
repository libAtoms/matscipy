"""Manybody calculator definition."""

import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Mapping
from ...calculators.calculator import MatscipyCalculator
from ...neighbours import Neighbourhood
from ...numpy_tricks import mabincount


# Broacast slices
_c = np.s_[..., np.newaxis]
_cc = np.s_[..., np.newaxis, np.newaxis]
_ccc = np.s_[..., np.newaxis, np.newaxis, np.newaxis]
_cccc = np.s_[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]

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
        'born_constants',        
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

        if isinstance(phi, defaultdict):
            self.phi = phi
        else:
            from .potentials import ZeroPair  # noqa
            self.phi = defaultdict(lambda: ZeroPair())
            self.phi.update(phi)

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

        # compute dɸ/dxi * dΘ/dRX * rX
        dpdxi_dtdRX_rX = ein('t,qt,tqc->tqc', dpdxi[ij_t], dtheta_qt, r_tqc)
        dpdR_r = dphi_cp[0][_c] * r_pc  # compute dɸ/dR * r

        # Assembling triplet force contribution for each pair in triplet
        f_nc = sum(
            self._assemble_triplet_to_atom(i, dpdxi_dtdRX_rX[:, a], n)
            - self._assemble_triplet_to_atom(j, dpdxi_dtdRX_rX[:, a], n)

            # Loop of pairs in the ijk triplet
            for a, (i, j) in enumerate([(i_p[ij_t], j_p[ij_t]),   # ij pair
                                        (i_p[ik_t], j_p[ik_t]),   # ik pair
                                        (j_p[ij_t], j_p[ik_t])])  # jk pair
        )

        # Assembling the pair force contributions
        f_nc += self._assemble_pair_to_atom(i_p, dpdR_r, n) \
            - self._assemble_pair_to_atom(j_p, dpdR_r, n)

        # Stresses
        s_cc = ein('tXi,tXj->ij', dpdxi_dtdRX_rX, r_tqc)  # outer + sum triplets
        s_cc += ein('pi,pj->ij', dpdR_r, r_pc)  # outer + sum pairs
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

    def get_born_elastic_constants(self, atoms):
        """
        Compute the Born (affine) elastic constants.
        """
        if self.atoms is None:
            self.atoms = atoms

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
        ddtheta_qt = np.zeros((6, n_t))

        for t in np.unique(t_t):
            m = t_t == t  # type mask
            R = rsq_tq[m].T  # distances squared

            # Computing energy and gradient
            theta_t[m] = self.theta[t](*R)
            dtheta_qt[:, m] = self.theta[t].gradient(*R)
            ddtheta_qt[:, m] = self.theta[t].hessian(*R)

        # Aggregating xi
        xi_p = self._assemble_triplet_to_pair(ij_t, theta_t, n_p)

        # Pair potential data
        phi_p = np.zeros(n_p)
        dphi_cp = np.zeros((2, n_p))
        ddphi_cp = np.zeros((3, n_p))

        for t in np.unique(t_p):
            m = t_p == t  # type mask

            phi_p[m] = self.phi[t](rsq_p[m], xi_p[m])
            dphi_cp[:, m] = self.phi[t].gradient(rsq_p[m], xi_p[m])
            ddphi_cp[:, m] = self.phi[t].hessian(rsq_p[m], xi_p[m])

        C_cccc = np.zeros((3,3,3,3))

        # Term 1 vanishes 

        # Term 2 
        ddpddR = ddphi_cp[0]
        C_cccc += (ddpddR[_cccc] * ein('pa,pb,pm,pn->pabmn', r_pc, r_pc, r_pc, r_pc)).sum(axis=0)

        # Term 3
        dpdxi = dphi_cp[1][ij_t]
        C_cccc += (
            dpdxi[_cccc] *
            (
              ddtheta_qt[0][_cccc] * ein('ta,tb,tm,tn->tabmn', r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 0]) \
            + ddtheta_qt[1][_cccc] * ein('ta,tb,tm,tn->tabmn', r_tqc[:, 1], r_tqc[:, 1], r_tqc[:, 1], r_tqc[:, 1]) \
            + ddtheta_qt[2][_cccc] * ein('ta,tb,tm,tn->tabmn', r_tqc[:, 2], r_tqc[:, 2], r_tqc[:, 2], r_tqc[:, 2]) \
            + ddtheta_qt[3][_cccc] * (ein('ta,tb,tm,tn->tabmn', r_tqc[:, 2], r_tqc[:, 2], r_tqc[:, 1], r_tqc[:, 1]) \
                                    + ein('ta,tb,tm,tn->tabmn', r_tqc[:, 1], r_tqc[:, 1], r_tqc[:, 2], r_tqc[:, 2])) \
            + ddtheta_qt[4][_cccc] * (ein('ta,tb,tm,tn->tabmn', r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 2], r_tqc[:, 2]) \
                                    + ein('ta,tb,tm,tn->tabmn', r_tqc[:, 2], r_tqc[:, 2], r_tqc[:, 1], r_tqc[:, 1])) \
            + ddtheta_qt[5][_cccc] * (ein('ta,tb,tm,tn->tabmn', r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 1], r_tqc[:, 1]) \
                                    + ein('ta,tb,tm,tn->tabmn', r_tqc[:, 1], r_tqc[:, 1], r_tqc[:, 0], r_tqc[:, 0])) 
            )
            ).sum(axis=0)

        # Term 4
        ddpdRdxi = ddphi_cp[2][ij_t]
        C_cccc += (ddpdRdxi[_cccc] *
                      (dtheta_qt[0][_cccc] * (ein('ta,tb,tm,tn->tabmn', r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 0]) \
                                         + ein('ta,tb,tm,tn->tabmn', r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 0])) \
                     + dtheta_qt[1][_cccc] * (ein('ta,tb,tm,tn->tabmn', r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 1], r_tqc[:, 1]) \
                                         + ein('ta,tb,tm,tn->tabmn', r_tqc[:, 1], r_tqc[:, 1], r_tqc[:, 0], r_tqc[:, 0]) ) \
                     + dtheta_qt[2][_cccc] * (ein('ta,tb,tm,tn->tabmn', r_tqc[:, 0], r_tqc[:, 0], r_tqc[:, 2], r_tqc[:, 2]) \
                                         + ein('ta,tb,tm,tn->tabmn', r_tqc[:, 2], r_tqc[:, 2], r_tqc[:, 0], r_tqc[:, 0]) ) \
                      )
                ).sum(axis=0)

        # Term 5
        ddpddxi = ddphi_cp[1]
        # Replace later!
        dtdRx_rXrX = self._assemble_triplet_to_pair(ij_t,
                         dtheta_qt[0][_cc] * ein('ta,tb->tab', r_tqc[:, 0], r_tqc[:, 0]) \
                       + dtheta_qt[1][_cc] * ein('ta,tb->tab', r_tqc[:, 1], r_tqc[:, 1]) \
                       + dtheta_qt[2][_cc] * ein('ta,tb->tab', r_tqc[:, 2], r_tqc[:, 2]), n_p)
        C_cccc += (ddpddxi[_cccc] * ein('pab,pmn->pabmn', dtdRx_rXrX, dtdRx_rXrX)).sum(axis=0)

        return 2 * C_cccc / atoms.get_volume()


