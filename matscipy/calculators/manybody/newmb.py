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

    def _masked_compute(self, atoms, order):
        """Compute requested derivatives of phi and theta."""
        if not isinstance(order, list):
            order = [order]

        i_p, j_p, r_pc = self.neighbourhood.get_pairs(atoms, 'ijD')
        ij_t, ik_t, jk_t, r_tqc = self.neighbourhood.get_triplets(atoms, 'ijkD')

        # Pair and triplet types
        t_p = self.neighbourhood.pair_type(
            *(atoms.numbers[i] for i in (i_p, j_p))
        )
        t_t = self.neighbourhood.triplet_type(
            *(atoms.numbers[i] for i in (i_p[ij_t], j_p[ij_t], j_p[ik_t]))
        )

        derivatives = np.array([
            ('__call__', 1, 1),
            ('gradient', 2, 3),
            ('hessian', 3, 6),
        ], dtype=object)

        phi_res = {
            d[0]: np.zeros([d[1], len(r_pc)]) for d in derivatives[order]
        }

        theta_res = {
            d[0]: np.zeros([d[2], len(r_tqc)]) for d in derivatives[order]
        }

        # Do not allocate array for theta_t if energy is explicitely requested
        if '__call__' in theta_res:
            theta_t = theta_res['__call__']
            extra_compute_theta = False
        else:
            theta_t = np.zeros([1, len(r_tqc)])
            extra_compute_theta = True

        # Squared distances
        rsq_p = np.sum(r_pc**2, axis=-1)
        rsq_tq = np.sum(r_tqc**2, axis=-1)

        for t in np.unique(t_t):
            m = t_t == t  # type mask
            R = rsq_tq[m].T  # distances squared

            # Required derivative order
            for attr, res in theta_res.items():
                res[:, m] = getattr(self.theta[t], attr)(*R)

            # We need value of theta to compute xi
            if extra_compute_theta:
                theta_t[:, m] = self.theta[t](*R)

        # Aggregating xi
        xi_p = self._assemble_triplet_to_pair(ij_t, theta_t.squeeze(), len(r_pc))

        for t in np.unique(t_p):
            m = t_p == t  # type mask

            # Required derivative order
            for attr, res in phi_res.items():
                res[:, m] = getattr(self.phi[t], attr)(
                    rsq_p[m], xi_p[m]
                )

        return phi_res.values(), theta_res.values()

    def calculate(self, atoms, properties, system_changes):
        """Calculate properties on atoms."""
        super().calculate(atoms, properties, system_changes)

        # Topology information
        i_p, j_p, r_pc = self.neighbourhood.get_pairs(atoms, 'ijD')
        ij_t, ik_t, jk_t, r_tqc = self.neighbourhood.get_triplets(atoms, 'ijkD')
        n = len(atoms)

        # Request energy and gradient
        (phi_p, dphi_cp), (theta_t, dtheta_qt) = \
            self._masked_compute(atoms, order=[0, 1])

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
        """Compute the Born (affine) elastic constants."""
        if self.atoms is None:
            self.atoms = atoms

        # Topology information
        r_pc, = self.neighbourhood.get_pairs(atoms, 'D')
        ij_t, r_tqc = self.neighbourhood.get_triplets(atoms, 'iD')

        (dphi_cp, ddphi_cp), (dtheta_qt, ddtheta_qt) = \
            self._masked_compute(atoms, order=[1, 2])

        # Term 1 vanishes
        C_cccc = np.zeros([3] * 4)

        # Term 2
        ddpddR = ddphi_cp[0]
        C_cccc += ein('p,pa,pb,pm,pn->abmn', ddpddR, r_pc, r_pc, r_pc, r_pc)

        # Term 3
        dpdxi = dphi_cp[1][ij_t]

        # Combination indices involved in term 3
        X = [0, 1, 2, 2, 1, 0, 2, 0, 1]
        Y = [0, 1, 2, 1, 2, 2, 0, 1, 0]
        XY = [0, 1, 2, 3, 3, 4, 4, 5, 5]

        C_cccc += ein('t,qt,tqa,tqb,tqm,tqn->abmn',
                      dpdxi,
                      ddtheta_qt[XY],
                      r_tqc[:, X], r_tqc[:, X],
                      r_tqc[:, Y], r_tqc[:, Y])

        # Term 4
        ddpdRdxi = ddphi_cp[2][ij_t]

        # Combination indices involved in term 4
        # also implicitely symmetrizes ?
        X = [0, 0, 0, 1, 0, 2]
        Y = [0, 0, 1, 0, 2, 0]
        XY = [0, 0, 1, 1, 2, 2]

        C_cccc += ein('t,qt,tqa,tqb,tqm,tqn->abmn',
                      ddpdRdxi,
                      dtheta_qt[XY],
                      r_tqc[:, X], r_tqc[:, X],
                      r_tqc[:, Y], r_tqc[:, Y])

        # Term 5
        ddpddxi = ddphi_cp[1]
        dtdRx_rXrX = self._assemble_triplet_to_pair(
            ij_t, ein('qt,tqa,tqb->tab', dtheta_qt, r_tqc, r_tqc), len(r_pc),
        )

        C_cccc += ein('p,pab,pmn->abmn', ddpddxi, dtdRx_rXrX, dtdRx_rXrX)

        return 2 * C_cccc / atoms.get_volume()

    def get_nonaffine_forces(self, atoms):
        """Compute non-affine forces."""


    def get_hessian(self, atoms):
        """
        Hessian
        """
        pass
