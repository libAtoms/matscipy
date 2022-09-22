"""Manybody calculator definition."""

import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Mapping
from scipy.sparse import bsr_matrix
from ...calculators.calculator import MatscipyCalculator
from ...neighbours import Neighbourhood, first_neighbours, find_indices_of_reversed_pairs
from ...numpy_tricks import mabincount
from ...elasticity import full_3x3_to_Voigt_6_stress


__all__ = ["Manybody"]

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
        'dynamical_matrix',
        'born_constants',
        'nonaffine_forces',
        'birch_coefficients',
        'elastic_constants',
    ]

    _voigt_seq = [0, 5, 4, 5, 1, 3, 4, 3, 2]

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

    class _idx:
        """Helper class for index algebra."""

        def __init__(self, idx, sign=1):
            self.idx = idx
            self.sign = sign

        def __eq__(self, other):
            return self.idx == other.idx and self.sign == other.sign

        def __str__(self):
            return ("-" if self.sign < 0 else "") + self.idx

        def __repr__(self):
            return str(self)

        def __mul__(self, other):
            return type(self)(self.idx + other.idx, self.sign * other.sign)

        def __neg__(self):
            return type(self)(self.idx, -self.sign)

        def offdiagonal(self):
            for c in "ijk":
                if self.idx.count(c) > 1:
                    return False
            return True

    def __init__(self, phi: Mapping[int, Phi], theta: Mapping[int, Theta],
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

    @classmethod
    def sum_ij_pi_ij_n(cls, n, pairs, values_p):
        r"""Compute :math:`\sum_{ij}\pi_{ij|n}\Chi_{ij}`."""
        i_p, j_p = pairs
        return (
            + cls._assemble_pair_to_atom(i_p, values_p, n)
            - cls._assemble_pair_to_atom(j_p, values_p, n)
        )  # yapf: disable

    @classmethod
    def sum_ij_sum_X_pi_X_n(cls, n, pairs, triplets, values_tq):
        r"""Compute :math:`\sum_{ij}\sum_{k\neq i,j}\sum_{X}\pi_{X|n}\Chi_X`."""
        i_p, j_p = pairs
        ij_t, ik_t = triplets

        return sum(
            + cls._assemble_triplet_to_atom(i, values_tq[:, q], n)
            - cls._assemble_triplet_to_atom(j, values_tq[:, q], n)

            # Loop of pairs in the ijk triplet
            for q, (i, j) in enumerate([(i_p[ij_t], j_p[ij_t]),   # ij pair
                                        (i_p[ik_t], j_p[ik_t]),   # ik pair
                                        (j_p[ij_t], j_p[ik_t])])  # jk pair

        )  # yapf: disable

    @classmethod
    def sum_ijk_tau_XY_mn(cls, n, triplets, tr_p, X, Y, values_t):
        triplets = {
            k: v for k, v in zip(["ij", "ik", "jk", "ji", "ki", "kj"],
                                 list(triplets) + [tr_p[t] for t in triplets])
        }

        # All indices in τ_XY|mn
        indices = X[np.newaxis] * Y[np.newaxis].T

        # Avoid double counting symmetric indices
        if np.all(X == Y):
            indices = indices[~np.tri(2, 2, -1, dtype=bool)]

        return sum(
            idx.sign
            * cls._assemble_triplet_to_pair(triplets[idx.idx], values_t, n)
            # Indices relevant for off-diagonal terms
            for idx in np.ravel(indices) if idx.offdiagonal()
        )

    @classmethod
    def _X_indices(cls):
        i, j, k = map(cls._idx, 'ijk')
        return np.array([[i, -j],
                         [i, -k],
                         [j, -k]])

    @classmethod
    def sum_XY_sum_ijk_tau_XY_mn(cls, n, triplets, tr_p, values_tXY):
        X_indices = cls._X_indices()

        return sum(
            cls.sum_ijk_tau_XY_mn(n, triplets, tr_p, X, Y, values_tXY[:, x, y])
            for (x, X), (y, Y) in combinations_with_replacement(
                    enumerate(X_indices), r=2
            )
        )

    @classmethod
    def sum_XX_sum_ijk_tau_XX_mn(cls, n, triplets, tr_p, values_tX):
        X_indices = cls._X_indices()

        return sum(
            cls.sum_ijk_tau_XY_mn(n, triplets, tr_p, X, X, values_tX[:, x])
            for x, X in enumerate(X_indices)
        )

    @classmethod
    def sum_X_sum_ijk_tau_ijX_mn(cls, n, triplets, tr_p, values_tX):
        X_indices = cls._X_indices()

        return sum(
            cls.sum_ijk_tau_XY_mn(n, triplets, tr_p,
                                  X_indices[0], X, values_tX[:, x])
            for x, X in enumerate(X_indices)
        )

    @classmethod
    def sum_X_sum_ijk_tau_ij_XOR_X_mn(cls, n, triplets, tr_p, values_tX):
        X_indices = cls._X_indices()

        return sum(
            cls.sum_ijk_tau_XY_mn(n, triplets, tr_p,
                                  X_indices[0], X, values_tX[:, x + 1])
            for x, X in enumerate(X_indices[1:])
        )

    def _masked_compute(self, atoms, order, list_ij=None, list_ijk=None):
        """Compute requested derivatives of phi and theta."""
        if not isinstance(order, list):
            order = [order]

        if list_ijk is None and list_ij is None:
            i_p, j_p, r_pc = self.neighbourhood.get_pairs(atoms, 'ijD')
            ij_t, ik_t, r_tqc = self.neighbourhood.get_triplets(atoms, 'ijD')
        else:
            i_p, j_p, r_pc = list_ij
            ij_t, ik_t, r_tqc = list_ijk

        # Pair and triplet types
        t_p = self.neighbourhood.pair_type(*(atoms.numbers[i]
                                             for i in (i_p, j_p)))
        t_t = self.neighbourhood.triplet_type(*(atoms.numbers[i]
                                                for i in (i_p[ij_t], j_p[ij_t],
                                                          j_p[ik_t])))

        derivatives = np.array([
            ('__call__', 1, 1),
            ('gradient', 2, 3),
            ('hessian', 3, 6),
        ], dtype=object)

        phi_res = {
            d[0]: np.zeros([d[1], len(r_pc)])
            for d in derivatives[order]
        }

        theta_res = {
            d[0]: np.zeros([d[2], len(r_tqc)])
            for d in derivatives[order]
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
        xi_p = self._assemble_triplet_to_pair(ij_t, theta_t.squeeze(),
                                              len(r_pc))

        for t in np.unique(t_p):
            m = t_p == t  # type mask

            # Required derivative order
            for attr, res in phi_res.items():
                res[:, m] = getattr(self.phi[t], attr)(rsq_p[m], xi_p[m])

        return phi_res.values(), theta_res.values()

    def calculate(self, atoms, properties, system_changes):
        """Calculate properties on atoms."""
        super().calculate(atoms, properties, system_changes)

        # Topology information
        i_p, j_p, r_pc = self.neighbourhood.get_pairs(atoms, 'ijD')
        ij_t, ik_t, r_tqc = self.neighbourhood.get_triplets(atoms, 'ijD')
        n = len(atoms)

        # Request energy and gradient
        (phi_p, dphi_cp), (theta_t, dtheta_qt) = \
            self._masked_compute(atoms, order=[0, 1],
                                 list_ij=[i_p, j_p, r_pc],
                                 list_ijk=[ij_t, ik_t, r_tqc])

        # Energy
        epot = 0.5 * phi_p.sum()

        # Forces
        dpdxi = dphi_cp[1]

        # compute dɸ/dxi * dΘ/dRX * rX
        dpdxi_dtdRX_rX = ein('t,qt,tqc->tqc', dpdxi[ij_t], dtheta_qt, r_tqc)
        dpdR_r = dphi_cp[0][_c] * r_pc  # compute dɸ/dR * r

        # Assembling triplet force contribution for each pair in triplet
        f_nc = self.sum_ij_sum_X_pi_X_n(n, (i_p, j_p), (ij_t, ik_t),
                                        dpdxi_dtdRX_rX)

        # Assembling the pair force contributions
        f_nc += self.sum_ij_pi_ij_n(n, (i_p, j_p), dpdR_r)

        # Stresses
        s_cc = ein('tXi,tXj->ij', dpdxi_dtdRX_rX,
                   r_tqc)  # outer + sum triplets
        s_cc += ein('pi,pj->ij', dpdR_r, r_pc)  # outer + sum pairs
        s_cc *= 1 / atoms.get_volume()

        # Update results
        self.results.update({
            "energy": epot,
            "free_energy": epot,
            "stress": full_3x3_to_Voigt_6_stress(s_cc),
            "forces": f_nc,
        })

    def get_born_elastic_constants(self, atoms):
        """Compute the Born (affine) elastic constants."""
        if self.atoms is None:
            self.atoms = atoms

        # Topology information
        r_pc = self.neighbourhood.get_pairs(atoms, 'D')
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

        C_cccc += ein('t,qt,tqa,tqb,tqm,tqn->abmn', dpdxi, ddtheta_qt[XY],
                      r_tqc[:, X], r_tqc[:, X], r_tqc[:, Y], r_tqc[:, Y])

        # Term 4
        ddpdRdxi = ddphi_cp[2][ij_t]

        # Combination indices involved in term 4
        # also implicitely symmetrizes ?
        X = [0, 0, 0, 1, 0, 2]
        Y = [0, 0, 1, 0, 2, 0]
        XY = [0, 0, 1, 1, 2, 2]

        C_cccc += ein('t,qt,tqa,tqb,tqm,tqn->abmn', ddpdRdxi, dtheta_qt[XY],
                      r_tqc[:, X], r_tqc[:, X], r_tqc[:, Y], r_tqc[:, Y])

        # Term 5
        ddpddxi = ddphi_cp[1]
        dtdRx_rXrX = self._assemble_triplet_to_pair(
            ij_t,
            ein('qt,tqa,tqb->tab', dtheta_qt, r_tqc, r_tqc),
            len(r_pc),
        )

        C_cccc += ein('p,pab,pmn->abmn', ddpddxi, dtdRx_rXrX, dtdRx_rXrX)

        return 2 * C_cccc / atoms.get_volume()

    def get_nonaffine_forces(self, atoms):
        """Compute non-affine forces."""
        n = len(atoms)
        i_p, j_p, r_pc = self.neighbourhood.get_pairs(atoms, 'ijD')
        ij_t, ik_t, r_tqc = self.neighbourhood.get_triplets(atoms, 'ijD')

        (dphi_cp, ddphi_cp), (dtheta_qt, ddtheta_qt) = \
            self._masked_compute(atoms, order=[1, 2])

        # Term 1 and 2 have the same structure, we assemble @ same time
        e = np.eye(3)  # I have no idea what e is
        dpdR, ddpddR = dphi_cp[0], ddphi_cp[0]
        term_12_pcab = (
            ein('p,pa,bg->pgab', dpdR, r_pc, e)  # term 1
            + ein('p,pb,ag->pgab', dpdR, r_pc, e)  # term 1
            + 2 * ein('p,pa,pb,pg->pgab', ddpddR, r_pc, r_pc, r_pc)  # term 2
        )

        # Assemble pair terms
        naf_ncab = self.sum_ij_pi_ij_n(n, (i_p, j_p), term_12_pcab)

        # Term 3
        # Here we sum over Y in the inner loop, over X in the assembly
        # because there is a pi_{X|n} in the sum
        # terms 3 and 5 actually have the same structure
        # maybe group up?
        dpdxi = dphi_cp[1][ij_t]
        # turn voigt dtdRxdRy to 3x3
        ddtdRXdRY = ddtheta_qt[self._voigt_seq].reshape(3, 3, -1)

        term_3_tXcab = 2 * ein('XYt,tYa,tYb,tXc->tXcab', ddtdRXdRY, r_tqc,
                               r_tqc, r_tqc)
        term_3_tXcab += (
            ein('Xt,tXb,ag->tXgab', dtheta_qt, r_tqc, e)
            + ein('Xt,tXa,bg->tXgab', dtheta_qt, r_tqc, e))

        term_3_tXcab *= dpdxi[_cccc]

        naf_ncab += self.sum_ij_sum_X_pi_X_n(n, (i_p, j_p), (ij_t, ik_t),
                                             term_3_tXcab)

        # Term 4
        # Here we have to sub-terms:
        #  - one sums over X in the inner loop and has pi_{ij|n}
        #    => sub-term 1 (defined on pairs)
        #  - one has pi_{X|n}
        #    => sub-term 2 (define on triplets)
        ddpdRdxi = ddphi_cp[2][ij_t]
        dtdRX = dtheta_qt
        term_4_1_pab = self._assemble_triplet_to_pair(
            ij_t,
            ein(
                't,Xt,tXa,tXb,tc->tcab',  # note: sum over X
                ddpdRdxi,
                dtdRX,
                r_tqc,
                r_tqc,
                r_tqc[:, 0]),
            len(i_p),
        )

        term_4_2_tXcab = ein('t,Xt,ta,tb,tXc->tXcab', ddpdRdxi, dtdRX,
                             r_tqc[:, 0], r_tqc[:, 0], r_tqc)

        # assembling sub-terms
        naf_ncab += 2 * self.sum_ij_pi_ij_n(n, (i_p, j_p), term_4_1_pab)
        naf_ncab += 2 * self.sum_ij_sum_X_pi_X_n(n, (i_p, j_p),
                                                 (ij_t, ik_t), term_4_2_tXcab)

        # Term 5
        ddpddxi = ddphi_cp[1][ij_t]
        dtdRY = dtdRX  # just for clarity
        term_5_1_pab = self._assemble_triplet_to_pair(
            ij_t, ein('qt,tqa,tqb->tab', dtdRX, r_tqc, r_tqc), len(i_p))

        term_5_2_tYgab = ein('t,Yt,tab,tYg->tYgab', ddpddxi, dtdRY,
                             term_5_1_pab[ij_t], r_tqc)
        naf_ncab += 2 * self.sum_ij_sum_X_pi_X_n(n, (i_p, j_p),
                                                 (ij_t, ik_t), term_5_2_tYgab)

        return naf_ncab

    def get_hessian(self, atoms, format='sparse', divide_by_masses=False):
        """Compute hessian."""
        cutoff = self.neighbourhood.cutoff

        # We nned twice the cutoff to get jk
        i_p, j_p, r_p, r_pc = self.neighbourhood.get_pairs(
            atoms, 'ijdD', cutoff=2*cutoff
        )

        # TODO: make sure this works with different atom types
        mask = r_p > cutoff

        tr_p = find_indices_of_reversed_pairs(i_p, j_p, r_p)

        ij_t, ik_t, jk_t, r_tq, r_tqc = self.neighbourhood.get_triplets(
            atoms, 'ijkdD', neighbours=[i_p, j_p, r_p, r_pc]
        )

        n = len(atoms)
        nb_pairs = len(i_p)
        nb_triplets = len(ij_t)

        # Otherwise we get a segmentation fault because ij_t is empty
        if nb_triplets == 0:
            raise RuntimeError("No triplet in hessian computation!")

        first_n = first_neighbours(n, i_p)
        first_p = first_neighbours(nb_pairs, ij_t)

        (dphi_cp, ddphi_cp), (dtheta_qt, ddtheta_qt) = \
            self._masked_compute(atoms, order=[1, 2],
                                 list_ij=[i_p, j_p, r_pc],
                                 list_ijk=[ij_t, ik_t, r_tqc])

        # Masking extraneous pair contributions
        dphi_cp[:, mask] = 0
        ddphi_cp[:, mask] = 0

        # Term 1, merge with T2 in the end
        e = np.identity(3)
        dpdR = dphi_cp[0]
        H_pcc = ein('p,ab->pab', dpdR, -e)

        # Term 2, merge with T1 in the end
        ddpddR = ddphi_cp[0]
        H_pcc -= ein('p,pa,pb->pab', 2 * ddpddR, r_pc, r_pc)

        # Term 3
        dpdxi = dphi_cp[1]
        dpdxi = dpdxi[ij_t]
        dtdRX = dtheta_qt
        ddtdRXdRY = ddtheta_qt[self._voigt_seq].reshape(3, 3, -1)

        dp_dt_e = ein('t,Xt,ab->tXab', dpdxi, dtdRX, e)
        dp_ddt_rX_rY = ein('t,XYt,tXa,tYb->tXYab', 2 * dpdxi, ddtdRXdRY,
                           r_tqc, r_tqc)

        H_pcc += self.sum_XY_sum_ijk_tau_XY_mn(nb_pairs, (ij_t, ik_t, jk_t),
                                               tr_p, dp_ddt_rX_rY)
        H_pcc += self.sum_XX_sum_ijk_tau_XX_mn(nb_pairs, (ij_t, ik_t, jk_t),
                                               tr_p, dp_dt_e)

        # Term 4
        ddpdRdxi = ddphi_cp[2]
        ddpdRdxi = ddpdRdxi[ij_t]
        dtdRX = dtheta_qt

        ddp_dt_rij_rX = ein('t,Xt,ta,tXb->tXab', 2 * ddpdRdxi, dtdRX,
                            r_tqc[:, 0], r_tqc)

        H_pcc += self.sum_X_sum_ijk_tau_ijX_mn(nb_pairs, (ij_t, ik_t, jk_t),
                                               tr_p, ddp_dt_rij_rX)

        H_pcc -= self._assemble_triplet_to_pair(tr_p[ij_t], ddp_dt_rij_rX[:, 0], nb_pairs)

        # Term 5
        ddpddxi = ddphi_cp[1]
        ddpddxi = ddpddxi[ij_t]
        dtdRX = dtheta_qt

        # Pair
        H_pcc += ein('p,p,p,pa,pb->pab', -2 * ddphi_cp[1],
                                        self._assemble_triplet_to_pair(ij_t, dtdRX[0], nb_pairs),
                                        self._assemble_triplet_to_pair(ij_t, dtdRX[0], nb_pairs),
                                        r_pc, r_pc)

        # Triplet
        dtdRx_rx = ein('Xt,tXa->tXa', dtdRX, r_tqc)
        ddp_dtdRx_rx_dtdRy_ry = ein('t,tXa,tYb->tXYab', 2 * ddpddxi, self._assemble_triplet_to_pair(ij_t, dtdRx_rx, nb_pairs)[ij_t], dtdRx_rx)

        H_pcc += self.sum_X_sum_ijk_tau_ij_XOR_X_mn(nb_pairs, (ij_t, ik_t, jk_t),
                                               tr_p, ddp_dtdRx_rx_dtdRy_ry[:, 0])

        # Quadruplets
        H_pcc -= self._assemble_triplet_to_pair(ik_t, ddp_dtdRx_rx_dtdRy_ry[:, 1, 1], nb_pairs)
        H_pcc -= self._assemble_triplet_to_pair(jk_t, ddp_dtdRx_rx_dtdRy_ry[:, 2, 2], nb_pairs)
        H_pcc -= self._assemble_triplet_to_pair(ik_t, ddp_dtdRx_rx_dtdRy_ry[:, 1, 2], nb_pairs)
        H_pcc -= self._assemble_triplet_to_pair(tr_p[jk_t], ddp_dtdRx_rx_dtdRy_ry[:, 2, 1], nb_pairs)

        H_pcc += ein('p,pa,pb->pab', 2 * ddphi_cp[1], self._assemble_triplet_to_pair(ij_t, dtdRx_rx[:, 1], nb_pairs),
                                                      self._assemble_triplet_to_pair(ij_t, dtdRx_rx[:, 2], nb_pairs))

        # Deal with ij_im / ij_in expression
        for im_in in range(nb_triplets):
            pair_im = ij_t[im_in]
            pair_in = ik_t[im_in]
            pair_mn = jk_t[im_in]

            for t in range(first_p[pair_im], first_p[pair_im + 1]):

                pair_ij = ik_t[t]

                if pair_ij != pair_im and pair_ij != pair_in:
                    rim_c = r_pc[pair_im]
                    rin_c = r_pc[pair_in]
                    rsq_im = np.sum(r_pc[pair_im]**2)
                    rsq_in = np.sum(r_pc[pair_in]**2)
                    rsq_ij = np.sum(r_pc[pair_ij]**2)

                    # Distances jm and jn
                    rjn_c = r_pc[pair_in] - r_pc[pair_ij]
                    rjm_c = r_pc[pair_im] - r_pc[pair_ij]
                    rsq_jm = np.sum(rjm_c**2)
                    rsq_jn = np.sum(rjn_c**2)

                    # TODO: Assumes monoatomic system at the moment
                    H_pcc[pair_mn] += ddphi_cp[1][pair_ij] * np.outer(self.theta[1].gradient(rsq_ij, rsq_im, rsq_jm)[1] * rim_c,
                                                                      self.theta[1].gradient(rsq_ij, rsq_in, rsq_jn)[1] * rin_c)

                    H_pcc[pair_mn] += ddphi_cp[1][pair_ij] * np.outer(self.theta[1].gradient(rsq_ij, rsq_im, rsq_jm)[2] * rjm_c,
                                                                      self.theta[1].gradient(rsq_ij, rsq_in, rsq_jn)[2] * rjn_c)

                    H_pcc[pair_mn] += 2 * ddphi_cp[1][pair_ij] * np.outer(self.theta[1].gradient(rsq_ij, rsq_im, rsq_jm)[1] * rim_c,
                                                                          self.theta[1].gradient(rsq_ij, rsq_in, rsq_jn)[2] * rjn_c)
        # Symmetrization with H_nm
        H_pcc += H_pcc.transpose(0, 2, 1)[tr_p]

        # Compute the diagonal elements by bincount the off-diagonal elements
        H_acc = -self._assemble_pair_to_atom(i_p, H_pcc, n)

        if divide_by_masses:
            mass_p = atoms.get_masses()
            H_pcc /= np.sqrt(mass_p[i_p] * mass_p[j_p])[_cc]
            H_acc /= mass_p[_cc]

        H = (
            bsr_matrix((H_pcc, j_p, first_n), shape=(3 * n, 3 * n))
            + bsr_matrix((H_acc, np.arange(n), np.arange(n + 1)),
                         shape=(3 * n, 3 * n))
        )

        return H
