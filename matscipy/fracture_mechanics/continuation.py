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
"""Force-only pseudo-arclength tracing of clamped crack clusters (lattice-trapping S-curves).

A clamped cluster (`clusters.clamped_cluster`) in a `SinclairCrack` with `variable_alpha=False,
energy="total"` is the problem: free atoms move, clamped atoms sit on the K-field. The tracer works
in x = (u, k) with k = K/K_G and u the free-atom displacements relative to the field; the arclength
metric is ds^2 = |du|^2 + alpha^2 dk^2. Correctors use a surrogate Jacobian (the Hessian of any
matscipy many-body potential, e.g. Stillinger-Weber for Si, plus X-H springs, scaled by one
calibrated factor) with good-Broyden updates from every force call, then Newton-Krylov. For large
clusters the surrogate can stay sparse, with the Broyden updates as a low-rank correction
(`LowRankJacobian`, `TraceSettings(surrogate="sparse")`).
"""
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from ase import Atoms
from scipy.sparse.linalg import LinearOperator, gmres, splu


class ClampedProblem:
    """x = (u flat over free atoms, k = K/K_G) on a clamped SinclairCrack. Free atoms: region 1
    (groups != 0); clamped atoms ride the K-field. `U` is the field per unit k (all atoms).

    `atoms`: the built cluster (`clamped_cluster`/`cluster_like`), whose geometry is the start; the
    crack's own atoms at u = 0 put every H cap on its host (a cap's reference is its host's), so
    pass it whenever the cluster is passivated. `k_build` is the load the crack was set up at."""

    ADVANCE_BREAK_RATIO = 1.55  # crack-line bond counted broken at 1.55 bulk bonds unless set

    def __init__(self, crack, K_G, atoms=None):
        if crack.variable_alpha or not crack.variable_k:
            raise ValueError("ClampedProblem needs a clamped SinclairCrack: variable_alpha=False, "
                             "variable_k=True")
        if crack.kII != 0:
            raise ValueError("ClampedProblem traces mode I only: kII must be 0")
        n_caps = int((crack.cryst.arrays.get("h_host", np.full(len(crack.cryst), -1)) >= 0).sum())
        if atoms is not None and len(atoms) != len(crack.atoms):
            raise ValueError(f"atoms has {len(atoms)} atoms, the crack {len(crack.atoms)}")
        if n_caps and atoms is None:
            raise ValueError("a passivated cluster needs atoms= (the built cluster): at u = 0 the "
                             "crack puts every H cap on its host")
        self.crack, self.kg = crack, float(K_G)
        self.k_build = float(crack.kI) / self.kg
        self.atoms = crack.atoms.copy()
        if atoms is not None:
            self.atoms.positions = atoms.positions
        self.atoms.set_constraint()
        self.atoms.calc = None
        self.free = np.flatnonzero(crack.regionI)
        self.clamp = np.flatnonzero(~crack.regionI)
        self.nf = 3 * len(self.free)
        self.base = np.asarray(crack.cryst.positions, float)
        self.U = crack.u_cle(kI=self.kg, kII=0.0)
        self.tip_bond = (int(crack.cryst.info["bond1"]), int(crack.cryst.info["bond2"]))
        self.ncalls = 0
        self._cache = None

    def positions(self, x):
        pos = self.base + x[-1] * self.U
        pos[self.free] += np.reshape(x[:-1], (-1, 3))
        return pos

    def x_from(self, positions, k):
        u = positions[self.free] - (self.base[self.free] + k * self.U[self.free])
        return np.concatenate([u.ravel(), [float(k)]])

    def atoms_at(self, x):
        a = self.atoms.copy()
        a.positions = self.positions(x)
        return a

    def evaluate(self, x):
        """(energy, forces on all atoms) at x; one force call unless x repeats the last one."""
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1], self._cache[2]
        self.crack.set_dofs(np.concatenate([x[:-1], [x[-1] * self.kg]]))
        E = float(self.crack.get_potential_energy())
        F = np.array(self.crack.atoms.get_forces())
        self.ncalls += 1
        self._cache = (np.array(x, float), E, F)
        return E, F

    def residual(self, x):
        return self.evaluate(x)[1][self.free].ravel().copy()

    def bond_between(self, x, i, j):
        p = self.positions(x)
        d = p[i] - p[j]
        cz = self.atoms.cell[2, 2]
        d[2] -= cz * np.round(d[2] / cz)
        return float(np.linalg.norm(d))

    def dr_max(self, x1, x0):
        d = self.positions(x1)[self.free] - self.positions(x0)[self.free]
        return float(np.linalg.norm(d, axis=1).max())

    @property
    def advance_break(self):
        """Bond length (A) at which a crack-line bond counts as broken (settable)."""
        if getattr(self, "_advance_break", None) is None:
            self._advance_break = self.ADVANCE_BREAK_RATIO * np.sqrt(3.0) / 4.0 * \
                self.crack.cryst.info["a0"]
        return self._advance_break

    @advance_break.setter
    def advance_break(self, v):
        self._advance_break = float(v)

    @property
    def line_bonds(self):
        """Heavy-atom bonds crossing the cleavage plane at or ahead of the tip, sorted along the
        propagation direction: (i, j, reference length)."""
        if getattr(self, "_line_bonds", None) is None:
            ref = self.base
            tip = np.array([self.crack.cryst.info["tip_x"], self.crack.cryst.info["tip_y"]])
            cz = self.atoms.cell[2, 2]
            dmax = 1.1 * np.sqrt(3.0) / 4.0 * self.crack.cryst.info["a0"]
            heavy = np.flatnonzero(self.atoms.numbers != 1)
            out = []
            for n, a in enumerate(heavy):
                for b in heavy[n + 1:]:
                    v = ref[b] - ref[a]
                    v[2] -= cz * np.round(v[2] / cz)
                    d = float(np.linalg.norm(v))
                    xm = ref[a, 0] + 0.5 * v[0] - tip[0]
                    if d < dmax and (ref[a, 1] - tip[1]) * (ref[b, 1] - tip[1]) < 0 and xm > -1.0:
                        out.append((xm, int(a), int(b), d))
            self._line_bonds = [(a, b, d) for _, a, b, d in sorted(out)]
        return self._line_bonds

    def advance(self, x):
        """Crack advance in bonds: the sum over crack-line bonds of the fraction each is stretched
        from its reference length towards `advance_break` (clipped to [0, 1])."""
        tot = 0.0
        for a, b, d0 in self.line_bonds:
            tot += float(np.clip((self.bond_between(x, a, b) - d0) / (self.advance_break - d0),
                                 0.0, 1.0))
        return tot


class HessianSurrogate:
    """Approximate Jacobian dF/dx: the Hessian of `calc` (a matscipy many-body calculator with a
    `hessian` property) on the heavy atoms, times a calibrated scale, plus anisotropic X-H springs
    (stretch ks, bend kb, eV/A^2) for H caps (`h_host`). `jac` is dense; `jac_sparse` keeps the
    calculator's sparse Hessian sparse (for `TraceSettings(surrogate="sparse")`)."""

    def __init__(self, problem, calc, ks=16.0, kb=1.5, scale=1.0):
        self.p, self.calc = problem, calc
        self.ks, self.kb, self.scale = ks, kb, scale
        a = problem.atoms
        self.heavy = np.flatnonzero(a.numbers != 1)
        self.h = np.flatnonzero(a.numbers == 1)
        self.host = a.arrays["h_host"] if "h_host" in a.arrays else -np.ones(len(a), int)

    def _heavy_hessian(self, pos):
        sub = Atoms(self.p.atoms.numbers[self.heavy], positions=pos[self.heavy],
                    cell=self.p.atoms.cell, pbc=self.p.atoms.pbc)
        sub.calc = self.calc
        return self.calc.get_property("hessian", sub)

    def _springs(self, pos):
        """(cap, host, 3x3 X-H spring matrix) for every H cap."""
        for h in self.h:
            s = self.host[h]
            nv = pos[h] - pos[s]
            nv /= np.linalg.norm(nv)
            yield h, s, self.ks * np.outer(nv, nv) + self.kb * (np.eye(3) - np.outer(nv, nv))

    def hessian(self, x):
        pos = self.p.positions(x)
        n_at = len(pos)
        Hs = self._heavy_hessian(pos)
        Hs = Hs.toarray() if hasattr(Hs, "toarray") else np.asarray(Hs)
        H = np.zeros((n_at, 3, n_at, 3))
        H[np.ix_(self.heavy, range(3), self.heavy, range(3))] = \
            Hs.reshape(len(self.heavy), 3, len(self.heavy), 3) * self.scale
        for h, s, K in self._springs(pos):
            H[h, :, h, :] += K
            H[s, :, s, :] += K
            H[h, :, s, :] -= K
            H[s, :, h, :] -= K
        return H.reshape(3 * n_at, 3 * n_at)

    def jac(self, x):
        p = self.p
        n_at = len(p.atoms)
        H = self.hessian(x).reshape(n_at, 3, n_at, 3)
        HII = H[np.ix_(p.free, range(3), p.free, range(3))].reshape(p.nf, p.nf)
        HIa = H[np.ix_(p.free, range(3), np.arange(n_at), range(3))]
        dFdk = -np.einsum("iajb,jb->ia", HIa, p.U).ravel()
        return np.hstack([-HII, dFdk[:, None]])

    def hessian_sparse(self, x):
        """`hessian` as a scipy.sparse CSR matrix (3N x 3N), never densified."""
        pos = self.p.positions(x)
        n_at = len(pos)
        Hs = sp.coo_matrix(self._heavy_hessian(pos))
        d3 = (3 * self.heavy[:, None] + np.arange(3)).ravel()
        rows, cols, vals = [d3[Hs.row]], [d3[Hs.col]], [Hs.data * self.scale]
        b = np.arange(3)
        for h, s, K in self._springs(pos):
            for i, j, sign in ((h, h, 1), (s, s, 1), (h, s, -1), (s, h, -1)):
                rows.append(np.repeat(3 * i + b, 3))
                cols.append(np.tile(3 * j + b, 3))
                vals.append(sign * K.ravel())
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(3 * n_at, 3 * n_at)).tocsr()

    def jac_sparse(self, x):
        """`jac` as a scipy.sparse CSR matrix (nf x nf+1): [-H_II | dF/dk]."""
        p = self.p
        rows = self.hessian_sparse(x)[(3 * p.free[:, None] + np.arange(3)).ravel()]
        dFdk = -(rows @ p.U.ravel())
        return sp.hstack([-rows[:, (3 * p.free[:, None] + np.arange(3)).ravel()],
                          sp.csr_matrix(dFdk[:, None])]).tocsr()

    def calibrate(self, x, F0=None, n=2, eps=5e-3, seed=0, clip=(0.3, 3.0)):
        """Fit the scale from n finite-difference Hessian-vector products of the true forces along
        random unit vectors over the heavy-atom free dofs; clipped to `clip`."""
        p = self.p
        F0 = p.residual(x) if F0 is None else F0
        self.scale = 1.0
        Js = self.jac_sparse(x)
        is_heavy = np.repeat(np.isin(p.free, self.heavy), 3)
        rng = np.random.default_rng(seed)
        num = den = 0.0
        for _ in range(n):
            v = np.zeros(len(x))
            v[:-1][is_heavy] = rng.normal(size=is_heavy.sum())
            v /= np.linalg.norm(v)
            Jv = (p.residual(x + eps * v) - F0) / eps
            num += v[:-1] @ Jv
            den += v[:-1] @ (Js @ v)
        self.scale = float(np.clip(num / den, *clip))
        return self.scale


@dataclass
class CorrectorResult:
    y: np.ndarray
    F: np.ndarray
    iters: int
    calls: int
    ok: bool


class LowRankJacobian:
    """Corrector Jacobian B = S + U V^T for large clusters (`TraceSettings(surrogate="sparse")`).

    S (nf x nf+1, scipy.sparse) is the surrogate Jacobian at the last refresh, U V^T (U: nf x m,
    V: nf+1 x m) the limited-memory good-Broyden correction learned from the true forces since,
    one column pair per update, compressed by truncated SVD to at most `max_rank` terms at each
    refresh. The bordered system [B; t] d = r is solved by block elimination on S's free-free
    block S_u (one sparse LU per refresh, reused by the tangent, every corrector iteration and the
    Krylov preconditioner) and Sherman-Morrison-Woodbury for U V^T (S_u^-1 U cached), plus one step
    of iterative refinement, which guards the block elimination near folds, where S_u (a Hessian)
    is nearly singular. Exact algebra: the same B as the dense path, without the (3N)^2 storage and
    O(N^3) solves."""

    def __init__(self, S, U=None, V=None, max_rank=120):
        self.S = S.tocsr() if sp.issparse(S) else sp.csr_matrix(np.asarray(S))
        self.n = self.S.shape[0]
        self.U = np.zeros((self.n, 0)) if U is None else np.asarray(U, float)
        self.V = np.zeros((self.n + 1, 0)) if V is None else np.asarray(V, float)
        self.max_rank = int(max_rank)
        self._lu = None  # splu of S_u, on the first solve
        self._W = np.zeros((self.n, 0))  # S_u^-1 U, grown column by column
        self._wb = None  # S_u^-1 b, b = S's k column

    @property
    def shape(self):
        return (self.n, self.n + 1)

    def __matmul__(self, y):
        return self.S @ y + self.U @ (self.V.T @ y)

    def copy(self):
        c = LowRankJacobian(self.S, self.U.copy(), self.V.copy(), self.max_rank)
        c._lu, c._wb = self._lu, self._wb  # S is shared and never modified in place
        c._W = self._W[:, :c.U.shape[1]].copy()
        return c

    def toarray(self):
        return self.S.toarray() + self.U @ self.V.T

    def update(self, dy, dF):
        """Good Broyden: B += (dF - B dy) dy^T / |dy|^2, as one new column pair."""
        self.U = np.hstack([self.U, ((dF - self @ dy) / (dy @ dy))[:, None]])
        self.V = np.hstack([self.V, dy[:, None]])

    def refresh(self, Sn, mode="projected", Q=None):
        """The B for a new surrogate Jacobian Sn (`BroydenCorrector.refresh`): "reset" Sn,
        "carry" Sn + U V^T, "projected" Sn + U V^T - (Sn - S) Q Q^T; compressed to max_rank."""
        Sn = Sn.tocsr() if sp.issparse(Sn) else sp.csr_matrix(np.asarray(Sn))
        if mode == "reset":
            return LowRankJacobian(Sn, max_rank=self.max_rank)
        U, V = self.U, self.V
        if mode == "projected" and Q is not None and Q.shape[1]:
            U = np.hstack([U, -((Sn - self.S) @ Q)])
            V = np.hstack([V, Q])
        out = LowRankJacobian(Sn, U, V, self.max_rank)
        out.compress()
        return out

    def compress(self):
        """Truncated SVD of U V^T to at most max_rank terms (drops the smallest)."""
        if self.U.shape[1] <= self.max_rank:
            return
        qu, ru = np.linalg.qr(self.U)
        qv, rv = np.linalg.qr(self.V)
        a, sv, bt = np.linalg.svd(ru @ rv.T)
        r = self.max_rank
        self.U = qu @ (a[:, :r] * sv[:r])
        self.V = qv @ bt[:r].T
        self._W = np.zeros((self.n, 0))

    def _factor(self):
        if self._lu is None:
            self._lu = splu(self.S[:, :self.n].tocsc())
            self._wb = self._lu.solve(self.S[:, self.n].toarray().ravel())

    def _W_cols(self):
        m = self.U.shape[1]
        if self._W.shape[1] < m:
            new = self._lu.solve(np.ascontiguousarray(self.U[:, self._W.shape[1]:]))
            self._W = np.hstack([self._W, new.reshape(self.n, -1)])
        return self._W[:, :m]

    def _solve_once(self, t, rF, rg):
        tu, tk = t[:self.n], t[self.n]
        den = tk - tu @ self._wb
        z = self._lu.solve(rF)
        dk = (rg - tu @ z) / den
        x0 = np.concatenate([z - self._wb * dk, [dk]])  # [S; t]^-1 r by block elimination
        m = self.U.shape[1]
        if m == 0:
            return x0
        W = self._W_cols()
        dkw = -(tu @ W) / den
        Z = np.vstack([W - np.outer(self._wb, dkw), dkw[None, :]])  # [S; t]^-1 [U; 0]
        return x0 - Z @ np.linalg.solve(np.eye(m) + self.V.T @ Z, self.V.T @ x0)

    def bordered_solve(self, t, rhs_F, rhs_g, refine=1):
        """Solve [B; t] d = [rhs_F; rhs_g], with `refine` steps of iterative refinement."""
        self._factor()
        d = self._solve_once(t, rhs_F, rhs_g)
        for _ in range(refine):
            d = d + self._solve_once(t, rhs_F - self @ d, rhs_g - t @ d)
        return d


def bordered_solve(J, t, rhs_F, rhs_g):
    """Solve [J; t] d = [rhs_F; rhs_g] (the pseudo-arclength bordered system); J dense or a
    `LowRankJacobian`."""
    if isinstance(J, LowRankJacobian):
        return J.bordered_solve(t, rhs_F, rhs_g)
    return np.linalg.solve(np.vstack([J, t[None, :]]), np.concatenate([rhs_F, [rhs_g]]))


def _limit(d, y, dr_fn, step_limit):
    dm = dr_fn(y + d, y)
    return d * (step_limit / dm) if dm > step_limit else d


class BroydenCorrector:
    """Quasi-Newton on the bordered system F(y) = 0, t.(y - yp) = 0; B starts from the surrogate
    Jacobian and gets a good-Broyden rank-1 update from every force call. `Q` (orthonormal columns)
    spans the update directions since the last `refresh`: the subspace on which B holds secant
    information from the true forces rather than the surrogate's. B is a dense array or a
    `LowRankJacobian` (`max_rank` terms, used when a refresh starts one from a sparse Jn)."""

    def __init__(self, ftol=2e-3, max_iter=8, step_limit=0.1, max_rank=120):
        self.ftol, self.max_iter, self.step_limit = ftol, max_iter, step_limit
        self.max_rank = max_rank
        self.B = None
        self.Q = None

    def update(self, dy, dF):
        if dy @ dy < 1e-14:
            return
        if isinstance(self.B, LowRankJacobian):
            self.B.update(dy, dF)
        else:
            self.B += np.outer(dF - self.B @ dy, dy) / (dy @ dy)
        q = dy / np.linalg.norm(dy)
        if self.Q is None:
            self.Q = q[:, None].copy()
            return
        for _ in range(2):  # Gram-Schmidt, twice for stability
            q = q - self.Q @ (self.Q.T @ q)
        nq = np.linalg.norm(q)
        if nq > 1e-6:
            self.Q = np.hstack([self.Q, (q / nq)[:, None]])

    def refresh(self, Jn, Js, mode="projected"):
        """Merge a new surrogate Jacobian Jn (previous one Js) into B and restart the span.

        "projected" adds Jn - Js only off the span Q: on Q, B already holds the true forces' current
        secants, which contain that change, and adding it there too counts it twice per refresh
        (near a fold, where the critical curvature changes fast, this made the tangent retrace the
        stable branch). "carry" adds it everywhere (the behaviour before), "reset" sets B = Jn.
        A `LowRankJacobian` B takes the same rule on its low-rank part (Js is then its S). With no
        B yet, B = Jn (a `LowRankJacobian` if Jn is sparse); with no previous Js, a dense B has no
        change to merge and is kept."""
        if isinstance(self.B, LowRankJacobian):
            self.B = self.B.refresh(Jn, mode, self.Q)
        elif self.B is None and sp.issparse(Jn):
            self.B = LowRankJacobian(Jn, max_rank=self.max_rank)
        elif mode == "reset" or self.B is None:
            self.B = Jn.copy()
        elif Js is not None:
            dJ = Jn - Js
            if mode == "projected" and self.Q is not None:
                dJ = dJ - (dJ @ self.Q) @ self.Q.T
            self.B = self.B + dJ
        self.Q = None

    def tangent(self, sec):
        """Unit null vector of B bordered by and oriented along the unit secant `sec` (sec if the
        bordered solve fails)."""
        try:
            tau = bordered_solve(self.B, sec, np.zeros(len(sec) - 1), 1.0)
        except (np.linalg.LinAlgError, RuntimeError):  # splu raises RuntimeError when singular
            return sec
        nrm = np.linalg.norm(tau)
        if not np.isfinite(nrm) or nrm == 0:
            return sec
        tau = tau / nrm
        return tau if tau @ sec > 0 else -tau

    def correct(self, F_fn, y, Fy, yp, t, dr_fn):
        calls, fprev = 0, np.inf
        it = 0
        for it in range(self.max_iter + 1):
            g = t @ (y - yp)
            fm = np.abs(Fy).max()
            if fm < self.ftol and abs(g) < 1e-6:
                return CorrectorResult(y, Fy, it, calls, True)
            if it == self.max_iter or (it >= 3 and fm > 0.7 * fprev) or fm > 50:
                break
            fprev = fm
            d = _limit(bordered_solve(self.B, t, -Fy, -g), y, dr_fn, self.step_limit)
            Fn = F_fn(y + d)
            calls += 1
            self.update(d, Fn - Fy)
            y, Fy = y + d, Fn
        return CorrectorResult(y, Fy, it, calls, False)


class KrylovCorrector:
    """Jacobian-free Newton-Krylov on the bordered system (GMRES; finite-difference J v, or an
    exact `hvp(y, v)` = J(y) v when given), right preconditioned by an approximate Jacobian (dense
    or `LowRankJacobian`); at most max_calls force calls (hvp calls are not counted)."""

    def __init__(self, ftol=2e-3, max_iter=8, eps_fd=1e-3, eta=0.2, gmres_max=60, step_limit=0.1,
                 max_calls=40):
        self.ftol, self.max_iter, self.eps_fd = ftol, max_iter, eps_fd
        self.eta, self.gmres_max = eta, gmres_max
        self.step_limit, self.max_calls = step_limit, max_calls

    def correct(self, F_fn, y, Fy, yp, t, dr_fn, prec_J=None, hvp=None):
        n = len(y)
        if prec_J is None:
            psolve = None
        elif isinstance(prec_J, LowRankJacobian):  # factored once per surrogate refresh
            psolve = lambda z: prec_J.bordered_solve(t, z[:-1], z[-1], refine=0)  # noqa: E731
        else:
            lu = sla.lu_factor(np.vstack([prec_J, t[None, :]]))
            psolve = lambda z: sla.lu_solve(lu, z)  # noqa: E731
        calls = [0]
        best = (y.copy(), Fy.copy())
        f0 = np.abs(Fy).max()

        class _Out(Exception):
            pass

        def Fb(yv):
            if calls[0] >= self.max_calls:
                raise _Out
            calls[0] += 1
            return F_fn(yv)

        it = 0
        try:
            for it in range(self.max_iter + 1):
                g = t @ (y - yp)
                fm = np.abs(Fy).max()
                if fm < self.ftol and abs(g) < 1e-6:
                    return CorrectorResult(y, Fy, it, calls[0], True)
                if it == self.max_iter or not np.isfinite(fm) or fm > 50 * max(f0, self.ftol):
                    break

                def mv(v, y=y, Fy=Fy):
                    if hvp is not None:
                        return np.concatenate([hvp(y, v), [t @ v]])
                    nv = np.linalg.norm(v)
                    if nv == 0:
                        return np.zeros(n)
                    e = self.eps_fd / nv
                    return np.concatenate([(Fb(y + e * v) - Fy) / e, [t @ v]])

                rhs = -np.concatenate([Fy, [g]])
                if psolve is not None:
                    op = LinearOperator((n, n), matvec=lambda z: mv(psolve(z)))
                else:
                    op = LinearOperator((n, n), matvec=mv)
                z, _ = gmres(op, rhs, rtol=self.eta, atol=0.0, restart=self.gmres_max, maxiter=1)
                d = psolve(z) if psolve is not None else z
                y = y + _limit(d, y, dr_fn, self.step_limit)
                Fy = Fb(y)
                if np.abs(Fy).max() < np.abs(best[1]).max():
                    best = (y.copy(), Fy.copy())
        except _Out:
            pass
        return CorrectorResult(best[0], best[1], it, calls[0], False)


@dataclass
class TraceSettings:
    ftol: float = 2e-3            # eV/A, max force component on free atoms
    alpha: float = 2.0            # A per K_G in the arclength metric
    ds0: float = 0.05
    ds_min: float = 1e-3
    ds_max: float = 0.6
    dr_cap: float = 0.05          # A, max atomic move of a predictor step
    n_target: int = 3             # corrector iterations aimed at per step
    max_iter: int = 8
    broyden_retries: int = 2
    surrogate_every: int = 5      # refresh the surrogate Jacobian every this many points
    max_points: int = 400
    k_window: tuple = (0.0, 3.0)
    stop_after_folds: int = 2
    extra_after_last_fold: float = 0.25  # |dk| past the last fold before stopping
    reject_factor: float = 3.0    # a step moving an atom > reject_factor * dr_cap is rejected
    max_calls_per_point: int = 30
    krylov_max_calls: int = 40
    corrector_step_limit: float = 0.1
    start_max_iter: int = 200     # force calls of the fixed-k start relaxation
    max_advance: float = None     # crack advance (bonds) per step; None = off
    # predictor: "jacobian" = null vector of the corrector's B, oriented along the secant;
    # "secant" = the chord between the last two points (before). A point accepted at
    # max|F| < ftol sits |H^-1 F| off the curve along the soft modes, which on clusters of > ~1300
    # atoms is comparable to ds; the chord then points tens of degrees off the curve, and a
    # predictor accepted without correction repeats it, so the corrector's hyperplane ends up
    # nearly along the branch and every step is rejected as a jump.
    tangent: str = "jacobian"
    refresh: str = "projected"    # surrogate refresh into B: projected / carry / reset
    # corrector Jacobian B: "dense" = the surrogate's Jacobian as a dense array (rank-1 updated);
    # "sparse" = the surrogate's Jacobian kept sparse (its `jac_sparse` if it has one, else `jac`)
    # plus a limited-memory Broyden correction of at most max_rank terms (`LowRankJacobian`):
    # the same B to round-off, without the (3N)^2 matrices and dense O(N^3) solves
    surrogate: str = "dense"
    max_rank: int = 120

    def __post_init__(self):
        if self.tangent not in ("jacobian", "secant"):
            raise ValueError(f"tangent must be 'jacobian' or 'secant', not {self.tangent!r}")
        if self.refresh not in ("projected", "carry", "reset"):
            raise ValueError(f"refresh must be 'projected', 'carry' or 'reset', not "
                             f"{self.refresh!r}")
        if self.surrogate not in ("dense", "sparse"):
            raise ValueError(f"surrogate must be 'dense' or 'sparse', not {self.surrogate!r}")
        if self.max_rank < 1:
            raise ValueError(f"max_rank must be positive, not {self.max_rank!r}")


class _Stall(Exception):
    pass


def _relax_start(p, x, st):
    """Relax the free atoms at fixed k with ASE LBFGS (clamped atoms fixed)."""
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS
    k = float(x[-1])

    class _Truth(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            E, F = p.evaluate(p.x_from(atoms.positions, k))
            self.results = {"energy": E, "forces": F}

    a = p.atoms_at(x)
    a.set_constraint(FixAtoms(indices=p.clamp))
    a.calc = _Truth()
    LBFGS(a, maxstep=2 * st.dr_cap, logfile=None).run(fmax=st.ftol, steps=st.start_max_iter)
    xs = p.x_from(a.positions, k)
    if np.abs(p.residual(xs)).max() >= st.ftol:
        raise RuntimeError("start relaxation did not reach ftol at fixed k")
    return xs


def _k_turns(ks):
    """dk changes sign at ks[-2] (a fold); ks[0] is the start. Not when ks[-3] is the start: the
    start is only relaxed to max|F| < ftol at fixed k, so it can sit further off the curve than the
    first step is long, and the first corrected point can then lie below it in k (crackcell,
    110_110 at 18816 atoms: 0.8 -> 0.790 -> 0.794, counted as a fold at 0.750)."""
    return len(ks) >= 4 and (ks[-2] - ks[-3]) * (ks[-1] - ks[-2]) < 0


def _fold_estimate(ys, ks):
    s = np.array([0.0, np.linalg.norm(ys[1] - ys[0]),
                  np.linalg.norm(ys[1] - ys[0]) + np.linalg.norm(ys[2] - ys[1])])
    c = np.polyfit(s - s[1], ks, 2)
    if c[0] == 0:
        return float(ks[1])
    sv = -c[1] / (2 * c[0])
    return float(np.polyval(c, sv)) if abs(sv) <= s[2] - s[0] else float(ks[1])


def trace_s_curve(problem, surrogate, k0, *, direction=1, settings=None, callback=None):
    """Trace the S-curve K(u) of a clamped crack cluster from load k0 = K/K_G.

    The start (the problem's geometry, moved rigidly with the field from its build load to k0) is
    relaxed at fixed k; then a tangent predictor (`tangent`; capped to dr_cap and, with max_advance,
    to that crack advance) and bordered Broyden corrector (Newton-Krylov after broyden_retries
    halvings), with the surrogate Jacobian refreshed (`refresh`) every surrogate_every points.
    The corrector Jacobian is dense or sparse plus low rank (`surrogate`; the surrogate's `jac`
    may return either). If the problem has `hvp(x, v)`, the exact directional derivative of the
    free-atom forces dF/dx(x) v (x and v in (u, k)), Newton-Krylov uses it instead of finite
    differences of the forces.
    Folds are sign changes of dk/ds (not at the first point after the start); the branch label
    starts "stable" and alternates at each fold. Stops after stop_after_folds folds plus
    extra_after_last_fold in k, on leaving k_window moving outward, at max_points, or when a step
    needs more than max_calls_per_point force calls or ds falls below ds_min ("stalled").
    `callback(frame)` is called for every accepted point.

    Returns (frames, status): extxyz-ready Atoms with arrays `forces` and info `K_over_KG`, `K`,
    `energy`, `branch`, `point_index`, `method`, `ds`, `calls`; status one of "folds", "window",
    "complete", "stalled".
    """
    st, p = settings or TraceSettings(), problem
    al = st.alpha
    capped = st.max_advance is not None
    c0 = [p.ncalls]

    def to_x(y):
        x = y.copy()
        x[-1] = y[-1] / al
        return x

    def to_y(x):
        y = np.array(x, float)
        y[-1] = x[-1] * al
        return y

    def F(y):
        if p.ncalls - c0[0] >= st.max_calls_per_point:
            raise _Stall
        return p.residual(to_x(y))

    def dr(y1, y0):
        return p.dr_max(to_x(y1), to_x(y0))

    def adv(y1, y0):
        return abs(p.advance(to_x(y1)) - p.advance(to_x(y0)))

    def jy(yy):
        xx = to_x(yy)
        if st.surrogate == "sparse":
            J = surrogate.jac_sparse(xx) if hasattr(surrogate, "jac_sparse") else surrogate.jac(xx)
            d = np.ones(J.shape[1])
            d[-1] = 1.0 / al
            return (sp.csr_matrix(J) @ sp.diags(d)).tocsr()
        J = surrogate.jac(xx)
        J = J.toarray() if sp.issparse(J) else np.array(J, float)
        J[:, -1] /= al
        return J

    hvp = None
    if callable(getattr(p, "hvp", None)):
        def hvp(yy, v):
            return p.hvp(to_x(yy), to_x(v))

    broyden = BroydenCorrector(st.ftol, st.max_iter, st.corrector_step_limit, st.max_rank)
    krylov = KrylovCorrector(st.ftol, st.max_iter, step_limit=st.corrector_step_limit,
                             max_calls=st.krylov_max_calls)
    frames, folds = [], []
    branch = ["stable"]

    def emit(y, method, ds):
        xx = to_x(y)
        E, Fall = p.evaluate(xx)
        a = p.atoms_at(xx)
        a.set_array("forces", Fall)
        a.info.update(K_over_KG=float(xx[-1]), K=float(xx[-1] * p.kg), energy=E, branch=branch[0],
                      point_index=len(frames), method=method, ds=float(ds),
                      calls=int(p.ncalls - c0[0]))
        frames.append(a)
        if callback is not None:
            callback(a)

    x = p.x_from(p.atoms.positions, p.k_build)
    x[-1] = k0
    x = _relax_start(p, x, st)
    surrogate.calibrate(x)
    y = to_y(x)
    Js = jy(y)
    broyden.B = LowRankJacobian(Js, max_rank=st.max_rank) if st.surrogate == "sparse" else \
        Js.copy()
    Js_age = 0
    Fy = p.residual(x)
    emit(y, "start", 0.0)
    ys, Fs, ks = [y], [Fy], [x[-1]]
    e_k = np.zeros(len(y))
    e_k[-1] = direction
    t = bordered_solve(broyden.B, e_k, np.zeros(len(y) - 1), 1.0)
    t = t / np.linalg.norm(t) * np.sign(t[-1] * direction or 1.0)
    ds, status, k_last_fold = st.ds0, "complete", None
    while len(frames) < st.max_points:
        yprev, Fprev = ys[-1], Fs[-1]
        if len(ys) >= 2:
            t = (yprev - ys[-2]) / np.linalg.norm(yprev - ys[-2])
            if st.tangent == "jacobian":
                t = broyden.tangent(t)
        dm = dr(yprev + ds * t, yprev)
        if dm > st.dr_cap:
            ds *= st.dr_cap / dm
        if capped:
            da = adv(yprev + ds * t, yprev)
            if da > st.max_advance:
                ds *= st.max_advance / da
        c0[0], attempt = p.ncalls, 0
        try:
            while True:
                if ds < st.ds_min:
                    raise _Stall
                yp = yprev + ds * t
                Fp = F(yp)
                broyden.update(yp - yprev, Fp - Fprev)
                res, used = broyden.correct(F, yp, Fp, yp, t, dr), "broyden"
                if not res.ok and attempt > st.broyden_retries:
                    y0, F0 = (yp, Fp) if np.abs(Fp).max() <= np.abs(res.F).max() else (res.y, res.F)
                    res = krylov.correct(F, y0, F0, yp, t, dr, prec_J=broyden.B, hvp=hvp)
                    used = "krylov"
                ok = res.ok and dr(res.y, yprev) <= st.reject_factor * st.dr_cap and \
                    t @ (res.y - yprev) > 0 and \
                    not (capped and adv(res.y, yprev) > 2 * st.max_advance)
                if ok:
                    break
                ds *= 0.5
                attempt += 1
                if attempt == st.broyden_retries + 1:  # fresh surrogate before the Krylov attempts
                    Jn = jy(yprev)
                    broyden.refresh(Jn, Js, st.refresh)  # keeps the learned correction
                    Js, Js_age = Jn, 0
        except _Stall:
            status = "stalled"
            break
        ys.append(res.y)
        Fs.append(res.F)
        ks.append(res.y[-1] / al)
        if _k_turns(ks):
            k_last_fold = _fold_estimate(ys[-3:], np.array(ks[-3:]))
            folds.append(k_last_fold)
            branch[0] = "unstable" if branch[0] == "stable" else "stable"
        emit(res.y, used, ds)
        ds_next = ds * np.clip(st.n_target / max(res.iters, 1), 0.5, 1.6)
        if capped:
            da = adv(res.y, yprev)
            if da > 0:
                ds_next = min(ds_next, ds * st.max_advance / da)
        ds = float(np.clip(ds_next, st.ds_min, st.ds_max))
        Js_age += 1
        if Js_age >= st.surrogate_every:
            Jn = jy(res.y)
            broyden.refresh(Jn, Js, st.refresh)
            Js, Js_age = Jn, 0
        k_now, dk = res.y[-1] / al, (res.y[-1] - yprev[-1]) / al
        lo, hi = st.k_window
        if (k_now < lo and dk <= 0) or (k_now > hi and dk >= 0):
            status = "window"
            break
        if k_last_fold is not None and len(folds) >= st.stop_after_folds and \
                abs(k_now - k_last_fold) > st.extra_after_last_fold:
            status = "folds"
            break
    return frames, status
