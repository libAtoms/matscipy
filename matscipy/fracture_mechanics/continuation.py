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
calibrated factor) with good-Broyden updates from every force call, then Newton-Krylov.
"""
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from ase import Atoms
from scipy.sparse.linalg import LinearOperator, gmres


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
    (stretch ks, bend kb, eV/A^2) for H caps (`h_host`)."""

    def __init__(self, problem, calc, ks=16.0, kb=1.5, scale=1.0):
        self.p, self.calc = problem, calc
        self.ks, self.kb, self.scale = ks, kb, scale
        a = problem.atoms
        self.heavy = np.flatnonzero(a.numbers != 1)
        self.h = np.flatnonzero(a.numbers == 1)
        self.host = a.arrays["h_host"] if "h_host" in a.arrays else -np.ones(len(a), int)

    def hessian(self, x):
        p = self.p
        pos = p.positions(x)
        n_at = len(pos)
        sub = Atoms(p.atoms.numbers[self.heavy], positions=pos[self.heavy], cell=p.atoms.cell,
                    pbc=p.atoms.pbc)
        sub.calc = self.calc
        Hs = self.calc.get_property("hessian", sub)
        Hs = Hs.toarray() if hasattr(Hs, "toarray") else np.asarray(Hs)
        H = np.zeros((n_at, 3, n_at, 3))
        H[np.ix_(self.heavy, range(3), self.heavy, range(3))] = \
            Hs.reshape(len(self.heavy), 3, len(self.heavy), 3) * self.scale
        for h in self.h:
            s = self.host[h]
            nv = pos[h] - pos[s]
            nv /= np.linalg.norm(nv)
            K = self.ks * np.outer(nv, nv) + self.kb * (np.eye(3) - np.outer(nv, nv))
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

    def calibrate(self, x, F0=None, n=2, eps=5e-3, seed=0, clip=(0.3, 3.0)):
        """Fit the scale from n finite-difference Hessian-vector products of the true forces along
        random unit vectors over the heavy-atom free dofs; clipped to `clip`."""
        p = self.p
        F0 = p.residual(x) if F0 is None else F0
        self.scale = 1.0
        Js = self.jac(x)
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


def bordered_solve(J, t, rhs_F, rhs_g):
    """Solve [J; t] d = [rhs_F; rhs_g] (the pseudo-arclength bordered system)."""
    return np.linalg.solve(np.vstack([J, t[None, :]]), np.concatenate([rhs_F, [rhs_g]]))


def _limit(d, y, dr_fn, step_limit):
    dm = dr_fn(y + d, y)
    return d * (step_limit / dm) if dm > step_limit else d


class BroydenCorrector:
    """Quasi-Newton on the bordered system F(y) = 0, t.(y - yp) = 0; B starts from the surrogate
    Jacobian and gets a good-Broyden rank-1 update from every force call."""

    def __init__(self, ftol=2e-3, max_iter=8, step_limit=0.1):
        self.ftol, self.max_iter, self.step_limit = ftol, max_iter, step_limit
        self.B = None

    def update(self, dy, dF):
        if dy @ dy < 1e-14:
            return
        self.B += np.outer(dF - self.B @ dy, dy) / (dy @ dy)

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
    """Jacobian-free Newton-Krylov on the bordered system (GMRES; finite-difference J v), right
    preconditioned by an approximate Jacobian; at most max_calls force calls."""

    def __init__(self, ftol=2e-3, max_iter=8, eps_fd=1e-3, eta=0.2, gmres_max=60, step_limit=0.1,
                 max_calls=40):
        self.ftol, self.max_iter, self.eps_fd = ftol, max_iter, eps_fd
        self.eta, self.gmres_max = eta, gmres_max
        self.step_limit, self.max_calls = step_limit, max_calls

    def correct(self, F_fn, y, Fy, yp, t, dr_fn, prec_J=None):
        n = len(y)
        lu = sla.lu_factor(np.vstack([prec_J, t[None, :]])) if prec_J is not None else None
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
                    nv = np.linalg.norm(v)
                    if nv == 0:
                        return np.zeros(n)
                    e = self.eps_fd / nv
                    return np.concatenate([(Fb(y + e * v) - Fy) / e, [t @ v]])

                rhs = -np.concatenate([Fy, [g]])
                if lu is not None:
                    op = LinearOperator((n, n), matvec=lambda z: mv(sla.lu_solve(lu, z)))
                else:
                    op = LinearOperator((n, n), matvec=mv)
                z, _ = gmres(op, rhs, rtol=self.eta, atol=0.0, restart=self.gmres_max, maxiter=1)
                d = sla.lu_solve(lu, z) if lu is not None else z
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
    relaxed at fixed k; then secant predictor (capped to dr_cap and, with max_advance, to that crack
    advance) and bordered Broyden corrector (Newton-Krylov after broyden_retries halvings), with the
    surrogate Jacobian refreshed every surrogate_every points. Folds are sign changes of dk/ds; the
    branch label starts "stable" and alternates at each fold. Stops after stop_after_folds folds plus
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
        J = surrogate.jac(to_x(yy)).copy()
        J[:, -1] /= al
        return J

    broyden = BroydenCorrector(st.ftol, st.max_iter, st.corrector_step_limit)
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
    broyden.B = Js.copy()
    Js_age = 0
    Fy = p.residual(x)
    emit(y, "start", 0.0)
    ys, Fs = [y], [Fy]
    e_k = np.zeros(len(y))
    e_k[-1] = direction
    t = bordered_solve(broyden.B, e_k, np.zeros(len(y) - 1), 1.0)
    t = t / np.linalg.norm(t) * np.sign(t[-1] * direction or 1.0)
    ds, status, k_last_fold = st.ds0, "complete", None
    while len(frames) < st.max_points:
        yprev, Fprev = ys[-1], Fs[-1]
        if len(ys) >= 2:
            t = (yprev - ys[-2]) / np.linalg.norm(yprev - ys[-2])
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
                    res, used = krylov.correct(F, y0, F0, yp, t, dr, prec_J=broyden.B), "krylov"
                ok = res.ok and dr(res.y, yprev) <= st.reject_factor * st.dr_cap and \
                    t @ (res.y - yprev) > 0 and \
                    not (capped and adv(res.y, yprev) > 2 * st.max_advance)
                if ok:
                    break
                ds *= 0.5
                attempt += 1
                if attempt == st.broyden_retries + 1:  # fresh surrogate before the Krylov attempts
                    Jn = jy(yprev)
                    broyden.B += Jn - Js  # keep the learned correction to the surrogate
                    Js, Js_age = Jn, 0
        except _Stall:
            status = "stalled"
            break
        ys.append(res.y)
        Fs.append(res.F)
        if len(ys) >= 3:
            k3 = [v[-1] / al for v in ys[-3:]]
            if (k3[1] - k3[0]) * (k3[2] - k3[1]) < 0:
                k_last_fold = _fold_estimate(ys[-3:], np.array(k3))
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
            broyden.B += Jn - Js
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
