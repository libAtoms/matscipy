Fracture Mechanics
==================

Cracking is the process of generating new surface area by splitting the material
apart. The module :mod:`matscipy.fracture_mechanics` provides functionality for
calculating continuum linear elastic displacement fields near crack tips,
including support for anisotropy in the elastic response. 

The module also implements generation of atomic structures that are deformed
according to this near-tip field. This functionality has been used to quantify
lattice trapping, which is the pinning of cracks due to the discreteness of the
atomic lattice, and to compare simulations with experimental measurements of
crack speeds in silicon. An example of this is provided in the 
quasi-static fracture tutorial linked below. 

Finally, there is support for flexible boundary conditions in fracture simulations
using the formalism proposed by Sinclair, where the finite atomistic domain is
coupled to an infinite elastic continuum. We also provide an extension of this
approach to give a flexible boundary scheme that uses numerical continuation to
obtain full solution paths for cracks.

Clamped clusters and S-curves
-----------------------------

:func:`matscipy.fracture_mechanics.clusters.clamped_cluster` builds a clamped K-field crack
cluster: an oriented block with a clamped skin, loaded about the midpoint of the tip bond, with
optional H caps on cut bonds (free crack faces stay bare).
:func:`~matscipy.fracture_mechanics.clusters.cluster_like` puts a reference cluster's topology on
another material's lattice and field, so clusters of different potentials map atom by atom.
:mod:`matscipy.fracture_mechanics.continuation` traces the lattice-trapping S-curve K(u) of such a
cluster with any ASE calculator, by force-only pseudo-arclength continuation in a
:class:`~matscipy.fracture_mechanics.crack.SinclairCrack` in clamped mode (``variable_alpha=False``,
``energy="total"``), with the Hessian of a matscipy many-body potential as the corrector's
surrogate Jacobian:

.. code-block:: python

    from ase.units import GPa, J, m
    from matscipy.calculators.manybody import Manybody
    from matscipy.calculators.manybody.explicit_forms.stillinger_weber import (
        Stillinger_Weber_PRB_31_5262_Si, StillingerWeber)
    from matscipy.fracture_mechanics.clusters import clamped_cluster
    from matscipy.fracture_mechanics.continuation import (ClampedProblem, HessianSurrogate,
                                                          TraceSettings, trace_s_curve)
    from matscipy.fracture_mechanics.crack import (CubicCrystalCrack, SinclairCrack,
                                                   regions_from_groups)

    sw = Manybody(**StillingerWeber(Stillinger_Weber_PRB_31_5262_Si))
    calc = sw  # any ASE calculator: the model whose S-curve is traced
    crk = CubicCrystalCrack([1, -1, 0], [1, 1, 0], C11=151.42 * GPa, C12=76.42 * GPa,
                            C44=56.45 * GPa)
    KG = crk.k1g(1.6657 * J / m**2)
    a = clamped_cluster(crk, 0.6 * KG, (4, 4, 1), (0.5, 1.0), a0=5.43095,
                        crack_surface=[1, -1, 0], crack_front=[1, 1, 0], passivate=False)
    cryst = a.copy()
    cryst.positions = a.arrays["reference"]
    regions_from_groups(cryst)
    crack = SinclairCrack(crk, cryst, calc, 0.6 * KG, alpha=a.info["tip_x"] - cryst.cell[0, 0] / 2,
                          vacuum=0.0, variable_alpha=False, variable_k=True, energy="total")
    p = ClampedProblem(crack, KG, atoms=a)  # atoms: the built cluster is the start geometry
    frames, status = trace_s_curve(p, HessianSurrogate(p, sw), 0.6, settings=TraceSettings())

Each frame carries ``K_over_KG``, ``energy`` and ``branch`` (stable or unstable) in ``info`` and the
forces in ``arrays``; ``status`` says why the trace stopped (``"folds"``, ``"window"``,
``"complete"`` or ``"stalled"``).

On clusters of more than a few thousand atoms the dense corrector Jacobian ((3N)^2 storage, an
O(N^3) solve per corrector iteration) is the limit. ``TraceSettings(surrogate="sparse")`` keeps the
surrogate's Hessian sparse (``HessianSurrogate.jac_sparse``; any surrogate whose ``jac`` returns a
``scipy.sparse`` matrix works too) and holds the Broyden updates as a low-rank correction
(:class:`~matscipy.fracture_mechanics.continuation.LowRankJacobian`, at most ``max_rank`` terms):
one sparse LU per surrogate refresh serves every bordered solve. It is the same Jacobian to
round-off, so the trace is the same. Stillinger-Weber, unpassivated (110)[1-10] clusters, start
k = 0.8, ``ftol=2e-3``, ``k_window=(0.3, 3)``, ``max_calls_per_point=60``,
``start_max_iter=300``, stopped just after the second fold (Apple M3 Pro laptop):

=====  ======  ======  ===================  =============  ========  =====  ========
atoms  B       points  K+ / K- (K_G)        calls / point  rejected  wall   peak RSS
=====  ======  ======  ===================  =============  ========  =====  ========
864    dense   164     1.575250 / 0.601770  1.45           3         80 s   0.57 GB
864    sparse  164     1.575250 / 0.601770  1.45           3         64 s   0.33 GB
1536   dense   192     1.574234 / 0.590595  1.35           2         236 s  1.36 GB
1536   sparse  192     1.574234 / 0.590595  1.35           2         134 s  0.48 GB
3456   sparse  239     1.575392 / 0.580519  1.29           3         456 s  0.90 GB
=====  ======  ======  ===================  =============  ========  =====  ========

Dense and sparse folds agree to 5e-8. If the problem object provides ``hvp(x, v)``, the exact
directional derivative of the free-atom forces, the Newton-Krylov fallback uses it instead of
finite differences of the forces.

.. toctree::

    quasi_static_crack