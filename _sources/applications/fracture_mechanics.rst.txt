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

.. toctree::

    quasi_static_crack