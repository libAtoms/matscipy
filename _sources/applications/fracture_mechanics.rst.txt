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

.. toctree::

    quasi_static_crack