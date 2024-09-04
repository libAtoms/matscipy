Plasticity
==========

For large loads, solids can respond with irreversible deformation. One form of irreversibility is plasticity, that is carried by extended defects, the dislocations, in crystals. The module :mod:`matscipy.dislocation` implements tools for studying structure and movement of dislocations. Construction and analysis of model atomic systems is implemented for compact and dissociated screw, as well as edge dislocations in cubic crystals. The implementation supports ideal straight as well as kinked dislocations. 

Creating an atomistic system containing a dislocation requires a rather deep knowledge of crystallographic system, elasticity theory as well as hands on experience with atomistic simulations. Thus, it can be challenging for people not familiar to the field. Within this module we attempt to create a flexible, friendly and pythonic tool to enable atomistic simulations of dislocations with ease. The base of the model is :class:`matscipy.dislocation.CubicCrystalDislocation` class that contains all the necessary information to create an atomistic dislocation. To start experimenting, a user only has to choose the dislocation of interest and provide a lattice parameter and elastic constants.

Some dislocation systems feature additional complexity, as they can dissociate into two partial dislocations connected by a stacking fault. These kinds of dislocations are implemented within the subclass :class:`matscipy.dislocation.CubicCrystalDissociatedDislocation` and follow a near-identical interface. :mod:`matscipy.gamma_surface` implements classes which assist in the modeling of stacking faults, as well as more general gamma surfaces. The classes can be initialised based on a dislocation system, or by known axes. 

Installation and tests
----------------------

:mod:`matscipy.dislocation` module relies on the anisotropic elasticity theory solution within Stroh formalism implemented in `atomman <https://www.ctcms.nist.gov/potentials/atomman/>`_ package. `atomman <https://www.ctcms.nist.gov/potentials/atomman/>`_ is not a part of default dependency of :mod:`matscipy`, however it can be easily installed with ``pip install atomman``. To test your installation you can run ``pytest test_dislocation.py`` from ``tests`` directory of the repository. If atomman is not installed, most of the tests will be skipped resulting in the output similar to the following: ``ssssssssssssss..ssssssss..s.ssssssss.s`` where ``s`` stands for skipped test and ``.`` for passed tests. You can see which test are skipped together with the reasoning behind by running pytest in verbose mode with ``pytest -v test_dislocation.py``. Once you have atomman installed you should have corresponding tests passing and should be able to use the code. If you plan to use `LAMMPS <https://lammps.org>`_ and `lammps python module <https://docs.lammps.org/Python_module.html>`_ there are couple of test that will test your installation when `lammps python module <https://docs.lammps.org/Python_module.html>`_ is detected.

The majority of the tests (~70 %) are used during the development and adding new dislocations to the module. The idea is to check if `Dislocation analysis (DXA) <https://www.ovito.org/manual/reference/pipelines/modifiers/dislocation_analysis.html>`_ algorithm detects the same dislocation as we intend to create. These tests require `OVITO python interface <https://www.ovito.org/manual/python/index.html>`_ to be installed and are skipped if it is not detected. If you do not plan to add new structures to the module, you can safely ignore these tests.

Tutorials:
----------

.. toctree::
    
    cylinder_configurations.ipynb
    quadrupole_dislocations.ipynb
    disloc_mobility.ipynb
    multispecies_dislocations.ipynb
    gamma_surfaces.ipynb
    gamma_surface_advanced.ipynb