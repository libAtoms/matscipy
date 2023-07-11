Plasticity
==========

For large loads, solids can respond with irreversible deformation. One form of irreversibility is plasticity, that is carried by extended defects, the dislocations, in crystals. The module :mod:`matscipy.dislocation` implements tools for studying structure and movement of dislocations. Construction and analysis of model atomic systems is implemented for compact and dissociated screw, as well as edge dislocations in cubic crystals. The implementation supports ideal straight as well as kinked dislocations. 

Creating an atomistic system containing a dislocation requires a rather deep knowledge of crystallographic system, elastisity theory as well as hands on experience with atomistic simulations. Thus, it can be challenging for people not familiar to the field. Within this module we attempt to create a flexible, friendly and pythonic tool to enable atomistic simulations of dislocations with ease. The base of the model is :class:`matscipy.dislocation.CubicCrystalDislocation` class that contains all the necessary information to create an atomistic dislocation. To start experimenting, a user only has to choose the dislocation of interest and provide a lattice parameter and elastic constants.

.. toctree::

    cylinder_configurations.ipynb