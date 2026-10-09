Structure and topology generation
===================

Non-reactive force fields for molecular dynamics simulations typically consist of non-bonded and bonded interaction terms. The latter require an explicit specification of the interatomic bonding topology, i.e. which atoms are involved in bond, angle and dihedral interactions. `matscipy` provides efficient tools to generate this topology for an atomic structure based on `matscipy`'s neighbour list, and then assign the relevant force field parameters to each interaction term. `matscipy` also includes the input and output routines for reading and writing the corresponding control files for LAMMPS.

.. toctree::

   setup_non-reactive_simulations
   topology_building_aC
