Change log
==========

v1.1.0 (not yet released)
-------------------------

- Fixes for recent ASE and numpy 2.0
- Fix correct number of cores for quad of dissociated dislocations

v1.0.0 (24Jan24)
----------------

- JOSS paper!
- Significant updates documentation packages
- Displacement field and associated deformation tensors for dislocations in anisotropic elastic media
- Multiple improvements to arc-length continuation for flexible boundary conditions (NCFlex): Mode II fracture,
  mixed mode I/II fracture, parallel calculations
- Tools for generation of stacking fault & gamma surface (generalised stacking fault) images
- Committees of models as a calculator
- Compatibility with recent ASE and scipy versions
- Scripts are now an installable command-line interface; experimental scripts remain in a "staging" subdirectory
- Updated documentation packages
- Added tool for generating dislocation quadrupoles for any `CubicCrystalDislocation` or `CubicCrystalDissociatedDislocation`
- Refactor of dislocation internals & testing to allow for easy integration of new dislocation types and displacement solver methods.

v0.8.0 (12Apr23) 
----------------

- Calculator for traditional Ewald summation 
- Analytic computation of Hessian and zero-temperature elastic constants for potentials with electrostatic interaction
- Writing and reading inhomogenous (and non numeric) data with savetbl/readtbl
- Hessian-based preconditioner
- Changed build system to Meson
- Removed `contact_mechanics` submodule. Stable alternatives:
  [ContactMechanics](https://github.com/ContactEngineering/ContactMechanics) or
  [Tamaas](https://gitlab.com/tamaas/tamaas)
- New interface for `ManybodyCalculator`, which simplifies definition of
  manybody potential and derivatives

v0.7.0 (29Jul21)
----------------

- Calculator for Manybody potentials e.g. Stillinger-Weber and Bond-order
- Specific parametrizations for Tersoff-Brenner potentials, Stillinger-Weber and Kumagai 
- Analytic computation of Hessian matrix for manybody potentials
- Analytic computation of zero-temperature elastic constants for pair potential and manybody potentials
- CLI for electrochemistry functions
- Proper molecular id numbering in electrochemistry
- Generation of dislocations

v0.6.0 (10Sep20)
----------------

- Numerical computation of the Hessian matrix 
- Calculator for polydisperse systems in which particles interact via a pair potential 
- Analytic computation of Hessian for polydisperse systems
- Bug fix in tests eam_calculator_forces_hessian 

v0.5.1, v0.5.2 (4Sep20)
-----------------------

- Enabling automatic publishing on PyPI

v0.5.0 (4Sep20)
---------------

- Sinclair flexible boundary conditions for cracks and arc-length continuation
- Bug fix in neighbour list search that lead to occasional segfaults

v0.4.0 (5May20)
---------------

- Analytic computation of Hessian for EAM potentials
- Neighbor list can be used without ASE
- Python-3 compatibilty for CASTEP socket calculator
- Electrochemistry module with Poisson-Nernst-Planck solver
- Barostat for sliding systems (Tribol. Lett. 39, 49 (2010))
- Support for kinks in screw dislocations

v0.3.0 (4May19)
---------------

- Analytic computation of Hessian matrix for pair potentials
- Creation of screw and edge dislocation, visualization of dislocations
  dislocation barrier calculation.

v0.2.0 (10May18)
----------------

- QM/MM calculator.
- Neighbor list now accepts dictionary that describes cutoff distances.

v0.1.4 (4Oct17)
---------------

- Compatibility with MS C++ compiler.

v0.1.3 (4Oct17)
--------------

- Regression fix: Add C++11 compiler flag for C extension modules.

v0.1.2 (29Sep17)
----------------

- First release of matscipy: Neighbour list, elastic constants, cracks and more
