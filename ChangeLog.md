Change log
==========

v0.8.0 (not yet released)
-------------------------

- Calculator for bond-order potentials
- Hessian calculation for bond-order potentials
- Specific parameterization for Kumagai and Tersoff

v0.7.0 (17Nov20)
----------------

- CLI for electrochemistry functions
- Proper molecular id numbering in electrochemistry

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
