Matscipy
========

Matscipy is a generic materials science toolbox built around the `Atomic
Simulation Environment (ASE) <https://wiki.fysik.dtu.dk/ase/>`__. It provides
useful routines for:

- Plasticity and dislocations
- Fracture mechanics
- Electro-chemistry
- Tribology
- Elastic properties

In addition to domain-specific routines, it also implements a set of
general-purpose, low-level utilies:

- Efficient neighbour lists
- Atomic strain
- Ring analysis
- Correlation functions
- Second order potential derivatives

Quick start
-----------

Matscipy can be installed on Windows, Linux and x86 macos with::

  python3 -m pip install matscipy

To get the latest version directly (requires a working compiler)::

  python3 -m pip install git+https://github.com/libAtoms/matscipy.git

Compiled up-to-date wheels for Windows, Linux and x86 macos can be found `here
<https://github.com/libAtoms/matscipy/actions/workflows/build-wheels.yml>`__.

Documentation
-------------

`Sphinx <http://sphinx-doc.org/>`__-generated documentation for the project can
be found `here <http://libatoms.github.io/matscipy/>`__. Since Matscipy is built
on top of ASE's `Atoms
<https://wiki.fysik.dtu.dk/ase/ase/atoms.html#module-ase.atoms>`__ and
`Calculator <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`__
objects, ASE's documentation is a good complement to Matscipy's.

Seeking help
------------

`Issues <https://github.com/libAtoms/matscipy/issues>`__ can be used to ask
questions about Matscipy.

Contributing
------------

Contributions, in the form of bug reports, improvement suggestions,
documentation or pull requests, are welcome.

Running tests
~~~~~~~~~~~~~

To run the tests locally, from Matscipy's root directory::

  python3 -m pip install .[test]  # installs matscipy + test dependencies
  cd tests/
  python3 -m pytest .

Dependencies
------------

The package requires:

-  **numpy** - http://www.numpy.org/
-  **scipy** - http://www.scipy.org/
-  **ASE** - https://wiki.fysik.dtu.dk/ase/

Optional packages:

-  **quippy** - http://www.github.com/libAtoms/QUIP
-  **atomistica** - https://www.github.com/Atomistica/atomistica
-  **chemview** - https://github.com/gabrielelanaro/chemview

Funding
-------

**matscipy** was partially funded by the Deutsch Forschungsgemeinschaft (project `258153560 <https://gepris.dfg.de/gepris/projekt/258153560>`__) and by the Engineering and Physical Sciences Research Council (grants `EP/P002188/1 <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/P002188/1>`__, `EP/R012474/1 <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/R012474/1>`__ and `EP/R043612/1 <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/R043612/1>`__).

