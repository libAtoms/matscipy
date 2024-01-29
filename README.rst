|Tests| |Wheels| |JOSS|

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
general-purpose, low-level utilities:

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

Editable installs
~~~~~~~~~~~~~~~~~

When developing `matscipy`, it can be useful to have an editable install of
the source directory. This means that changes to the source code are directly
reflected in the `matscipy` install. We are using *Meson* and *meson-python* as a
build system, and there are some `restriction to editable installs <https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html>`__.

The editable install only works with the
`--no-build-isolation` option::

  python3 -m pip install --no-build-isolation --editable .[test]

If you get the message::

  ERROR: Tried to form an absolute path to a dir in the source tree.

then you are most likely try to install into a Python virtual environment that
is located inside your source directory. This is not possible; your virtual
environment needs to be located outside of the source directory.


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

Citing matscipy
---------------

Please cite the following publication if you use matscipy::

  @article{Grigorev2024,
    author = {Grigorev, Petr and Frérot, Lucas and Birks, Fraser and Gola, Adrien and Golebiowski, Jacek and Grießer, Jan and Hörmann, Johannes L. and Klemenz, Andreas and Moras, Gianpietro and Nöhring, Wolfram G. and Oldenstaedt, Jonas A. and Patel, Punit and Reichenbach, Thomas and Rocke, Thomas and Shenoy, Lakshmi and Walter, Michael and Wengert, Simon and Zhang, Lei and Kermode, James R. and Pastewka, Lars},
    doi = {10.21105/joss.05668},
    journal = {Journal of Open Source Software},
    month = jan,
    number = {93},
    pages = {5668},
    title = {{matscipy: materials science at the atomic scale with Python}},
    url = {https://joss.theoj.org/papers/10.21105/joss.05668},
    volume = {9},
    year = {2024}
  }

Funding
-------

**matscipy** was partially funded by the Deutsch Forschungsgemeinschaft (project `258153560 <https://gepris.dfg.de/gepris/projekt/258153560>`__) and by the Engineering and Physical Sciences Research Council (grants `EP/P002188/1 <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/P002188/1>`__, `EP/R012474/1 <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/R012474/1>`__ and `EP/R043612/1 <https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/R043612/1>`__).

.. |Tests| image:: https://github.com/libAtoms/matscipy/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/libAtoms/matscipy/actions/workflows/tests.yml

.. |Wheels| image:: https://github.com/libAtoms/matscipy/actions/workflows/wheels.yml/badge.svg
   :target: https://github.com/libAtoms/matscipy/actions/workflows/wheels.yml

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.05668/status.svg
   :target: https://doi.org/10.21105/joss.05668
