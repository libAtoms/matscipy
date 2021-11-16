Matscipy
========

This repository contains generic Python materials science tools built
around the `Atomic Simulation Environment
(ASE) <https://wiki.fysik.dtu.dk/ase/>`__.

Build status
------------

|Build Status|

Compilation/installation follows the standard distutils route:

::

   python setup.py build
   python setup.py install

If building on Mac OS X, we recommend you use the GCC toolchain

::

   CC=gcc CXX=g++ python setup.py build
   
If you have a recent version of gcc you may also need to set

::

    CFLAGS="-std=c99"

Documentation
-------------

`Sphinx <http://sphinx-doc.org/>`__-generated documentation for the
project can be found `here <http://libatoms.github.io/matscipy/>`__.

Dependencies
------------

The package requires :

-  **numpy** - http://www.numpy.org/
-  **scipy** - http://www.scipy.org/
-  **ASE** - https://wiki.fysik.dtu.dk/ase/

Optional packages :

-  **quippy** - http://www.github.com/libAtoms/QUIP
-  **atomistica** - https://www.github.com/Atomistica/atomistica
-  **chemview** - https://github.com/gabrielelanaro/chemview

.. |Build Status| image:: https://travis-ci.org/libAtoms/matscipy.svg?branch=master
   :target: https://travis-ci.org/libAtoms/matscipy
