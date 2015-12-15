matscipy
========

This repository contains generic Python materials science tools build around the
[Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

Build status
============

[![Build Status](https://travis-ci.org/libAtoms/matscipy.svg?branch=master)](https://travis-ci.org/libAtoms/matscipy)

Compilation/installation follows the standard distutils route:

    python setup.py build
    python setup.py install

If building on Mac OS X, we recommend you use the GCC toolchain
    
    CC=gcc CXX=g++ python setup.py build

Documentation
=============

[Sphinx](http://sphinx-doc.org/)-generated documentation for the project can be found [here](http://libatoms.github.io/matscipy/).

Dependencies
============

The package requires

* numpy - http://www.numpy.org/
* scipy (optional) - http://www.scipy.org/
* ASE - https://wiki.fysik.dtu.dk/ase/

Optional packages

* quippy - http://www.github.com/libAtoms/QUIP
* atomistica - https://www.github.com/Atomistica/atomistica
* chemview - https://github.com/gabrielelanaro/chemview
