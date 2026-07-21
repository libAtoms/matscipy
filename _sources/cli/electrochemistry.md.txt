# Electrochemistry

## Overview of commands

* `matscipy-poisson-nernst-planck`: Command line interface to the functionality of
  `matscipy.electrochemistry.posson_nernst_planck_solver` and
  `matscipy.electrochemistry.posson_nernst_planck_solver_fenics`.
  If available `poisson-nernst-planck` will make use of the third-party
  `FEniCS` finite elements solver, but fall back to our own controlled-volumes
  solver otherwise.
* `matscipy-continuous2discrete`: Command line interface to the functionality of
  `matscipy.electrochemistry.continuous2discrete`.
* `matscipy-stericify`: Command line interface to the functionality of 
  `matscipy.electrochemistry.steric_correction`.

## Usage

`matscipy-poisson-nernst-planck`, `matscipy-continuous2discrete` and
`matscipy-stericify` executable scripts offer simple command line interfaces
to solve arbitrary (1D) Poisson-Nernst-Planck systems, to sample
discrete coordinate sets from continuous distributions, and to
assure steric radii for all coordinate points in a sample.
Type `matscipy-poisson-nernst-planck --help`, `matscipy-continuous2discrete --help`, and 
`matscipy-stericify --help` for usage information.

A simple sample usage to generate a discrete coordinate set from
the continuous solution of Poisson-Nernst-Planck equations for
0.1 mM NaCl aqueous solution across a 100 nm gap and a 0.05 V potential drop
would look like this

    matscipy-poisson-nernst-planck -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.txt
    matscipy-continuous2discrete --verbose NaCl.txt NaCl.lammps

for PNP solution in plain text file and according coordinate samples LAMMPS
data file, or like this

    matscipy-poisson-nernst-planck -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.npz
    matscipy-continuous2discrete --verbose NaCl.npz NaCl.xyz

for PNP solution in binary numpy .npz file and coordinate samples in generic
xyz file, or as a pipeline

    matscipy-poisson-nernst-planck -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose | continuous2discrete --verbose > NaCl.xyz

for text and xyz format streams.

Steric effects at the interface leading to a compact Stern layer can be either
modeled explicitly via enforcing a linear potential regime within the compact
layer region, or implicitly by excluding the compact layer region from the
computation domain and applying Robin boundary conditions. Latter is the default
via command line interface.

    matscipy-poisson-nernst-planck -c 0.1 0.1 -u 0.05 -l 1.0e-7 --lambda-s 5.0e-10 -bc cell-robin --verbose NaCl.npz

In order to impose a steric radius to the coordinates in some data file, use

    matscipy-stericify --verbose -r 2.0 -- NaCl.lammps stericNaCl.lammps
