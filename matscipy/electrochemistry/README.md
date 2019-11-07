# Introduction
Samples discrete coordinate sets from arbitrary continuous distributions

## Background
In order to investigate the electrochemical double layer at interfaces, this tool samples discrete coordinate sets from classical continuum solutions to Poisson-Nernst-Planck systems. This reduces to the Poisson-Boltzmann distribution as an analyitc solution for the special case of a binary electrolyte half space. Coordinate sets are stored as .xyz or LAMMPS data files.

![pic](poisson-bolzmann-sketch.png)

### Content
* `continuous2discrete.py`: Sampling and plotting
* `poisson_boltzmann_distribution.py`: generate potential and densities

### Usage
`pnp` (Poisson-Nernst-Planck) and `c2d` (continuous2discrete) executables
offer simple command line interfaces to solve arbitrary (1D)
Poisson-Nernst-Planck systems and to sample discrete coordinate
sets from continuous distributions. Type `pnp --help` and `c2d --help` for
usage information.

A simple sample usage to generate a discrete coordinate set from
the continuous solution of Poisson-Nernst-Planck equations for
0.1 mM NaCl aqueous solution across a 100 nm gap and a 0.05 V potential drop
would look like this

    pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.txt
    c2d --verbose NaCl.txt NaCl.lammps

for PNP solution in plain text file and according coordinate samples LAMMPS
data file, or like this

    pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.npz
    c2d --verbose NaCl.npz NaCl.xyz

for PNP solution in binary numpy .npz file and coordinate samples in generic
xyz file, or as a pipeline

    pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose | c2d --verbose > NaCl.xyz

for text and xyz format streams.
