# Introduction
Samples discrete coordinate sets from arbitrary continuous distributions

# Background
In order to investigate the electrochemical double layer at interfaces, these
tool samples discrete coordinate sets from classical continuum solutions to
Poisson-Nernst-Planck systems. This reduces to the Poisson-Boltzmann
distribution as an analyitc solution for the special case of a binary
electrolyte half space. Coordinate sets are stored as .xyz or LAMMPS data files.

![pic](poisson-bolzmann-sketch.png)

# Content
* `continuous2discrete.py`: sampling and plotting
* `poisson_boltzmann_distribution.py`: generate potential and density
  distributions by solving full Poisson-Nernst-Planck systems.

# Usage
See `../../scripts/electrochemistry/README.md`.
