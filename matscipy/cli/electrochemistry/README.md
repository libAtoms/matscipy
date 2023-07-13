### Content
* `pnp.py`: Command line interface to `poisson_boltzmann_distribution.py`
* `c2d.py`: Command line interface to `continuous2discrete.py`
* `stericify.py`: Command line interface to `steric_correction.py`

### Usage
`pnp.py` (Poisson-Nernst-Planck), `c2d.py` (continuous2discrete) and
`stericiy.py` executable scripts offer simple command line interfaces
to solve arbitrary (1D) Poisson-Nernst-Planck systems, to sample
discrete coordinate sets from continuous distributions, and to
assure steric radii for all coordinate points in a sample.
Type `pnp.py --help`, `c2d.py --help`, and `stericiy.py --help`
for usage information.

A simple sample usage to generate a discrete coordinate set from
the continuous solution of Poisson-Nernst-Planck equations for
0.1 mM NaCl aqueous solution across a 100 nm gap and a 0.05 V potential drop
would look like this

    pnp.py -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.txt
    c2d.py --verbose NaCl.txt NaCl.lammps

for PNP solution in plain text file and according coordinate samples LAMMPS
data file, or like this

    pnp.py -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.npz
    c2d.py --verbose NaCl.npz NaCl.xyz

for PNP solution in binary numpy .npz file and coordinate samples in generic
xyz file, or as a pipeline

    pnp.py -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose | c2d.py --verbose > NaCl.xyz

for text and xyz format streams.

Steric effects at the interface leading to a compact Stern layer can be either
modeled explicitly via enforcing a linear potential regime within the compact
layer region, or implicitly by excluding the compact layer region from the
computation domain and applying Robin boundary conditions. Latter is the default
via command line interface.  

    pnp.py -c 0.1 0.1 -u 0.05 -l 1.0e-7 --lambda-s 5.0e-10 -bc cell-robin --verbose NaCl.npz

In order to impose a steric radius to the coordinates in some data file, use

    stericify.py --verbose -r 2.0 -- NaCl.lammps stericNaCl.lammps

Headers (docstrings) of scripts contain more examples.
