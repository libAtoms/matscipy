### Content
* `c2d.py`: Command line interface to `continuous2discrete.py`
* `pnp.py`: Command line interface to `poisson_boltzmann_distribution.py`

The following assumes you have made these scripts available for execution
under the names `c2d` and `pnp`, i.e. by adding this directory to your `PATH`
or creating suitable symbolic links somewhere else.

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

Steric effects at the interface leading to a compact Stern layer can be either
modeled explicitly via enforcing a linear potential regime within the compact
layer region, or implicitly by excluding the compact layer region from the
computation domain and applying Robin boundary conditions. Latter is the default
via command line interface.  

    pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 --lambda-s 5.0e-10 -bc cell-robin --verbose NaCl.npz
