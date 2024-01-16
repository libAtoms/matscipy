# Fracture Mechanics Examples

This directory contains `params.py` parameters for the `matscipy-quasisstatic-fracture` and `matscipy-sinclair-crack` CLI tools and for some other ancillary scripts, as follows:

Usage is to change the current working directory to that containing the `params.py` file then
to invoke the corresponding command line tool or script. In some cases multiple parameter sets
are provided, e.g. for different physical systems. The desired one should be copied elsewhere
and renamed to `params.py` before invoking the relveant script.

- `quasistatic_fracture` - example input for the `matscipy-quasistatic-fracture` CLI tool
- `sinclair-crack` - example input for the `matscipy-sinclair-crack` CLI tool
- `ideal_brittle_solid/` - example for `staging/fracture_mechanics/run_ideal_brittle_solid.py` which models fracture in an ideal harmonic solid (following the approach of D. Holland and M. P. Marder, Cracks and Atoms, Advanced Materials.)
- `make_crack_thin_strip` - example for `staging/fracture_mechanics/make_crack_thin_strip.py`, which constructs a fracture simulation cell with 'thin strip' boundary conditions. The same parameter files serves as input for the `staging/fracture_mechanics/run_crack_thin_strip` script which runs MD in this geometry.
- `quartz_crack` - example input for `staging/fracture_mechanics/quartz_crack.py` which models fracture of quartz
