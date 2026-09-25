# Diffusion

## `matscipy-rms`

The `matscipy-rms` utility script computes the rms displacements of all atoms in a NetCDF trajectory file. It
gracefully handles jumps of atoms through periodic boundaries by applying a minimum image convention to each
displacement step.

Example of use:

```bash
matscipy-rms traj.nc
```
