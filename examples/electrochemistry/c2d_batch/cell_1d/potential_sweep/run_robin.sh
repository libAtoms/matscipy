#!/bin/bash
# solves Poisson-Nernst-Planck system for different concentrations
srcdir=../../../pnp_batch/cell_1d/potential_sweep/data_robin
mkdir -p data_robin
mkdir -p log_robin
for f in ${srcdir}/*.txt; do
    n=$(basename $f)
    b=${n%.txt}
    cmd="continuous2discrete --verbose --names Na Cl --charges 1 -1 --box 28e-9 28e-9 20e-9"
    cmd="${cmd} --mol-id-offset 0 0 --log log_robin/${b}.log ${f} data_robin/${b}.lammps"
    echo "Run '$cmd'..."
    $cmd
done
