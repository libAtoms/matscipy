#!/bin/bash
# applies steric correction for systems at different potential
srcdir=../../../c2d_batch/cell_1d/potential_sweep/data_robin
mkdir -p data_robin
mkdir -p log_robin
for f in ${srcdir}/*.txt; do
    n=$(basename $f)
    b=${n%.lammps}
    cmd='stericify --verbose --names Na Cl --radii 1.5'
    cmd="${cmd} --log log_robin/${b}.log ${f} data_robin/${b}.lammps"
    echo "Run '$cmd'..."
    $cmd
done
