#!/bin/bash
# applies steric correction for systems at different potential
srcdir=../../../c2d_batch/cell_1d/potential_sweep/data_std
mkdir -p data_std
mkdir -p log_std
for f in ${srcdir}/*.lammps; do
    n=$(basename $f)
    b=${n%.lammps}
    cmd='stericify --verbose --names Na Cl --radii 1.5'
    cmd="${cmd} --log log_std/${b}.log ${f} data_std/${b}.lammps"
    echo "Run '$cmd'..."
    $cmd
    # echo $f, $n, $b
done
