#!/bin/bash
# samples discrete coordinate sets from continuous distributions for different concentrations
srcdir=../../../pnp_batch/cell_1d/concentration_sweep/data_std
mkdir -p data_std
mkdir -p log_std
for f in ${srcdir}/*.txt; do
    n=$(basename $f)
    b=${n%.txt}
    cmd="continuous2discrete --verbose --names Na Cl --charges 1 -1 --box 28e-9 28e-9 3e-9"
    cmd="${cmd} --mol-id-offset 0 0 --log log_std/${b}.log ${f} data_std/${b}.lammps"
    echo "Run '$cmd'..."
    $cmd
    # echo $f, $n, $b
done
