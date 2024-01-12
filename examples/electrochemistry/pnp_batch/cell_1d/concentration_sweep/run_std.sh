#!/bin/bash
# solves Poisson-Nernst-Planck system for different concentrations
mkdir -p data_std
mkdir -p log_std
C=( 0.1 1 10 100 500 1000 2000 )
for c in ${C[@]}; do
    cmd="poisson-nernst-planck --verbose -l 30.0e-10 -c $c $c -z 1 -1 -bc cell -u 0.05"
    cmd="${cmd} --log log_std/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V.log"
    cmd="${cmd} data_std/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V.txt"
    echo "Run '$cmd'..."
    $cmd
done
