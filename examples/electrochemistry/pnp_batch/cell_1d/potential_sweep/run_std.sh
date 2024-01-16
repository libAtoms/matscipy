#!/bin/bash
# solves Poisson-Nernst-Planck system for different concentrations
mkdir -p data_std
mkdir -p log_std
U=( 0.01 0.05 0.1 0.5 0.8 1 1.2 ) # V
c=1000 # mM
for u in ${U[@]}; do
    cmd="poisson-nernst-planck --verbose -l 20.0e-9 -c ${c} ${c} -z 1 -1 -bc cell -u ${u}"
    cmd="${cmd} --log log_std/NaCl_c_${c}_${c}_mM_z_+1_-1_l_20e-9_m_u_${u}_V.log"
    cmd="${cmd} data_std/NaCl_c_${c}_${c}_mM_z_+1_-1_l_20e-9_m_u_${u}_V.txt"
    echo "Run '$cmd'..."
    $cmd
done
