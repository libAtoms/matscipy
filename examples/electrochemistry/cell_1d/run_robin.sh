#!/bin/bash
# solves Poisson-Nernst-Planck system for different concentrations
mkdir -p data_robin
mkdir -p log_robin

C=( 2000 3000 4000 5000 )
for c in ${C[@]}; do
    cmd="pnp --verbose -l 30.0e-10 -c $c $c -z 1 -1 -bc cell-robin -u 0.05 --lambda-s 2.0e-10"
    cmd="${cmd} --log log_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V_lamda_s_2e-10_m.log"
    cmd="${cmd} data_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V_lambda_s_2e-10_m.txt"
    echo "Run '$cmd'..."
    $cmd
done
