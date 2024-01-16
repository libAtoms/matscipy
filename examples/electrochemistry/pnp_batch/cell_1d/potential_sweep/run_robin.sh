#!/bin/bash
# solves Poisson-Nernst-Planck system for different concentrations
mkdir -p data_robin
mkdir -p log_robin
U=( 0.01 0.05 0.1 0.5 0.8 1 1.2 ) # V
c=1000 # mM
lambda_S=2e-10 # 2 Ang
for u in ${U[@]}; do
    cmd="poisson-nernst-planck --verbose -l 20.0e-9 -c ${c} ${c} -z 1 -1 -bc cell-robin -u ${u} --lambda-s ${lambda_S}"
    cmd="${cmd} --log log_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_20e-9_m_u_${u}_V_lambda_S_${lambda_S}.log"
    cmd="${cmd} data_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_20e-9_m_u_${u}_V_lambda_S_${lambda_S}.txt"
    echo "Run '$cmd'..."
    $cmd
done
