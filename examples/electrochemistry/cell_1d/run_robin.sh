#!/bin/bash
# solves Poisson-Nernst-Planck system with different Stern layer thicknesses
mkdir -p data_robin
mkdir -p log_robin

#C=( 200 2000 3000 4000 )
c=2000
L=( 0 1e-10 5e-10 10e-10 20e-10 )
for l in ${L[@]}; do
    cmd="pnp --verbose --segments 200 --maxit 40 -l 30e-10 -c ${c} ${c} -z 1 -1 -bc cell-robin -u 0.05 --lambda-s ${l}"
    cmd="${cmd} --log log_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V_lamda_s_${l}_m.log"
    cmd="${cmd} data_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V_lambda_s_${l}_m.txt"
    echo "Run '$cmd'..."
    $cmd
done
