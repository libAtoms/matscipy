#!/bin/bash
# solves Poisson-Nernst-Planck system with different Stern layer thicknesses
mkdir -p data_robin
mkdir -p log_robin

#C=( 200 2000 3000 4000 )
c=1000
L=( 0 "1.0e-10" "2.0e-10" "5.0e-10" "10.0e-10" )
for l in ${L[@]}; do
    cmd="poisson-nernst-planck --verbose --segments 200 --maxit 20 --absolute-tolerance 1.0e-12 -bc cell-robin"
    cmd="${cmd} -l 30.0e-10 -c ${c} ${c} -z 1 -1 --lambda-s ${l} -u 0.05"
    cmd="${cmd} --log log_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V_lamda_s_${l}_m.log"
    cmd="${cmd} data_robin/NaCl_c_${c}_${c}_mM_z_+1_-1_l_30e-10_m_u_0.05_V_lambda_s_${l}_m.txt"
    echo "Run '$cmd'..."
    $cmd
done
