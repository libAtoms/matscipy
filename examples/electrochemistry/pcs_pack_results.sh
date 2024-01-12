#!/bin/bash

# Sample on parameter sweep through the poisson-nernst-planck > continuous2discrete > stericify pipeline.
# see sub-directories for detailed understanding.

system=cell_1d
parameter=concentration
steps=( pnp c2d stericify )
archive=pcs_${system}_${parameter}_sweep_results.tar

(

echo "### create ${archive} ###"
tar -cvf ${archive} pcs_pipeline.*
echo ""

for step in ${steps[@]}; do
  echo "### pack ${step} results ###"
  tar -rvf ${archive} ${step}_batch/${system}/${parameter}_sweep
  echo ""
done

echo "### pack myself ###"
# mock files added later into log
ls -1 pcs_pack_results.*

echo ""
echo "### ALL DONE ###"

) 2>&1 | tee pcs_pack_results.log

tar -rvf ${archive} pcs_pack_results.*
