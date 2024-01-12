#!/bin/bash

# Sample on parameter sweep through the poisson-nernst-planck > continuous2discrete > stericify pipeline.
# see sub-directories for detailed understanding.

system=cell_1d
parameter=concentration
steps=( pnp c2d stericify )
prefix=$(pwd)

for step in ${steps[@]}; do
  echo "### ${step} ###"
  (cd ${prefix}/${step}_batch/${system}/${parameter}_sweep; bash clean.sh; bash run.sh)
  echo "### ${step} DONE ###"
  echo ""
done

echo "### ALL DONE ###"
