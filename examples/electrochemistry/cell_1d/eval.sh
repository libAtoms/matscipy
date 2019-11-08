#!/bin/bash
# args: data directory, figure file name
python pnp_plot.py data_std fig_std.png
python pnp_plot.py data_robin fig_robin.png

mkdir -p data_cmp
cp data_std/*c_2000* data_cmp
cp data_robin/*c_2000* data_cmp

python pnp_plot.py data_cmp fig_cmp.png