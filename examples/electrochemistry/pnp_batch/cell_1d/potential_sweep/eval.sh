#!/bin/bash
# args: data directory, figure file name, sweep parameter, parameter unit
python pnp_plot.py data_std fig_std.png u 'u \> (\mathrm{V})'
python pnp_plot.py data_robin fig_robin.png u 'u \> (\mathrm{V})'
