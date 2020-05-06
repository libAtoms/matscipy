#!/bin/bash
# args: data directory, figure file name, sweep parameter, parameter unit
python pnp_plot.py data_std fig_std.png c 'c \> (\mathrm{mM})'
python pnp_plot.py data_robin fig_robin.png lambda_s '\lambda_S \> (\mathrm{\AA})'