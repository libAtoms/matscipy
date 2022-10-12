#
# Copyright 2019-2020 Johannes Hoermann (U. Freiburg)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import os.path, re, sys
import numpy as np
from glob import glob
from cycler import cycler
from itertools import cycle
from itertools import groupby
import matplotlib.pyplot as plt

# Ensure variable is defined
try:
    datadir
except NameError:
    try:
        datadir = sys.argv[1]
    except:
        datadir = 'data'

try:
    figfile
except NameError:
    try:
        figfile = sys.argv[2]
    except:
        figfile = 'fig.png'

try:
    param
except NameError:
    try:
        param = sys.argv[3]
    except:
        param = 'c'

try:
    param_unit
except NameError:
    try:
        param_label = sys.argv[4]
    except:
        param_label = 'c (\mathrm{mM})'

try:
    glob_pattern
except NameError:
    glob_pattern = os.path.join(datadir, 'NaCl*.txt')

def right_align_legend(leg):
    hp = leg._legend_box.get_children()[1]
    for vp in hp.get_children():
        for row in vp.get_children():
            row.set_width(100)  # need to adapt this manually
            row.mode= "expand"
            row.align="right"

# sort file names as normal humans expect
# https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
def alpha_num_order(x):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    return [ convert(c) for c in re.split('([0-9]+)', x) ]


dat_files = sorted(glob(glob_pattern),key=alpha_num_order)
N = len(dat_files) # number of data sets
M = 2 # number of species

# matplotlib settings
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

# plt.rc('axes', prop_cycle=default_cycler)

plt.rc('font',   size=MEDIUM_SIZE)       # controls default text sizes
plt.rc('axes',   titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes',   labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick',  labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick',  labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex

plt.rcParams["figure.figsize"] = (16,10) # the standard figure size

plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.markeredgewidth"]=1

# line styles
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
# linestyle_str = [
#     ('solid', 'solid'),      # Same as (0, ()) or '-'
#     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
#     ('dashed', 'dashed'),    # Same as '--'
#     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# color maps for potential and concentration plots
cmap_u = plt.get_cmap('Reds')
cmap_c = [plt.get_cmap('Oranges'), plt.get_cmap('Blues')]

# general line style cycler
line_cycler =   cycler( linestyle = [ s for _,s in linestyle_tuple ] )

# potential anc concentration cyclers
u_cycler    =   cycler( color = cmap_u( np.linspace(0.4,0.8,N) ) )
u_cycler    =   len(line_cycler)*u_cycler + len(u_cycler)*line_cycler
c_cyclers   = [ cycler( color =   cmap( np.linspace(0.4,0.8,N) ) ) for cmap in cmap_c ]
c_cyclers   = [ len(line_cycler)*c_cycler + len(c_cycler)*line_cycler for c_cycler in c_cyclers ]

# https://matplotlib.org/3.1.1/tutorials/intermediate/constrainedlayout_guide.html
fig, (ax1,ax2,ax3) = plt.subplots(
    nrows=1, ncols=3, figsize=[24,7], constrained_layout=True)

ax1.set_xlabel('z (nm)')
ax1.set_ylabel('potential (V)')
ax2.set_xlabel('z (nm)')
ax2.set_ylabel('concentration (mM)')
ax3.set_xlabel('z (nm)')
ax3.set_ylabel('concentration (mM)')

# ax1.axvline(x=pnp.lambda_D()*1e9, label='Debye Length', color='grey', linestyle=':')
species_label = [
    '$[\mathrm{Na}^+], ' + param_label + '$',
    '$[\mathrm{Cl}^-], ' + param_label + '$']

c_regex = re.compile(r'{}_(-?\d+(,\d+)*(\.\d+(e\d+)?)?)'.format(param))

c_graph_handles = [ [] for _ in range(M) ]
for f, u_style, c_styles in zip(dat_files,u_cycler,zip(*c_cyclers)):
    print("Processing {:s}".format(f))
    # extract nominal concentration from file name
    nominal_c = float( c_regex.search(f).group(1) )


    dat = np.loadtxt(f,unpack=True)
    x = dat[0,:]
    u = dat[1,:]
    c = dat[2:,:]

    c_label = '{:> 4.1f}'.format(nominal_c)
    # potential
    ax1.plot(x*1e9, u, marker=None, label=c_label, linewidth=1, **u_style)

    for i in range(c.shape[0]):
        # concentration
        ax2.plot(x*1e9, c[i], marker='',
            label=c_label, linewidth=2, **c_styles[i])
        # semilog concentration
        c_graph_handles[i].extend( ax3.semilogy(x*1e9, c[i], marker='',
            label=c_label, linewidth=2, **c_styles[i]) )

# legend placement
# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
u_legend = ax1.legend(loc='center right', title='potential, ${}$'.format(param_label), bbox_to_anchor=(-0.2,0.5) )
first_c_legend  = ax3.legend(handles=c_graph_handles[0], title=species_label[0], loc='upper left', bbox_to_anchor=(1.00, 1.02) )
second_c_legend = ax3.legend(handles=c_graph_handles[1], title=species_label[1], loc='lower left', bbox_to_anchor=(1.00,-0.02) )
ax3.add_artist(first_c_legend) # add automatically removed first legend again
c_legends = [ first_c_legend, second_c_legend ]
legends = [ u_legend, *c_legends ]

for l in legends:
    right_align_legend(l)

# https://matplotlib.org/3.1.1/tutorials/intermediate/constrainedlayout_guide.html
for l in legends:
    l.set_in_layout(False)
# trigger a draw so that constrained_layout is executed once
# before we turn it off when printing....
fig.canvas.draw()
# we want the legend included in the bbox_inches='tight' calcs.
for l in legends:
    l.set_in_layout(True)
# we don't want the layout to change at this point.
fig.set_constrained_layout(False)

# fig.tight_layout(pad=3.0, w_pad=2.0, h_pad=1.0)
# plt.show()
fig.savefig(figfile, bbox_inches='tight', dpi=100)