#
# Copyright 2020 Johannes Hoermann (U. Freiburg)
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

# %% [markdown]
# # Steric correction
#
# *Johannes Hörmann, 2020*
#
# Impose steric radii on a sample point distribution by minizing pseudo-potential.
# Pseudo-ptential follows formalism described in
#
# *L. Martinez, R. Andrade, E. G. Birgin, and J. M. Martínez, “PACKMOL: A package for building initial configurations for molecular dynamics simulations,” J. Comput. Chem., vol. 30, no. 13, pp. 2157–2164, 2009, doi: 10/chm6f7.*
#

# %%
# for dynamic module reload during testing, code modifications take immediate effect
# %load_ext autoreload
# %autoreload 2

# %%
# stretching notebook width across whole window
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
# basics & utilities
import itertools                # dealing with combinations and permutations
import logging                  # easy and neat handlin of log outpu
import matplotlib.pyplot as plt # plotting
import numpy as np              # basic numerics, in particular lin. alg.
import pandas as pd             # display stats neatly
import scipy.constants as sc    # fundamental constants
import scipy.spatial.distance   # quick methods for computing pairwise distances
import time                     # timing performance
import timeit                   # same purpose

# %%
# electrochemistry basics
from matscipy.electrochemistry import debye, ionic_strength

# %%
# Poisson-Bolzmann distribution
from matscipy.electrochemistry.poisson_boltzmann_distribution import gamma, potential, concentration, charge_density

# %%
# sampling
from scipy import interpolate
from matscipy.electrochemistry import continuous2discrete
from matscipy.electrochemistry import get_histogram
from matscipy.electrochemistry.utility import plot_dist

# %%
# steric correction

# target functions
from matscipy.electrochemistry.steric_correction import scipy_distance_based_target_function
from matscipy.electrochemistry.steric_correction import numpy_only_target_function
from matscipy.electrochemistry.steric_correction import brute_force_target_function

# closest pair functions
from matscipy.electrochemistry.steric_correction import brute_force_closest_pair
from matscipy.electrochemistry.steric_correction import planar_closest_pair
from matscipy.electrochemistry.steric_correction import scipy_distance_based_closest_pair

from matscipy.electrochemistry.steric_correction import apply_steric_correction

# %%
# 3rd party file output
import ase
import ase.io

# %%
# matscipy.electrochemistry makes extensive use of Python's logging module

# configure logging: verbosity level and format as desired
standard_loglevel   = logging.INFO
# standard_logformat  = ''.join(("%(asctime)s",
#  "[ %(filename)s:%(lineno)s - %(funcName)s() ]: %(message)s"))
standard_logformat  = "[ %(filename)s:%(lineno)s - %(funcName)s() ]: %(message)s"

# reset logger if previously loaded
logging.shutdown()
logging.basicConfig(level=standard_loglevel,
                    format=standard_logformat,
                    datefmt='%m-%d %H:%M')

# in Jupyter notebooks, explicitly modifying the root logger necessary
logger = logging.getLogger()
logger.setLevel(standard_loglevel)

# remove all handlers
for h in logger.handlers: logger.removeHandler(h)

# create and append custom handles
ch = logging.StreamHandler()
formatter = logging.Formatter(standard_logformat)
ch.setFormatter(formatter)
ch.setLevel(standard_loglevel)
logger.addHandler(ch)

# %%
# Test 1
logging.info("Root logger")

# %%
# Test 2
logger.info("Root Logger")

# %%
# Debug Test
logging.debug("Root logger")

# %% [markdown]
# ## Step 1: Solve for continuous concentration distributions
# See other sample case notebboks for details
#

# %%
# measures of box
xsize = ysize = 5e-9 # nm, SI units
zsize = 20e-9         # nm, SI units

# get continuum distribution, z direction
x = np.linspace(0, zsize, 2000)
c = [1,1]
z = [1,-1]
u = 0.05 

phi = potential(x, c, z, u)
C   = concentration(x, c, z, u)
rho = charge_density(x, c, z, u)


# %%
# potential and concentration distributions analytic solution 
# based on Poisson-Boltzmann equation for 0.1 mM NaCl aqueous solution 
# at interface 
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        

deb = debye(c, z) 

fig, ax1 = plt.subplots(figsize=[18,5])
ax1.set_xlabel('x (nm)')
ax1.plot(x/sc.nano, phi, marker='', color='red', label='Potential', linewidth=1, linestyle='--')
ax1.set_ylabel('potential (V)')
ax1.axvline(x=deb/sc.nano, label='Debye Length', color='orange')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='Bulk concentration of Na+ ions', color='grey', linewidth=1, linestyle=':')
ax2.plot(x/sc.nano, C[0], marker='', color='green', label='Na+ ions')
ax2.plot(x/sc.nano, C[1], marker='', color='blue', label='Cl- ions')
ax2.set_ylabel('concentration (mM)')

ax3 = ax1.twinx()
# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)

ax3.plot(x/sc.nano, rho, label='Charge density', color='grey', linewidth=1, linestyle='--')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')

#fig.legend(loc='center')
ax2.legend(loc='upper right', bbox_to_anchor=(-0.1, 1.02),fontsize=15)
ax1.legend(loc='center right', bbox_to_anchor=(-0.1,0.5), fontsize=15)
ax3.legend(loc='lower right', bbox_to_anchor=(-0.1, -0.02), fontsize=15)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2: Sample from distribution

# %%
# create distribution functions
distributions = [interpolate.interp1d(x,c) for c in C]

# sample discrete coordinate set
box3 = np.array([xsize, ysize, zsize])
sample_size = 200

# %%
samples = [ continuous2discrete(distribution=d, box=box3, count=sample_size) for d in distributions ]
species = ['Na+','Cl-']
for ion,sample,d in zip(species,samples,distributions):
    histx, histy, histz = get_histogram(sample, box=box3, n_bins=51)
    plot_dist(histx, 'Distribution of {:s} ions in x-direction'.format(ion), 
              reference_distribution=lambda x: np.ones(x.shape)*1/box3[0])
    plot_dist(histy, 'Distribution of {:s} ions in y-direction'.format(ion), 
              reference_distribution=lambda x: np.ones(x.shape)*1/box3[1])
    plot_dist(histz, 'Distribution of {:s} ions in z-direction'.format(ion), 
              reference_distribution=d)

# %% [markdown]
# ## Step 3: Enforce steric radii

# %% [markdown]
# Initial state of system:

# %%
# need all coordinates in one N x 3 array
xstacked = np.vstack(samples)

box6 = np.array([[0.,0.,0],box3]) # needs lower corner

n = xstacked.shape[0]
dim = xstacked.shape[1]

# benchmakr methods
mindsq, (p1,p2) = scipy_distance_based_closest_pair(xstacked)
pmin = np.min(xstacked,axis=0)
pmax = np.max(xstacked,axis=0)
mind = np.sqrt(mindsq)
logger.info("Minimum pair-wise distance in sample: {}".format(mind))
logger.info("First sample point in pair:    ({:8.4e},{:8.4e},{:8.4e})".format(*p1))
logger.info("Second sample point in pair    ({:8.4e},{:8.4e},{:8.4e})".format(*p2))
logger.info("Box lower boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[0]))
logger.info("Minimum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmin))
logger.info("Maximum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmax))
logger.info("Box upper boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[1]))

# %%
# apply penalty for steric overlap

# stats: method, x, res, dt, mind, p1, p2 , pmin, pmax
stats = [('initial',xstacked,None,0,mind,p1,p2,pmin,pmax)]

r = 2e-10 # 4 Angstrom steric radius
logger.info("Steric radius: {:8.4e}".format(r))

# see https://scipy-lectures.org/advanced/mathematical_optimization/index.html
methods = [
    #'Nelder-Mead', # not suitable
    'Powell',
    'CG',
    'BFGS',
    #'Newton-CG', # needs explicit Jacobian
    'L-BFGS-B' 
]
        
for m in methods:
    try:
        logger.info("### {} ###".format(m))
        t0 = time.perf_counter()
        x1, res = apply_steric_correction(xstacked,box=box6,r=r,method=m)
        t1 = time.perf_counter()
        dt = t1 - t0
        logger.info("{} s runtime".format(dt))
        
        mindsq, (p1,p2) = scipy_distance_based_closest_pair(x1)
        mind = np.sqrt(mindsq)
        pmin = np.min(x1,axis=0)
        pmax = np.max(x1,axis=0)

        stats.append([m,x1,res,dt,mind,p1,p2,pmin,pmax])
        
        logger.info("{:s} finished with".format(m))
        logger.info("    status = {}, success = {}, #it = {}".format(
            res.status, res.success, res.nit))
        logger.info("    message = '{}'".format(res.message))
        logger.info("Minimum pair-wise distance in final configuration: {:8.4e}".format(mind))
        logger.info("First sample point in pair:    ({:8.4e},{:8.4e},{:8.4e})".format(*p1))
        logger.info("Second sample point in pair    ({:8.4e},{:8.4e},{:8.4e})".format(*p2))
        logger.info("Box lower boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[0]))
        logger.info("Minimum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmin))
        logger.info("Maximum coordinates in sample: ({:8.4e},{:8.4e},{:8.4e})".format(*pmax))
        logger.info("Box upper boundary:            ({:8.4e},{:8.4e},{:8.4e})".format(*box6[1]))
    except:
        logger.warning("{} failed.".format(m))
        continue
        
stats_df = pd.DataFrame( [ { 
    'method':  s[0], 
    'runtime': s[3],
    'mind':    s[4],
    **{'p1{:d}'.format(i): c for i,c in enumerate(s[5]) },
    **{'p2{:d}'.format(i): c for i,c in enumerate(s[6]) }, 
    **{'pmin{:d}'.format(i): c for i,c in enumerate(s[7]) },
    **{'pmax{:d}'.format(i): c for i,c in enumerate(s[8]) }
} for s in stats] )

print(stats_df.to_string(float_format='%8.6g'))

# %% [markdown]
# L-BFGS-B fastest.

# %%
# Check difference between initial and final configuration, use last result (L-BFGS-B)
np.count_nonzero(xstacked - x1) # that many coordinates modified

# %%
# Check difference between initial and final configuration, use last result (L-BFGS-B)
np.linalg.norm(xstacked - x1) # euclidean distance between two sets

# %% [markdown]
# ## Step 4: Visualize results

# %%
# pick last result and split by species
steric_samples = [ x1[:sample_size,:], x1[sample_size:,:] ]

# %%
nbins = 101
for ion,sample,d in zip(species,steric_samples,distributions):
    histx, histy, histz = get_histogram(sample, box=box3, n_bins=nbins)
    plot_dist(histx, 'Distribution of {:s} ions in x-direction'.format(ion), 
              reference_distribution=lambda x: np.ones(x.shape)*1/box3[0])
    plot_dist(histy, 'Distribution of {:s} ions in y-direction'.format(ion), 
              reference_distribution=lambda x: np.ones(x.shape)*1/box3[1])
    plot_dist(histz, 'Distribution of {:s} ions in z-direction'.format(ion), 
              reference_distribution=d)

# %%
# Distribution of corrections
for ion,sample,steric_sample,d in zip(species,samples,steric_samples,distributions):
    hists = get_histogram(sample, box=box3, n_bins=nbins)    
    steric_hists = get_histogram(steric_sample, box=box3, n_bins=nbins)    
    # first entry is counts, second entry is bins
    diff_hists = [ (h[0] - hs[0], h[1]) for h,hs in zip(hists,steric_hists) ]
    for ax, h in zip( ['x','y','z'], diff_hists ):
        plot_dist(h, 'Difference from non-steric to steric {:s} ion sample in {:s}-direction'.format(ion, ax))

# %% [markdown]
# ## Step 5: Write to file
# We utilize ASE to export it to some standard format, i.e. LAMMPS data file.
# ASE speaks Ångström per default, thus we convert SI units:

# %%
symbols = ['Na','Cl']

# %%
system = ase.Atoms(
    cell=np.diag(box3/sc.angstrom),
    pbc=[True,True,False]) 
for symbol, sample, charge in zip(symbols,samples,z):
    system += ase.Atoms(
        symbols=symbol*sample_size,
        charges=[charge]*sample_size,
        positions=sample/sc.angstrom)
system

# %%
ase.io.write('NaCl_200_0.05V_5x5x20nm_at_interface_pb_distributed.lammps',system,format='lammps-data',units="real",atom_style='full')

# %%
steric_system = ase.Atoms(
    cell=np.diag(box3/sc.angstrom),
    pbc=[True,True,False]) 
for symbol, sample, charge in zip(symbols,steric_samples,z):
    steric_system += ase.Atoms(
        symbols=symbol*sample_size,
        charges=[charge]*sample_size,
        positions=sample/sc.angstrom)
steric_system

# %%
ase.io.write('NaCl_200_0.05V_5x5x20nm_at_interface_pb_distributed_steric_correction_2Ang.lammps',steric_system,format='lammps-data',units="real",atom_style='full')

# %% [markdown]
# Displacement visualization between non-steric and steric sample with Ovito:

# %% [markdown]
# ![Steric correction on 200 NaCl](steric_correction_on_200_NaCl_300px.png)

# %% [markdown]
# ## Other performance tests

# %% [markdown]
# ### Comparing target function implementations

# %%
# prepare coordinates and get system dimensions
xstacked = np.vstack(samples)

n   = xstacked.shape[0]
dim = xstacked.shape[1]

# normalize volume and coordinates
V = np.product(box6[1]-box6[0])
L = np.power(V,(1./dim))

x0 = xstacked / L

funcs = [
        brute_force_target_function,
        numpy_only_target_function,
        scipy_distance_based_target_function ]
func_names = ['brute','numpy','scipy']

# test for different scalings of coordinate set:
stats = []
K = np.exp(np.log(10)*np.arange(-3,3))
for k in K:
    lambdas = [ (lambda x0=xstacked,k=k,f=f: f(x0*k)) for f in funcs ]
    vals    = [ f() for f in lambdas ]
    times   = [ timeit.timeit(f,number=1) for f in lambdas ]
    diffs = scipy.spatial.distance.pdist(np.atleast_2d(vals).T,metric='euclidean')
    stats.append((k,*vals,*diffs,*times))

func_name_tuples = list(itertools.combinations(func_names,2))
diff_names = [ 'd_{:s}_{:s}'.format(f1,f2) for (f1,f2) in func_name_tuples ]
perf_names = [ 't_{:s}'.format(f) for f in func_names ]
fields =  ['k',*func_names,*diff_names,*perf_names]
dtypes = [ (field, '>f4') for field in fields ]
labeled_stats = np.array(stats,dtype=dtypes) 
stats_df = pd.DataFrame(labeled_stats)
print(stats_df.to_string(float_format='%8.6g'))

# %% [markdown]
# Scipy-based target function fastest.

# %% [markdown]
# ### Comparing closest pair implementations
# See https://www.researchgate.net/publication/266617010_NumPy_SciPy_Recipes_for_Data_Science_Squared_Euclidean_Distance_Matrices

# %%
# test minimum distance function implementations on random samples

N = 1000
dim = 3

funcs = [
        brute_force_closest_pair,
        scipy_distance_based_closest_pair,
        planar_closest_pair ]
func_names = ['brute','scipy','planar']
stats = []

for k in range(5):
    x = np.random.rand(N,dim)
    lambdas = [ (lambda x=x,f=f: f(x)) for f in funcs ]
    rets    = [ f() for f in lambdas ]
    vals    = [ v[0] for v in rets ]
    coords  = [ c for v in rets for p in v[1] for c in p ]
    times   = [ timeit.timeit(f,number=1) for f in lambdas ]
    diffs   = scipy.spatial.distance.pdist(
        np.atleast_2d(vals).T,metric='euclidean')
    stats.append((*vals,*diffs,*times,*coords))

func_name_tuples = list(itertools.combinations(func_names,2))
diff_names =  [ 'd_{:s}_{:s}'.format(f1,f2) for (f1,f2) in func_name_tuples ]
perf_names =  [ 't_{:s}'.format(f) for f in func_names ]
coord_names = [ 
    'p{:d}{:s}_{:s}'.format(i,a,f) for f in func_names for i in (1,2) for a in ('x','y','z') ]
float_fields = [*func_names,*diff_names,*perf_names,*coord_names]
dtypes = [ (field, 'f4') for field in float_fields ]
labeled_stats = np.array(stats,dtype=dtypes)
stats_df = pd.DataFrame(labeled_stats)
print(stats_df.T.to_string(float_format='%8.6g'))

# %% [markdown]
# Scipy-based implementation fastest.

# %%
print('{}'.format(system.symbols))

# %%
system.cell.array

# %%
np.array(system.get_cell_lengths_and_angles())
