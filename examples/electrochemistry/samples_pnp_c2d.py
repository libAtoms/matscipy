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

# %% [markdown]
# # Poisson-Nernst-Planck systems & continuous2discrete 
#
# *Johannes Hörmann, Lukas Elflein, 2019*
#
# from continuous electrochemical double layer theory to discrete coordinate sets

# %%
# for dynamic module reload during testing, code modifications take immediate effect
# %load_ext autoreload
# %autoreload 2

# %%
# stretching notebook width across whole window
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
# basics
import logging
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

# %%
# sampling
from scipy import interpolate
from matscipy.electrochemistry import continuous2discrete
from matscipy.electrochemistry import get_histogram
from matscipy.electrochemistry.utility import plot_dist

# %%
# electrochemistry basics
from matscipy.electrochemistry import debye, ionic_strength

# %%
# Poisson-Bolzmann distribution
from matscipy.electrochemistry.poisson_boltzmann_distribution import gamma, potential, concentration, charge_density

# %%
# Poisson-Nernst-Planck solver
from matscipy.electrochemistry import PoissonNernstPlanckSystem

# %%
# 3rd party file output
import ase
import ase.io

# %%
# PoissonNernstPlanckSystem makes extensive use of Python's logging module

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


# %%
# tiny helper for plotting
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# %% [markdown]
# # General Poisson-Nernst-Planck System

# %% [markdown]
# For general systems, i.e. a nanogap between two electrodes with not necessarily binary electrolyte, no closed analytic solution exists.
# Thus, we solve the full Poisson-Nernst-Planck system of equations. 

# %% [markdown]
# A binary Poisson-Nernst-Planck system corresponds to the transport problem in semiconductor physics.
# In this context, Debye length, charge carrier densities and potential are related as follows.

# %% [markdown]
# ## Excursus: Transport problem in PNP junction (German)

# %% [markdown]
# ### Debye length

# %% [markdown]
# Woher kommt die Debye-Länge
#
# $$ \lambda = \sqrt{ \frac{\varepsilon \varepsilon_0 k_B T}{q^2 n_i} }$$
#
# als natürliche Längeneinheit des Transportptoblems?
#
# Hier ist $n_i$ eine Referenzladungsträgerdichte, in der Regel die intrinsische Ladungsträgerdichte. 
# In dem Beispiel mit $N^+NN^+$-dotiertem Halbleiter erzeugen wir durch unterschiedliches Doping an den Rändern die erhöhte Donatorendichte $N_D^+ = 10^{20} \mathrm{cm}^{-3}$ und im mitteleren Bereich "Standarddonatorendichte" $N_D = 10^{18} \mathrm{cm}^{-3}$. Nun können wir als Referenz $n_i = N_D$ wählen und die Donatorendichten als $N_D = 1 \cdot n_i$ und $N_D^+ = 100 \cdot n_i$ ausdrücken. Diese normierte Konzentration nennen wir einfach $\tilde{N}_D$: $N_D = \tilde{N}_D \cdot n_i$.
#
# Ein ionisierter Donator trägt die Ladung $q$, ein Ladungsträger (in unserem Fall ein Elektron) trägt die Elementarladung $-q$. Die Raumladungsdichte $\rho$ in der Poissongleichung
#
# $$ \nabla^2 \varphi = - \frac{\rho}{\varepsilon \varepsilon_0}$$
#
# lässt sich also ganz einfach als $\rho = - (n - N_D) \cdot q = - (\tilde{n} - \tilde{N}_D) ~ n_i ~ q$ ausdrücken. 
#
# Konventionell wird das Potential auf $u = \frac{\phi ~ q}{k_B ~ T}$ normiert. Die Poissongleichung nimmt damit die Form
#
# $$\frac{k_B ~ T}{q} \cdot \nabla^2 u = \frac{(\tilde{n} - \tilde{N}_D) ~ n_i ~ q }{\varepsilon \varepsilon_0}$$
#
# oder auch 
#
# $$ \frac{\varepsilon ~ \varepsilon_0 ~ k_B ~ T}{q^2 n_i} \cdot \nabla^2 u = \lambda^2 \cdot \nabla^2 u = \tilde{n} - \tilde{N}_D$$
#
#

# %% [markdown]
# ### Dimensionless formulation

# %% [markdown]
# Poisson- und Drift-Diffusionsgleichung
#
# $$ 
# \lambda^2 \frac{\partial^2 u}{\partial x^2} = n - N_D
# $$
#
# $$ 
# \frac{\partial n}{\partial t} = - D_n \ \frac{\partial}{\partial x} \left( n \ \frac{\partial u}{\partial x} - \frac{\partial n}{\partial x} \right) + R
# $$
#
# Skaliert mit [l], [t]:
#
# $$ 
# \frac{\lambda^2}{[l]^2} \frac{\partial^2 u}{\partial \tilde{x}^2} = n - N
# $$
#
# und
#
# $$ 
# \frac{1}{[t]} \frac{\partial n}{\partial \tilde{t}} = - \frac{D_n}{[l]^2} \ \frac{\partial}{\partial x} \left( n \ \frac{\partial u}{\partial x} - \frac{\partial n}{\partial x} \right) + R 
# $$
#
# oder
#
# $$ 
# \frac{\partial n}{\partial \tilde{t}} = - \tilde{D}_n \ \frac{\partial}{\partial x} \left( n \ \frac{\partial u}{\partial x} - \frac{\partial n}{\partial x} \right) +  \tilde{R} 
# $$
#
# mit 
#
# $$ 
# \tilde{D}_n = D_n \frac{[t]}{[l]^2} \Leftrightarrow [t] = [l]^2 \ \frac{ \tilde{D}_n } { D_n } 
# $$
#     
# und
#
# $$ \tilde{R} = \frac{n - N_D}{\tilde{\tau}}$$ 
#
# mit $\tilde{\tau} = \tau / [t]$. 
#
# $\tilde{\lambda} = 1$ und $\tilde{D_n} = 1$ werden mit
# $[l] = \lambda$ und $[t] = \frac{\lambda^2}{D_n}$ erreicht:

# %% [markdown]
# ### Discretization

# %% [markdown]
# Naive Diskretisierung (skaliert):
#
# $$ \frac{1}{\Delta x^2} ( u_{i+1}-2u_i+u_{i-1} ) = n_i - N_i $$
#
# $$ \frac{1}{\Delta t} ( n_{i,j+1} - n_{i,j} ) = - \frac{1}{\Delta x^2} \cdot \left[ \frac{1}{4} (n_{i+1} - n_{i-1}) (u_{i+1} - u_{i-1}) + n_i ( u_{i+1} - 2 u_i + u_{i-1} ) - ( n_{i+1} - 2 n_i + n_{i-1} ) \right] + \frac{ n_i - N_i}{ \tilde{\tau} } $$
#
# Stationär:
#
# $$
#  u_{i+1}-2u_i+u_{i-1} - \Delta x^2 \cdot n_i + \Delta x^2 \cdot N_i = 0
# $$
#
# und
#
# $$
#   \frac{1}{4} (n_{i+1} - n_{i-1}) (u_{i+1} - u_{i-1}) + n_i ( u_{i+1} - 2 u_i + u_{i-1} ) - ( n_{i+1} - 2 n_i + n_{i-1} ) - \Delta x^2 \cdot \frac{ n_i - N_i}{ \tilde{\tau} } = 0
# $$

# %% [markdown]
# ### Newton-Iteration für gekoppeltes nicht-lineares Gleichungssystem

# %% [markdown]
# Idee: Löse nicht-lineares Finite-Differenzen-Gleichungssystem über Newton-Verfahren  
#
# $$ \vec{F}(\vec{x}_{k+1}) = F(\vec{x}_k + \Delta \vec{x}_k) \approx F(\vec{x}_k) + \mathbf{J_F}(\vec{x}_k) \cdot \Delta \vec{x}_k + \mathcal{O}(\Delta x^2)$$
#    
# mit Unbekannter $\vec{x_k} = \{u_1^k, \dots, u_N^k, n_1^k, \dots, n_N^k\}$  und damit
#
# $$ \Rightarrow \Delta \vec{x}_k = - \mathbf{J}_F^{-1} ~ F(\vec{x}_k)$$
#
# wobei die Jacobi-Matrix $2N \times 2N$ Einträge
#
# $$ \mathbf{J}_{ij}(\vec{x}_k) = \frac{\partial F_i}{\partial x_j} (\vec{x}_k) $$
#
# besitzt, die bei jedem Iterationsschritt für $\vec{x}_k$ ausgewertet werden.
# Der tatsächliche Aufwand liegt in der Invertierung der Jacobi-Matrix, um in jeder Iteration $k$ den Korrekturschritt $\Delta \vec{x}_k$ zu finden.m

# %% [markdown]
# $F(x)$ wird wie unten definiert als:
#
# $$
#  u_{i+1}-2u_i+u_{i-1} - \Delta x^2 \cdot n_i + \Delta x^2 \cdot N_i = 0
# $$
#
# und
#
# $$
#   \frac{1}{4} (n_{i+1} - n_{i-1}) (u_{i+1} - u_{i-1}) + n_i ( u_{i+1} - 2 u_i + u_{i-1} ) - ( n_{i+1} - 2 n_i + n_{i-1} ) - \Delta x^2 \cdot \frac{ n_i - N_i}{ \tilde{\tau} } = 0
# $$

# %% [markdown]
# ### Controlled-Volume

# %% [markdown]
# Drücke nicht-linearen Teil der Transportgleichung (genauer, des Flusses) über Bernoulli-Funktionen 
#
# $$ B(x) = \frac{x}{\exp(x)-1} $$ 
#
# aus (siehe Vorlesungsskript). Damit wir in der Nähe von 0 nicht "in die Bredouille geraten", verwenden wir hier lieber die Taylorentwicklung. In der Literatur (Selbherr, S. Analysis and Simulation of Semiconductor Devices, Spriger 1984) wird eine noch aufwendigere stückweise Definition empfohlen, allerdings werden wir im Folgenden sehen, dass unser Ansatz für dieses stationäre Problem genügt.
#

# %% [markdown]
# ## Implementation for Poisson-Nernst-Planck system

# %% [markdown]
# Poisson-Nernst-Planck system for $k = {1 \dots M}$ ion species in dimensionless formulation
#
# $$ \nabla^2 u + \rho(n_{1},\dots,n_{M}) = 0 $$
#
# $$ \nabla^2 n_k + \nabla ( z_k n_k \nabla u ) = 0 \quad \text{for} \quad k = 1 \dots M $$
#
# yields a naive finite difference discretization on $i = {1 \dots N}$ grid points for $k = {1 \dots M}$ ion species
#
# $$ \frac{1}{\Delta x^2} ( u_{i+1}-2u_i+u_{i-1} )  + \frac{1}{2} \sum_{k=1}^M z_k n_{i,k} = 0 $$
#
# $$ - \frac{1}{\Delta x^2} \cdot \left[ \frac{1}{4} z_k (n_{i+1,k} - n_{i-1,k}) (u_{i+1} - u_{i-1}) + z_k n_{i,k} ( u_{i+1} - 2 u_i + u_{i-1} ) + ( n_{i+1,k} - 2 n_{i,k} + n_{i-1,k} ) \right] $$
#
# or rearranged
#
# $$ u_{i+1}-2 u_i+u_{i-1} + \Delta x^2 \frac{1}{2} \sum_{k=1}^M z_k n_{i,k}  = 0 $$
#
# and
#
# $$
#   \frac{1}{4} z_k (n_{i+1,k} - n_{i-1,k}) (u_{i+1,k} - u_{i-1,k}) + z_k n_{i,k} ( u_{i+1} - 2 u_i + u_{i-1} ) - ( n_{i+1,k} - 2 n_{i,k} + n_{i-1,k} ) = 0
# $$

# %% [markdown]
# ### Controlled Volumes, 1D

# %% [markdown]
# Finite differences do not converge in our non-linear systems. Instead, we express non-linear part of the Nernts-Planck equations with Bernoulli function (Selberherr, S. Analysis and Simulation of Semiconductor Devices, Spriger 1984)
#
# $$ B(x) = \frac{x}{\exp(x)-1} $$ 

# %%
def B(x):
    return np.where( np.abs(x) < 1e-9,
        1 - x/2 + x**2/12 - x**4/720, # Taylor
        x / ( np.exp(x) - 1 ) )


# %%
xB = np.arange(-10,10,0.1)

# %%
plt.plot( xB ,B( xB ), label="$B(x)$")
plt.plot( xB, - B(-xB), label="$-B(-x)$")
plt.plot( xB, B(xB)-B(-xB), label="$B(x)-B(-x)$")
plt.legend()

# %% [markdown]
# Looking at (dimensionless) flux $j_k$ throgh segment $k$ in between grid points $i$ and $j$,
#
# $$ j_k = - \frac{dn}{dx} - z n \frac{du}{dx} $$
#
# for an ion species with number charge $z$ and (dimensionless) concentration $n$, 
# we assume (dimensionless) potential $u$ to behave linearly within this segment. The linear expression
#
# $$ u = \frac{u_j - u_i}{L_k} \cdot \xi_k + u_i = a_k \xi_k + u_i $$ 
#
# with the segment's length $L_k = \Delta x$ for uniform discretization, $\xi_k = x - x_i$ and proportionality factor $a_k = \frac{u_j - u_i}{L_k}$ leadsd to a flux
#
# $$ j_k = - \frac{dn}{d\xi} - z a_k n  $$
#
# solvable for $v$ via
#
# $$ \frac{dn}{d\xi} = - z a_k n - j_k $$
#
# or 
#
# $$ \frac{dn}{z a_k n + j_k} = - d\xi \text{.} $$
#
# We intergate from grid point $i$ to $j$
#
# $$ \int_{n_i}^{n_j} \frac{1}{z a_k n + j_k} dn = - L_k $$
#
# and find
#
# $$ \frac{1}{(z a_k)} \left[ \ln(j_k + z a_k n) \right]_{n_i}^{n^j} = - L_k $$
#
# or
#
# $$ \ln(j_k + z a_k n_j) - \ln(j_k + z a_k n_i) = - z a_k L_k $$
#
# which we solve for $j_k$ by rearranging
#
# $$ \frac{j_k + z a_k n_j}{j_k + z a_k n_i} = e^{- z a_k L_k} $$
#
# $$ j_k + z a_k n_j = (j_k + z a_k n_i) e^{- z a_k L_k} $$
#
# $$ j_k ( 1 - e^{- z a_k L_k} ) = - z a_k n_j  +  z a_k n_i e^{- z a_k L_k} $$
#
# $$j_k = \frac{z a_k n_j}{e^{- z a_k L_k} - 1}  +  \frac{ z a_k n_i e^{- z a_k L_k}}{ 1 - e^{- z a_k L_k}}$$
#
# $$j_k = \frac{1}{L_k} \cdot \left[ \frac{z a_k L_k n_j}{e^{- z a_k L_k} - 1}  +  \frac{ z a_k L_k n_i }{ e^{z a_k L_k} - 1} \right] $$
#
# or with $B(x) = \frac{x}{e^x-1}$ expressed as
#
# $$j_k = \frac{1}{L_k} \cdot \left[ - n_j B( - z a_k L_k ) +  n_i B( z a_k L_k) \right] $$
#
# and resubstituting $a_k = \frac{u_j - u_i}{L_k}$ as
#
# $$j_k = - \frac{1}{L_k} \cdot \left[ n_j B( z [u_i - u_j] ) - n_i B( z [u_j - u_i] ) \right] \ \text{.}$$
#
# When employing our 1D uniform grid with $j_k = j_{k-1}$ for all $k = 1 \dots N$,
#
# $$ j_k \Delta x  = n_{i+1} B( z [u_i - u_{i+1}] ) - n_i B( z [u_{i+1} - u_i] ) $$
#
# and
#
# $$ j_{k-1} \Delta x  = n_i B( z [u_{i-1} - u_i] ) - n_{i-1} B( z [u_i - u_{i-1}] ) $$
#
# require
#
# $$ n_{i+1} B( z [u_i - u_{i+1}] ) - n_i \left( B( z [u_{i+1} - u_i] ) + B( z [u_{i-1} - u_i] ) \right) + n_{i-1} B( z [u_i - u_{i-1}] ) = 0 $$

# %% [markdown]
# ## Test case 1: PNP interface system, 0.1 mM NaCl, positive potential u = 0.05 V

# %%
# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=1e-07
delta_u=0.05

# %%
# define desired system
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)
# constructor takes keyword arguments
#   c=array([0.1, 0.1]), z=array([ 1, -1]), L=1e-07, T=298.15, delta_u=0.05, relative_permittivity=79, vacuum_permittivity=8.854187817620389e-12, R=8.3144598, F=96485.33289
# with default values set for 0.1 mM NaCl aqueous solution across 100 nm  and 0.05 V potential drop

# %%
pnp.useStandardInterfaceBC()

# %%
pnp.output = True # let's Newton solver display convergence plots
uij, nij, lamj = pnp.solve()

# %% [markdown]
# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# %%
# analytic Poisson-Boltzmann distribution and numerical solution to full Poisson-Nernst-Planck system
x = np.linspace(0,L,100)
phi = potential(x, c, z, delta_u) 
C =   concentration(x, c, z, delta_u)
rho = charge_density(x, c, z, delta_u) 
deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[16,10])
ax1.axvline(x=deb, label='Debye Length', color='grey', linestyle=':')

ax1.plot(x/sc.nano, phi, marker='', color='tomato', label='potential, PB', linewidth=1, linestyle='--')
ax1.plot(pnp.grid/sc.nano, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax2.plot(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)

ax3.plot(x/sc.nano, rho, label='Charge density, PB', color='grey', linewidth=1, linestyle='--')
ax3.plot(pnp.grid/sc.nano, pnp.charge_density, label='Charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax4.semilogy(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax4.semilogy(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')


ax1.set_xlabel('z [nm]')
ax1.set_ylabel('potential (V)')
ax2.set_ylabel('concentration (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
ax4.set_ylabel('concentration (mM)')

#fig.legend(loc='center')
ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=15)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=15)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=15)

fig.tight_layout()
plt.show()

# %% [markdown]
# #### Potential at left and right hand side of domain

# %%
(pnp.potential[0],pnp.potential[-1])

# %% [markdown]
# #### Residual cation flux at interface and at open right hand side

# %%
( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Residual anion flux at interface and at open right hand side

# %%
(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Cation concentration at interface and at open right hand side

# %%
(pnp.concentration[0,0],pnp.concentration[0,-1])

# %% [markdown]
# #### Anion concentration at interface and at open right hand side

# %%
(pnp.concentration[1,0],pnp.concentration[1,-1])

# %% [markdown]
# ## Test case 2: PNP interface system, 0.1 mM NaCl, negative potential u = -0.05 V, analytical solution as initial values

# %%
# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=1e-07
delta_u=-0.05

# %%
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)

# %%
pnp.useStandardInterfaceBC()

# %%
# initial config
x = np.linspace(0, pnp.L, pnp.Ni)
phi = potential(x, c, z, delta_u) 
C = concentration(x, c, z, delta_u)

# %%
pnp.ni0 = C / pnp.c_unit # manually remove dimensions from analyatical solution

# %%
ui0 = pnp.initial_values()

# %%
plt.plot(ui0) # solution to linear Poisson equation under assumption of fixed charge density distribution

# %%
pnp.output = True # let's Newton solver display convergence plots
uij, nij, lamj = pnp.solve() # no faster convergence than above, compare convergence plots for test case 1

# %% [markdown]
# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# %%
# analytic Poisson-Boltzmann distribution and numerical solution to full Poisson-Nernst-Planck system
x = np.linspace(0,L,100)
phi = potential(x, c, z, delta_u) 
C =   concentration(x, c, z, delta_u)
rho = charge_density(x, c, z, delta_u) 
deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[16,10])
ax1.axvline(x=deb, label='Debye Length', color='grey', linestyle=':')

ax1.plot(x/sc.nano, phi, marker='', color='tomato', label='potential, PB', linewidth=1, linestyle='--')
ax1.plot(pnp.grid/sc.nano, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax2.plot(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)

ax3.plot(x/sc.nano, rho, label='Charge density, PB', color='grey', linewidth=1, linestyle='--')
ax3.plot(pnp.grid/sc.nano, pnp.charge_density, label='Charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax4.semilogy(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax4.semilogy(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')


ax1.set_xlabel('z [nm]')
ax1.set_ylabel('potential (V)')
ax2.set_ylabel('concentration (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
ax4.set_ylabel('concentration (mM)')

#fig.legend(loc='center')
ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=15)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=15)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=15)

fig.tight_layout()
plt.show()

# %% [markdown]
# #### Potential at left and right hand side of domain

# %%
(pnp.potential[0],pnp.potential[-1])

# %% [markdown]
# #### Residual cation flux at interface and at open right hand side

# %%
( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Residual anion flux at interface and at open right hand side

# %%
( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,1) )

# %% [markdown]
# #### Cation concentration at interface and at open right hand side

# %%
(pnp.concentration[0,0],pnp.concentration[0,-1])

# %% [markdown]
# #### Anion concentration at interface and at open right hand side

# %%
(pnp.concentration[1,0],pnp.concentration[1,-1])

# %% [markdown]
# ## Test case 3: PNP interface system, 0.1 mM NaCl, positive potential u = 0.05 V, 200 nm domain

# %%
# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=2e-07
delta_u=0.05

# %%
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)

# %%
pnp.useStandardInterfaceBC()

# %%
pnp.output = True
uij, nij, lamj = pnp.solve()

# %% [markdown]
# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# %%
# analytic Poisson-Boltzmann distribution and numerical solution to full Poisson-Nernst-Planck system
x = np.linspace(0,L,100)
phi = potential(x, c, z, delta_u) 
C =   concentration(x, c, z, delta_u)
rho = charge_density(x, c, z, delta_u) 
deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[16,10])
ax1.axvline(x=deb, label='Debye Length', color='grey', linestyle=':')

ax1.plot(x/sc.nano, phi, marker='', color='tomato', label='potential, PB', linewidth=1, linestyle='--')
ax1.plot(pnp.grid/sc.nano, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax2.plot(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)

ax3.plot(x/sc.nano, rho, label='Charge density, PB', color='grey', linewidth=1, linestyle='--')
ax3.plot(pnp.grid/sc.nano, pnp.charge_density, label='Charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax4.semilogy(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax4.semilogy(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')


ax1.set_xlabel('z [nm]')
ax1.set_ylabel('potential (V)')
ax2.set_ylabel('concentration (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
ax4.set_ylabel('concentration (mM)')

#fig.legend(loc='center')
ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=15)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=15)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=15)

fig.tight_layout()
plt.show()

# %% [markdown]
# Analytic PB and approximate PNP solution indistinguishable.

# %% [markdown]
# #### Potential at left and right hand side of domain

# %%
(pnp.potential[0],pnp.potential[-1])

# %% [markdown]
# #### Residual cation flux at interface and at open right hand side

# %%
( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Residual anion flux at interface and at open right hand side

# %%
(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Cation concentration at interface and at open right hand side

# %%
(pnp.concentration[0,0],pnp.concentration[0,-1])

# %% [markdown]
# #### Anion concentration at interface and at open right hand side

# %%
(pnp.concentration[1,0],pnp.concentration[1,-1])

# %% [markdown]
# ## Test case 4: 1D electrochemical cell, 0.1 mM NaCl, positive potential u = 0.05 V, 100 nm domain

# %%
# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=1e-07
delta_u=0.05

# %%
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)

# %%
pnp.useStandardCellBC()

# %%
pnp.output = True
xij = pnp.solve()

# %% [markdown]
# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# %%
# analytic Poisson-Boltzmann distribution and numerical solution to full Poisson-Nernst-Planck system
x = np.linspace(0,L,100)
phi = potential(x, c, z, delta_u) 
C =   concentration(x, c, z, delta_u)
rho = charge_density(x, c, z, delta_u) 
deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[16,10])
ax1.axvline(x=deb, label='Debye Length', color='grey', linestyle=':')

ax1.plot(x/sc.nano, phi, marker='', color='tomato', label='potential, PB', linewidth=1, linestyle='--')
ax1.plot(pnp.grid/sc.nano, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax2.plot(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)

ax3.plot(x/sc.nano, rho, label='Charge density, PB', color='grey', linewidth=1, linestyle='--')
ax3.plot(pnp.grid/sc.nano, pnp.charge_density, label='Charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax4.semilogy(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax4.semilogy(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')


ax1.set_xlabel('z [nm]')
ax1.set_ylabel('potential (V)')
ax2.set_ylabel('concentration (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
ax4.set_ylabel('concentration (mM)')

#fig.legend(loc='center')
ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=15)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=15)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=15)

fig.tight_layout()
plt.show()

# %%
# analytic Poisson-Boltzmann distribution and numerical solution to full Poisson-Nernst-Planck system
x = np.linspace(0,L,100)

deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[16,10])
ax1.set_xlabel('z [nm]')
ax1.plot(pnp.grid/sc.nano, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')
ax1.axvline(x=deb, label='Debye Length', color='grey', linestyle=':')

ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)
ax3.plot(pnp.grid/sc.nano, pnp.charge_density, label='charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

ax1.set_xlabel('z [nm]')
ax1.set_ylabel('potential (V)')
ax2.set_ylabel('concentration (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
ax4.set_xlabel('z [nm]')
ax4.set_ylabel('concentration (mM)')

#fig.legend(loc='center')
ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=15)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=15)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=15)

fig.tight_layout()
plt.show()

# %% [markdown]
# #### Potential at left and right hand side of domain

# %%
(pnp.potential[0],pnp.potential[-1])

# %% [markdown]
# #### Residual cation flux at interfaces

# %%
( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Residual anion flux at interfaces

# %%
(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Cation concentration at interfaces

# %%
(pnp.concentration[0,0],pnp.concentration[0,-1])

# %% [markdown]
# #### Anion concentration at interfaces

# %%
(pnp.concentration[1,0],pnp.concentration[1,-1])

# %% [markdown]
# #### Equilibrium cation and anion amount

# %%
( pnp.numberConservationConstraint(pnp.xij1,0,0), pnp.numberConservationConstraint(pnp.xij1,1,0) )

# %% [markdown]
# #### Initial cation and anion amount

# %%
( pnp.numberConservationConstraint(pnp.xi0,0,0), pnp.numberConservationConstraint(pnp.xi0,1,0) )

# %% [markdown]
# #### Species conservation

# %%
(pnp.numberConservationConstraint(pnp.xij1,0,
                                 pnp.numberConservationConstraint(pnp.xi0,0,0)), 
 pnp.numberConservationConstraint(pnp.xij1,1,
                                 pnp.numberConservationConstraint(pnp.xi0,1,0)) )

# %% [markdown]
# ## Test case 5: 1D electrochemical cell, 0.1 mM NaCl, positive potential u = 0.05 V, 100 nm domain, 0.5 nm compact layer

# %% [markdown]
# At high potentials or bulk concentrations, pure PNP systems yield unphysically high concentrations and steep gradients close to the boundary, as an ion's finite size is not accounted for.
# In addition, high gradients can lead to convergence issues. This problem can be alleviated by assuming a Stern layer (compact layer) at the interface. 
# This compact layer is parametrized by its thickness $\lambda_S$ and can be treated explicitly by prescribing a linear potential regime across the compact layer region, or by 
# the implicit parametrization of a compact layer with uniform charge density as Robin boundary conditions on the potential. 

# %%
c        = [1000,1000] # high concentrations close to NaCl's solubility limit in water
delta_u  = 0.05
L        = 30e-10 # tiny gap of 3 nm
lambda_S =  5e-10 # 0.5 nm Stern layer

# %%
pnp_no_compact_layer = PoissonNernstPlanckSystem(c,z,L,delta_u=delta_u, e=1e-12)

# %%
pnp_with_explicit_compact_layer = PoissonNernstPlanckSystem(c,z,L, delta_u=delta_u,lambda_S=lambda_S, e=1e-12)

# %%
pnp_with_implicit_compact_layer = PoissonNernstPlanckSystem(c,z,L, delta_u=delta_u,lambda_S=lambda_S, e=1e-12)

# %%
pnp_no_compact_layer.useStandardCellBC()

# %%
pnp_with_explicit_compact_layer.useSternLayerCellBC(implicit=False)

# %%
pnp_with_implicit_compact_layer.useSternLayerCellBC(implicit=True)

# %%
pnp_no_compact_layer.output = True
xij_no_compact_layer = pnp_no_compact_layer.solve()

# %%
pnp_with_explicit_compact_layer.output = True
xij_with_explicit_compact_layer = pnp_with_explicit_compact_layer.solve()

# %%
pnp_with_implicit_compact_layer.output = True
xij_with_implicit_compact_layer = pnp_with_implicit_compact_layer.solve()

# %%
x = np.linspace(0,L,100)

deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[18,10])

# 1 - potentials
ax1.axvline(x=deb/sc.nano, label='Debye Length', color='grey', linestyle=':')
ax1.plot(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.potential, marker='', color='tab:red', label='potential, without compact layer', linewidth=1, linestyle='-')
ax1.plot(pnp_with_explicit_compact_layer.grid/sc.nano, pnp_with_explicit_compact_layer.potential, marker='', color='tab:red', label='potential, with explicit compact layer', linewidth=1, linestyle='--')
ax1.plot(pnp_with_implicit_compact_layer.grid/sc.nano, pnp_with_implicit_compact_layer.potential, marker='', color='tab:red', label='potential, with Robin BC', linewidth=2, linestyle=':')

# 2 - conencentratiosn
ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')

ax2.plot(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer', linewidth=2, linestyle='-')
ax2.plot(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer', linewidth=2, linestyle='-')

ax2.plot(pnp_with_explicit_compact_layer.grid/sc.nano, pnp_with_explicit_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, with explicit compact layer', linewidth=2, linestyle='--')
ax2.plot(pnp_with_explicit_compact_layer.grid/sc.nano, pnp_with_explicit_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, with explicit compact layer', linewidth=2, linestyle='--')

ax2.plot(pnp_with_implicit_compact_layer.grid/sc.nano, pnp_with_implicit_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, with Robin BC', linewidth=2, linestyle=':')
ax2.plot(pnp_with_implicit_compact_layer.grid/sc.nano, pnp_with_implicit_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, with Robin BC', linewidth=2, linestyle=':')

# 3 - charge densities
ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)
ax3.plot(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.charge_density, label='charge density, without compact layer', color='grey', linewidth=1, linestyle='-')
ax3.plot(pnp_with_explicit_compact_layer.grid/sc.nano, pnp_with_explicit_compact_layer.charge_density, label='charge density, with explicit compact layer', color='grey', linewidth=1, linestyle='--')
ax3.plot(pnp_with_implicit_compact_layer.grid/sc.nano, pnp_with_implicit_compact_layer.charge_density, label='charge density, with Robin BC', color='grey', linewidth=1, linestyle=':')

# 4 - concentrations, semi log
ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')

ax4.semilogy(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer', linewidth=2, linestyle='-')
ax4.semilogy(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer', linewidth=2, linestyle='-')

ax4.semilogy(pnp_with_explicit_compact_layer.grid/sc.nano, pnp_with_explicit_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, with explicit compact layer', linewidth=2, linestyle='--')
ax4.semilogy(pnp_with_explicit_compact_layer.grid/sc.nano, pnp_with_explicit_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, with explicit compact layer', linewidth=2, linestyle='--')

ax4.semilogy(pnp_with_implicit_compact_layer.grid/sc.nano, pnp_with_implicit_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, with Robin BC', linewidth=2, linestyle=':')
ax4.semilogy(pnp_with_implicit_compact_layer.grid/sc.nano, pnp_with_implicit_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, with Robin BC', linewidth=2, linestyle=':')

ax1.set_xlabel('z [nm]')
ax1.set_ylabel('potential (V)')
ax2.set_ylabel('concentration (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
#ax3.yaxis.set_major_formatter(formatter)
ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2,10), useOffset=False, useMathText=False)
ax4.set_xlabel('z [nm]')
ax4.set_ylabel('concentration (mM)')

#fig.legend(loc='center')
ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=12)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=12)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=12)

fig.tight_layout()
plt.show()

# %% [markdown]
# #### Potential at left and right hand side of domain
#
# (pnp_no_compact_layer.potential[0],pnp_no_compact_layer.potential[-1])
#
# (pnp_with_explicit_compact_layer.potential[0],pnp_with_explicit_compact_layer.potential[-1])
#
# (pnp_with_implicit_compact_layer.potential[0],pnp_with_implicit_compact_layer.potential[-1])
#
# #### Residual cation flux at interfaces
#
# ( pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0) )
#
# ( pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0) )
#
# ( pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0) )
#
# #### Residual cation flux at interfaces
#
# ( pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1) )
#
# ( pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1) )
#
# ( pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1) )
#
# #### Cation concentration at interfaces
#
# (pnp_no_compact_layer.concentration[0,0],pnp_no_compact_layer.concentration[0,-1])
#
# (pnp_with_explicit_compact_layer.concentration[0,0],pnp_with_explicit_compact_layer.concentration[0,-1])
#
# (pnp_with_implicit_compact_layer.concentration[0,0],pnp_with_implicit_compact_layer.concentration[0,-1])
#
# #### Anion concentration at interfaces
# L
# (pnp_no_compact_layer.concentration[1,0],pnp_no_compact_layer.concentration[1,-1])
#
# (pnp_with_explicit_compact_layer.concentration[1,0],pnp_with_explicit_compact_layer.concentration[1,-1])
#
# (pnp_with_implicit_compact_layer.concentration[1,0],pnp_with_implicit_compact_layer.concentration[1,-1])
#
# #### Equilibrium cation and anion amount
#
# ( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,0) )
#
# ( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,0) )
#
# ( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,0) )
#
# #### Initial cation and anion amount
#
# ( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0) )
#
# ( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0) )
#
# ( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0) )
#
# #### Species conservation
#
# (pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,
#                                  pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0)), 
#  pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,
#                                  pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0)) )
#
# (pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,
#                                  pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0)), 
#  pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,
#                                  pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0)) )
#
# (pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,
#                                  pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0)), 
#  pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,
#                                  pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0)) )

# %% [markdown]
# ## Sample application of 1D electrochemical cell model:

# %% [markdown]
# We want to fill a gap of 3 nm between gold electrodes with 0.2 wt % NaCl aqueous solution, apply a small potential difference and generate an initial configuration for LAMMPS within a cubic box:

# %%
box_Ang=np.array([50.,50.,50.]) # Angstrom

# %%
box_m = box_Ang*sc.angstrom

# %%
box_m

# %%
vol_AngCube = box_Ang.prod() # Angstrom^3

# %%
vol_mCube = vol_AngCube*sc.angstrom**3

# %% [markdown]
# With a concentration of 0.2 wt %, we are close to NaCl's solubility limit in water.
# We estimate molar concentrations and atom numbers in our box:

# %%
# enter number between 0 ... 0.2 
weight_concentration_NaCl = 0.2 # wt %
# calculate saline mass density g/cm³
saline_mass_density_kg_per_L  = 1 + weight_concentration_NaCl * 0.15 / 0.20 # g / cm^3, kg / L
# see https://www.engineeringtoolbox.com/density-aqueous-solution-inorganic-sodium-salt-concentration-d_1957.html

# %%
saline_mass_density_g_per_L = saline_mass_density_kg_per_L*sc.kilo

# %%
molar_mass_H2O = 18.015 # g / mol
molar_mass_NaCl  = 58.44 # g / mol

# %%
cNaCl_M = weight_concentration_NaCl*saline_mass_density_g_per_L/molar_mass_NaCl # mol L^-1

# %%
cNaCl_mM = np.round(cNaCl_M/sc.milli) # mM

# %%
cNaCl_mM

# %%
n_NaCl = np.round(cNaCl_mM*vol_mCube*sc.value('Avogadro constant'))

# %%
n_NaCl

# %%
c = [cNaCl_mM,cNaCl_mM]
z = [1,-1]
L=box_m[2]
lamda_S = 2.0e-10
delta_u  = 0.5

# %%
pnp = PoissonNernstPlanckSystem(c,z,L, lambda_S=lambda_S, delta_u=delta_u, N=200, maxit=20, e=1e-6)

# %%
pnp.useSternLayerCellBC()

# %%
pnp.output = True
xij = pnp.solve()

# %%
# analytic Poisson-Boltzmann distribution and numerical solution to full Poisson-Nernst-Planck system
x = np.linspace(0,L,100)

deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[16,10])
ax1.set_xlabel('z [nm]')
ax1.plot(pnp.grid/sc.nano, pnp.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')
ax1.axvline(x=deb, label='Debye Length', color='grey', linestyle=':')

ax3 = ax1.twinx()
# Offset the right spine of ax3.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines["right"].set_position(("axes", 1.1))
# Having been created by twinx, ax3 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(ax3)
# Second, show the right spine.
ax3.spines["right"].set_visible(True)
ax3.plot(pnp.grid/sc.nano, pnp.charge_density, label='charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

ax1.set_xlabel('z [nm]')
ax1.set_ylabel('potential (V)')
ax2.set_ylabel('concentration (mM)')
ax3.set_ylabel(r'charge density $\rho \> (\mathrm{C}\> \mathrm{m}^{-3})$')
ax4.set_xlabel('z [nm]')
ax4.set_ylabel('concentration (mM)')

#fig.legend(loc='center')
ax1.legend(loc='upper right',  bbox_to_anchor=(-0.1,1.02), fontsize=15)
ax2.legend(loc='center right', bbox_to_anchor=(-0.1,0.5),  fontsize=15)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=15)

fig.tight_layout()
plt.show()

# %% [markdown]
# #### Potential at left and right hand side of domain

# %%
(pnp.potential[0],pnp.potential[-1])

# %% [markdown]
# #### Residual cation flux at interfaces

# %%
( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Residual anion flux at interfaces

# %%
(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )

# %% [markdown]
# #### Cation concentration at interfaces

# %%
(pnp.concentration[0,0],pnp.concentration[0,-1])

# %% [markdown]
# #### Anion concentration at interfaces

# %%
(pnp.concentration[1,0],pnp.concentration[1,-1])

# %% [markdown]
# #### Equilibrium cation and anion amount

# %%
( pnp.numberConservationConstraint(pnp.xij1,0,0), pnp.numberConservationConstraint(pnp.xij1,1,0) )

# %% [markdown]
# #### Initial cation and anion amount

# %%
( pnp.numberConservationConstraint(pnp.xi0,0,0), pnp.numberConservationConstraint(pnp.xi0,1,0) )

# %% [markdown]
# #### Species conservation

# %%
(pnp.numberConservationConstraint(pnp.xij1,0,
                                 pnp.numberConservationConstraint(pnp.xi0,0,0)), 
 pnp.numberConservationConstraint(pnp.xij1,1,
                                 pnp.numberConservationConstraint(pnp.xi0,1,0)) )

# %% [markdown]
# ## Sampling
# First, convert the physical concentration distributions into a callable "probability density":

# %%
pnp.concentration.shape

# %%
distributions = [interpolate.interp1d(pnp.grid,pnp.concentration[i,:]) for i in range(pnp.concentration.shape[0])]

# %% [markdown]
# Normalization is not necessary here. Now we can sample the distribution of our $Na^+$ ions in z-direction.

# %%
na_coordinate_sample = continuous2discrete(
    distribution=distributions[0], box=box_m, count=n_NaCl)
histx, histy, histz = get_histogram(na_coordinate_sample, box=box_m, n_bins=51)
plot_dist(histz, 'Distribution of Na+ ions in z-direction', reference_distribution=distributions[0])

# %%
cl_coordinate_sample = continuous2discrete(
    distributions[1], box=box_m, count=n_NaCl)
histx, histy, histz = get_histogram(cl_coordinate_sample, box=box_m, n_bins=51)
plot_dist(histx, 'Distribution of Cl- ions in x-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box_m[0])
plot_dist(histy, 'Distribution of Cl- ions in y-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box_m[1])
plot_dist(histz, 'Distribution of Cl- ions in z-direction', reference_distribution=distributions[1])

# %% [markdown]
# ## Write to file
# To visualize our sampled coordinates, we utilize ASE to export it to some standard format, i.e. .xyz or LAMMPS data file.
# ASE speaks Ångström per default, thus we convert SI units:

# %%
sample_size = int(n_NaCl)

# %%
sample_size

# %%
na_atoms = ase.Atoms(
    symbols='Na'*sample_size,
    charges=[1]*sample_size,
    positions=na_coordinate_sample/sc.angstrom,
    cell=box_Ang,
    pbc=[1,1,0])

cl_atoms = ase.Atoms(
    symbols='Cl'*sample_size,
    charges=[-1]*sample_size,
    positions=cl_coordinate_sample/sc.angstrom,
    cell=box_Ang,
    pbc=[1,1,0])

system = na_atoms + cl_atoms

system

ase.io.write('NaCl_c_4_M_u_0.5_V_box_5x5x10nm_lambda_S_2_Ang.xyz',system,format='xyz')

# %%
# LAMMPS data format, units 'real', atom style 'full'
# before ASE 3.19.0b1, ASE had issues with exporting atom style 'full' in LAMMPS data file format, so do not expect this line to work for older ASE versions
ase.io.write('NaCl_c_4_M_u_0.5_V_box_5x5x10nm_lambda_S_2_Ang.lammps',system,format='lammps-data',units="real",atom_style='full')
