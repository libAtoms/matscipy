#!/usr/bin/env python
# coding: utf-8

# # continuous2discrete 
# 
# *Johannes Hörmann, Lukas Elflein, 2019*
# 
# from continuous electrochemical double layer theory to discrete coordinate sets

# In[1]:


# for dynamic module reload during testing, code modifications take immediate effect
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# stretching notebook width across whole window
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[3]:


# basics
import logging
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


# In[4]:


# sampling
from scipy import interpolate
from matscipy.electrochemistry import continuous2discrete
from matscipy.electrochemistry import plot_dist
from matscipy.electrochemistry import get_histogram


# In[5]:


# electrochemistry basics
from matscipy.electrochemistry import debye, ionic_strength


# In[6]:


# Poisson-Bolzmann distribution
from matscipy.electrochemistry.poisson_boltzmann_distribution import gamma, potential, concentration, charge_density


# In[7]:


# Poisson-Nernst-Planck solver
from matscipy.electrochemistry import PoissonNernstPlanckSystem


# In[8]:


# 3rd party file output
import ase
import ase.io


# In[9]:


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


# In[10]:


# Test 1
logging.info("Root logger")


# In[11]:


# Test 2
logger.info("Root Logger")


# # The Poisson-Boltzman Distribution
# *Lukas Elflein, 2019*
# 
# In order to understand lubrication better, we simulate thin layers of lubricant on a metallic surface, solvated in water.
# Different structures of lubricant films are created by varying parameters like their concentration and the charge of the surface.
# The lubricant is somewhat solvable in water, thus parts of the film will diffuse into the bulk water.
# Lubricant molecules are charged, and their distribution is roughly exponential.
# 
# As simplification, we  first create a solution of ions (Na+, purple; Cl-, green) in water (not shown).
# ![pic](https://i.ibb.co/Yh8DxVM/showpicture.png)
# 
# Close to the positively charged metallic surface, the electric potential (red) will be highest, falling off exponentially when further away.
# This potential attracts negatively charged Chlorine ions, and pushes positively charged Natrium ions away, resulting in a higher (lower) concentration of Clorine (Natrium) near the surface.
# 
# 

# To calculate this, we first need to find out how ions are distributed in solution.
# A good description of the concentrations of our ion species, $c_{\mathrm{Na}^+}$ and $c_{\mathrm{Cl}^-}$ or $c_i$ for $i \in \{\mathrm{Na}^+, \mathrm{Cl}^-\}$, is given by the solution to the Poisson-Boltzmann equation, here expressed with molar concentrations, Faraday constant and molar gas constant
# 
# $
# \begin{align}
# c_i(x) &= c_i^\infty e^{-F \phi(x)/R T}\\
# \phi(x) &= \frac{2 R T}{F} \log\left(\frac{1 + \gamma e^{-\kappa x}}{1- \gamma e^{-\kappa z}}\right) 
#         \approx \frac{4 R T}{F} \gamma e^{-\kappa x} \\
# \gamma &= \tanh(\frac{F \phi_0}{4 R T})\\
# \kappa &= 1/\lambda_D\\
# \lambda_D &= \Big(\frac{\epsilon \epsilon_0 R T}{F^2 \sum_{i} c_i^\infty z_i^2} \Big)^\frac{1}{2}
# \end{align}
# $
# 
# or alternatively expressed with number concentrations, elementary charge and Boltzmann constant instead
# 
# $
# \begin{align}
# \rho_{i}(x) &= \rho_{i}^\infty e^{ -e \phi(z) \> / \> k_B T}\\
# \phi(x) &= \frac{2k_B T}{e} \> \log\left(\frac{1 + \gamma e^{-\kappa z}}{1- \gamma e^{-\kappa z}}\right) 
#         \approx \frac{4k_B T}{e} \gamma e^{-\kappa x} \\
# \gamma &= \tanh\left(\frac{e\phi_0}{4k_B T}\right)\\
# \kappa &= 1/\lambda_D\\
# \lambda_D &= \left(\frac{\epsilon \epsilon_0 k_B T}{\sum_{i} \rho_i^\infty e^2 z_i^2} \right)^\frac{1}{2}
# \end{align}
# $
# 
# with
# * $x$: distance from interface $[\mathrm{m}]$
# * $\phi_0$: potential at the surface $[\mathrm{V}]$
# * $\phi(z)$: potential in the solution $[\mathrm{V}]$
# * $k_B$: Boltzmann Constant $[\mathrm{J}\> \mathrm{K}^{-1}]$
# * $R$: molar gas constant $[\mathrm{J}\> \mathrm{mol}^{-1}\> \mathrm{K}^{-1}]$
# * $T$: temperature $[\mathrm{K}]$
# * $e$: elementary charge (or Euler's constant when exponentiated) $[\mathrm{C}]$
# * $F$: Faraday constant $[\mathrm{C}\> \mathrm{mol}^{-1}]$
# * $\gamma$: term from Gouy-Chapmann theory
#     * $\gamma \rightarrow 1$ for high potentials
#     * $\phi(z) \approx \phi_0 e^{-\kappa z}$ for low potentials $\phi_0 \rightarrow 0$
# * $\lambda_D$: Debye Length ($\approx 34.0\>\mathrm{nm}$ for NaCl, $10^{-4} \mathrm{M}$, $25^\circ \mathrm{C}$)
# * $c{i}$: molar concentration of ion species $i$ $[\mathrm{mol}\> \mathrm{m}^{-3}]$
# * $c_{i}^\infty$: bulk molar concentration (at infinity, where the solution is homogeneous) $[\mathrm{mol}\> \mathrm{m}^{-3}]$
# * $\rho_{i}$: number concentration of ion species $i$ $[\mathrm{m}^{-3}]$
# * $\rho_{i}^\infty$: bulk number concentration $[\mathrm{m}^{-3}]$
# * $\epsilon$: relative permittivity of the solution $[1]$
# * $\epsilon_0$: vacuum permittivity $[\mathrm{F}\> \mathrm{m}^{-1}]$
# * $z_i$: number charge of species $i$ $[1]$ 
# 
# 
# These equations are implemented in `poisson_boltzmann_distribution.py`

# In[12]:


# Notes on units
# universal gas constant R = N_A * k_B, [R] = J mol^-1 K^-1
# Faraday constant F = N_a e, [F] = C mol^-1
print("Note on constants and units:")
print("[F]   = {}".format(sc.unit('Faraday constant')))
print("[R]   = {}".format(sc.unit('molar gas constant')))
print("[e]   = {}".format(sc.unit('elementary charge')))
print("[k_B] = {}".format(sc.unit('Boltzmann constant')))
print("F/R   = {}".format(sc.value('Faraday constant')/sc.value('molar gas constant')))
print("e/k_B = {}".format(sc.value('elementary charge')/sc.value('Boltzmann constant')))
print("F/R   = e/k_B !")


# In[13]:


# Debye length of 0.1 mM NaCl aqueous solution
c = [0.1,0.1] # mM
z = [1,-1]
deb = debye(c,z) 
print('Debye Length of 10^-4 M saltwater: {} nm (Target: 30.52 nm)'.format(round(deb/sc.nano, 2)))


# In[14]:


C = np.logspace(-3, 3, 50) # mM, 
# NaCl molar mass 58.443 g/mol and solubility limit in water at about 360 g/L
# means concentrations as high as a few M (mol/L), i.e. >> 1000 mM, are possible
debyes = np.array([debye([c,c], [1,-1]) for c in C])
fig, (ax1,ax2) = plt.subplots(
    nrows=1, ncols=2, figsize=[12,4], constrained_layout=True)
ax1.set_xlabel('concentration (mM)') # mM is mol / m^3
ax1.set_ylabel('Debye length at 25° [nm]')
ax1.semilogx(C, debyes/sc.nano, marker='.')
ax2.set_xlabel('concentration (mM)') # mM is mol / m^3
ax2.set_ylabel('Debye length at 25° [nm]')
ax2.loglog(C, debyes/sc.nano, marker='.')
plt.show()


# The debye length depends on the concentration of ions in solution, at low concentrations it becomes large. We can reproduce literature debye lengths with our function, so everything looks good.
# 
# ## Gamma Function
# 
# Next we calculate the gamma function $\gamma = \tanh(\frac{e\Psi(0)}{4k_B T})$

# In[15]:


x = np.linspace(-0.5, 0.5, 40)
gammas = gamma(x, 298.15)
plt.xlabel('Potential $\phi$ (V)')
plt.ylabel('$\gamma(\phi)$ at 298.15 K (1)')
plt.plot(x, gammas, marker='o')
plt.show()


# ## Potential
# 
# We plug these two functions into the expression for the potential
# 
# $\phi(z) = \frac{2k_B T}{e} \log\Big(\frac{1 + \gamma e^{-\kappa z}}{1- \gamma e^{-\kappa z}}\Big) 
#         \approx \frac{4k_B T}{e} \gamma e^{-\kappa z}$

# In[16]:


x = np.linspace(0, 2*10**-7, 10000) # 200 nm
c = [0.1,0.1]
z = [1,-1]
psi = potential(x, c, z, u=0.05)
plt.xlabel('x (nm)')
plt.ylabel('Potential (V)')
plt.plot(x/sc.nano, psi, marker='')
plt.show()


# The potential is smooth and looks roughly exponential. Everything good so far.
# 
# ## Concentrations
# 
# Now we obtain ion concentrations $c_i$ from the potential $\phi(x)$ via
# 
# $c_{i}(x) = c_{i}^\infty e^{-F \phi(x) \> / \> R T}$

# In[17]:


x = np.linspace(0, 100*10**-9, 2000)
c = [0.1,0.1]
z = [1,-1]
u = 0.05 

phi = potential(x, c, z, u)
C   = concentration(x, c, z, u)
rho = charge_density(x, c, z, u)


# In[18]:


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


# Potential and concentrations behave as expected.
# 
# ## Sampling
# First, convert the physical concentration distributions into a callable "probability density":

# In[19]:


distributions = [interpolate.interp1d(x,c) for c in C]


# Normalization is not necessary here. Now we can sample the distribution of our $Na^+$ ions in z-direction.

# In[20]:


x = y = 50e-9
z = 100e-9
box = np.array([x, y, z])
sample_size = 1000


# In[21]:


from scipy import optimize


# In[22]:


na_coordinate_sample = continuous2discrete(
    distribution=distributions[0], box=box, count=sample_size)
histx, histy, histz = get_histogram(na_coordinate_sample, box=box, n_bins=51)
plot_dist(histz, 'Distribution of Na+ ions in z-direction', reference_distribution=distributions[0])


# In[23]:


cl_coordinate_sample = continuous2discrete(
    distributions[1], box=box, count=sample_size)
histx, histy, histz = get_histogram(cl_coordinate_sample, box=box, n_bins=51)
plot_dist(histx, 'Distribution of Cl- ions in x-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box[0])
plot_dist(histy, 'Distribution of Cl- ions in y-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box[1])
plot_dist(histz, 'Distribution of Cl- ions in z-direction', reference_distribution=distributions[1])


# ## Write to file
# To visualize our sampled coordinates, we utilize ASE to export it to some standard format, i.e. .xyz or LAMMPS data file.
# ASE speaks Ångström per default, thus we convert SI units:

# In[24]:


na_atoms = ase.Atoms(
    symbols='Na'*sample_size,
    charges=[1]*sample_size,
    positions=na_coordinate_sample/sc.angstrom,
    cell=box/sc.angstrom,
    pbc=[1,1,0])

cl_atoms = ase.Atoms(
    symbols='Cl'*sample_size,
    charges=[-1]*sample_size,
    positions=cl_coordinate_sample/sc.angstrom,
    cell=box/sc.angstrom,
    pbc=[1,1,0])

system = na_atoms + cl_atoms

system

ase.io.write('NaCl_0.1mM_0.05V_50x50x100nm_at_interface_poisson_boltzmann_distributed.xyz',system,format='xyz')


# In[25]:


# LAMMPS data format, units 'real', atom style 'full'
# before ASE 3.19.0b1, ASE had issues with exporting atom style 'full' in LAMMPS data file format, so do not expect this line to work for older ASE versions
ase.io.write('NaCl_0.1mM_0.05V_50x50x100nm_at_interface_poisson_boltzmann_distributed.lammps',system,format='lammps-data',units="real",atom_style='full')


# # General Poisson-Nernst-Planck System

# For general systems, i.e. a nanogap between two electrodes with not necessarily binary electrolyte, no closed analytic solution exists.
# Thus, we solve the full Poisson-Nernst-Planck system of equations. 

# A binary Poisson-Nernst-Planck system corresponds to the transport problem in semiconductor physics.
# In this context, Debye length, charge carrier densities and potential are related as follows.

# ## Excursus: Transport problem in PNP junction (German)

# ### Debye length

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

# ### Dimensionless formulation

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

# ### Discretization

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

# ### Newton-Iteration für gekoppeltes nicht-lineares Gleichungssystem

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

# ### Controlled-Volume

# Drücke nicht-linearen Teil der Transportgleichung (genauer, des Flusses) über Bernoulli-Funktionen 
# 
# $$ B(x) = \frac{x}{\exp(x)-1} $$ 
# 
# aus (siehe Vorlesungsskript). Damit wir in der Nähe von 0 nicht "in die Bredouille geraten", verwenden wir hier lieber die Taylorentwicklung. In der Literatur (Selbherr, S. Analysis and Simulation of Semiconductor Devices, Spriger 1984) wird eine noch aufwendigere stückweise Definition empfohlen, allerdings werden wir im Folgenden sehen, dass unser Ansatz für dieses stationäre Problem genügt.
# 

# ## Implementation for Poisson-Nernst-Planck system

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

# ### Controlled Volumes, 1D

# Finite differences do not converge in our non-linear systems. Instead, we express non-linear part of the Nernts-Planck equations with Bernoulli function (Selberherr, S. Analysis and Simulation of Semiconductor Devices, Spriger 1984)
# 
# $$ B(x) = \frac{x}{\exp(x)-1} $$ 

# In[26]:


def B(x):
    return np.where( np.abs(x) < 1e-9,
        1 - x/2 + x**2/12 - x**4/720, # Taylor
        x / ( np.exp(x) - 1 ) )


# In[27]:


xB = np.arange(-10,10,0.1)


# In[28]:


plt.plot( xB ,B( xB ), label="$B(x)$")
plt.plot( xB, - B(-xB), label="$-B(-x)$")
plt.plot( xB, B(xB)-B(-xB), label="$B(x)-B(-x)$")
plt.legend()


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

# ## Test case 1: PNP interface system, 0.1 mM NaCl, positive potential u = 0.05 V

# In[29]:


# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=1e-07
delta_u=0.05


# In[30]:


# define desired system
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)
# constructor takes keyword arguments
#   c=array([0.1, 0.1]), z=array([ 1, -1]), L=1e-07, T=298.15, delta_u=0.05, relative_permittivity=79, vacuum_permittivity=8.854187817620389e-12, R=8.3144598, F=96485.33289
# with default values set for 0.1 mM NaCl aqueous solution across 100 nm  and 0.05 V potential drop


# In[31]:


pnp.useStandardInterfaceBC()


# In[32]:


pnp.init()


# In[33]:


pnp.output = True # let's Newton solver display convergence plots
uij, nij, lamj = pnp.solve()


# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# In[34]:


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


# #### Potential at left and right hand side of domain

# In[35]:


(pnp.potential[0],pnp.potential[-1])


# #### Residual cation flux at interface and at open right hand side

# In[36]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Residual anion flux at interface and at open right hand side

# In[37]:


(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Cation concentration at interface and at open right hand side

# In[38]:


(pnp.concentration[0,0],pnp.concentration[0,-1])


# #### Anion concentration at interface and at open right hand side

# In[39]:


(pnp.concentration[1,0],pnp.concentration[1,-1])


# ## Test case 2: PNP interface system, 0.1 mM NaCl, negative potential u = -0.05 V, analytical solution as initial values

# In[40]:


# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=1e-07
delta_u=-0.05


# In[41]:


pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)


# In[42]:


pnp.useStandardInterfaceBC()


# In[43]:


pnp.init()


# In[44]:


# initial config
x = np.linspace(0, pnp.L, pnp.Ni)
phi = potential(x, c, z, delta_u) 
C = concentration(x, c, z, delta_u)


# In[45]:


pnp.ni0 = C / pnp.c_unit # manually remove dimensions from analyatical solution


# In[46]:


ui0 = pnp.initial_values()


# In[47]:


plt.plot(ui0) # solution to linear Poisson equation under assumption of fixed charge density distribution


# In[48]:


pnp.output = True # let's Newton solver display convergence plots
uij, nij, lamj = pnp.solve() # no faster convergence than above, compare convergence plots for test case 1


# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# In[49]:


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


# #### Potential at left and right hand side of domain

# In[50]:


(pnp.potential[0],pnp.potential[-1])


# #### Residual cation flux at interface and at open right hand side

# In[51]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Residual anion flux at interface and at open right hand side

# In[52]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,1) )


# #### Cation concentration at interface and at open right hand side

# In[53]:


(pnp.concentration[0,0],pnp.concentration[0,-1])


# #### Anion concentration at interface and at open right hand side

# In[54]:


(pnp.concentration[1,0],pnp.concentration[1,-1])


# ## Test case 3: PNP interface system, 0.1 mM NaCl, positive potential u = 0.05 V, 200 nm domain

# In[55]:


# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=2e-07
delta_u=0.05


# In[56]:


pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)


# In[57]:


pnp.useStandardInterfaceBC()


# In[58]:


pnp.init()


# In[59]:


pnp.output = True
uij, nij, lamj = pnp.solve()


# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# In[60]:


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


# Analytic PB and approximate PNP solution indistinguishable.

# #### Potential at left and right hand side of domain

# In[61]:


(pnp.potential[0],pnp.potential[-1])


# #### Residual cation flux at interface and at open right hand side

# In[62]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Residual anion flux at interface and at open right hand side

# In[63]:


(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Cation concentration at interface and at open right hand side

# In[64]:


(pnp.concentration[0,0],pnp.concentration[0,-1])


# #### Anion concentration at interface and at open right hand side

# In[65]:


(pnp.concentration[1,0],pnp.concentration[1,-1])


# ## Test case 4: 1D electrochemical cell, 0.1 mM NaCl, positive potential u = 0.05 V, 100 nm domain

# In[66]:


# Test case parameters
c=[0.1, 0.1]
z=[ 1, -1] 
L=1e-07
delta_u=0.05


# In[67]:


pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u)


# In[68]:


pnp.useStandardCellBC()


# In[69]:


pnp.init()


# In[70]:


pnp.output = True
xij = pnp.solve()


# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# In[71]:


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


# In[72]:


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


# #### Potential at left and right hand side of domain

# In[73]:


(pnp.potential[0],pnp.potential[-1])


# #### Residual cation flux at interfaces

# In[74]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Residual anion flux at interfaces

# In[75]:


(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Cation concentration at interfaces

# In[76]:


(pnp.concentration[0,0],pnp.concentration[0,-1])


# #### Anion concentration at interfaces

# In[77]:


(pnp.concentration[1,0],pnp.concentration[1,-1])


# #### Equilibrium cation and anion amount

# In[78]:


( pnp.numberConservationConstraint(pnp.xij1,0,0), pnp.numberConservationConstraint(pnp.xij1,1,0) )


# #### Initial cation and anion amount

# In[79]:


( pnp.numberConservationConstraint(pnp.xi0,0,0), pnp.numberConservationConstraint(pnp.xi0,1,0) )


# #### Species conservation

# In[80]:


(pnp.numberConservationConstraint(pnp.xij1,0,
                                 pnp.numberConservationConstraint(pnp.xi0,0,0)), 
 pnp.numberConservationConstraint(pnp.xij1,1,
                                 pnp.numberConservationConstraint(pnp.xi0,1,0)) )


# ## Test case 5: 1D electrochemical cell, 0.1 mM NaCl, positive potential u = 0.05 V, 100 nm domain, 0.5 nm compact layer

# At high potentials or bulk concentrations, pure PNP systems yield unphysically high concentrations and steep gradients close to the boundary, as an ion's finite size is not accounted for.
# In addition, high gradients can lead to convergence issues. This problem can be alleviated by assuming a Stern layer (compact layer) at the interface. 
# This compact layer is parametrized by its thickness $\lambda_S$ and can be treated explicitly by prescribing a linear potential regime across the compact layer region, or by 
# the implicit parametrization of a compact layer with uniform charge density as Robin boundary conditions on the potential. 

# In[81]:


c        = [1000,1000] # high concentrations close to NaCl's solubility limit in water
delta_u  = 0.05
L        = 30e-10 # tiny gap of 3 nm
lambda_S =  5e-10 # 0.5 nm Stern layer


# In[82]:


pnp_no_compact_layer = PoissonNernstPlanckSystem(c,z,L,delta_u=delta_u, e=1e-12)


# In[83]:


pnp_with_explicit_compact_layer = PoissonNernstPlanckSystem(c,z,L, delta_u=delta_u,lambda_S=lambda_S, e=1e-12)


# In[84]:


pnp_with_implicit_compact_layer = PoissonNernstPlanckSystem(c,z,L, delta_u=delta_u,lambda_S=lambda_S, e=1e-12)


# In[85]:


pnp_no_compact_layer.useStandardCellBC()


# In[86]:


pnp_with_explicit_compact_layer.useSternLayerCellBC(implicit=False)


# In[87]:


pnp_with_implicit_compact_layer.useSternLayerCellBC(implicit=True)


# In[88]:


pnp_no_compact_layer.init()


# In[89]:


pnp_with_explicit_compact_layer.init()


# In[90]:


pnp_with_implicit_compact_layer.init()


# In[91]:


pnp_no_compact_layer.output = True
xij_no_compact_layer = pnp_no_compact_layer.solve()


# In[92]:


pnp_with_explicit_compact_layer.output = True
xij_with_explicit_compact_layer = pnp_with_explicit_compact_layer.solve()


# In[93]:


pnp_with_implicit_compact_layer.output = True
xij_with_implicit_compact_layer = pnp_with_implicit_compact_layer.solve()


# In[94]:


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


# #### Potential at left and right hand side of domain

# In[95]:


(pnp_no_compact_layer.potential[0],pnp_no_compact_layer.potential[-1])


# In[96]:


(pnp_with_explicit_compact_layer.potential[0],pnp_with_explicit_compact_layer.potential[-1])


# In[97]:


(pnp_with_implicit_compact_layer.potential[0],pnp_with_implicit_compact_layer.potential[-1])


# #### Residual cation flux at interfaces

# In[98]:


( pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0) )


# In[99]:


( pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0) )


# In[100]:


( pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0) )


# #### Residual cation flux at interfaces

# In[101]:


( pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1) )


# In[102]:


( pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1) )


# In[103]:


( pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1) )


# #### Cation concentration at interfaces

# In[104]:


(pnp_no_compact_layer.concentration[0,0],pnp_no_compact_layer.concentration[0,-1])


# In[105]:


(pnp_with_explicit_compact_layer.concentration[0,0],pnp_with_explicit_compact_layer.concentration[0,-1])


# In[106]:


(pnp_with_implicit_compact_layer.concentration[0,0],pnp_with_implicit_compact_layer.concentration[0,-1])


# #### Anion concentration at interfaces

# In[107]:


(pnp_no_compact_layer.concentration[1,0],pnp_no_compact_layer.concentration[1,-1])


# In[108]:


(pnp_with_explicit_compact_layer.concentration[1,0],pnp_with_explicit_compact_layer.concentration[1,-1])


# In[109]:


(pnp_with_implicit_compact_layer.concentration[1,0],pnp_with_implicit_compact_layer.concentration[1,-1])


# #### Equilibrium cation and anion amount

# In[110]:


( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,0) )


# In[111]:


( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,0) )


# In[112]:


( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,0) )


# #### Initial cation and anion amount

# In[113]:


( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0) )


# In[114]:


( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0) )


# In[115]:


( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0) )


# #### Species conservation

# In[116]:


(pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,
                                 pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0)), 
 pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,
                                 pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0)) )


# In[117]:


(pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,
                                 pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0)), 
 pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,
                                 pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0)) )


# In[118]:


(pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,
                                 pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0)), 
 pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,
                                 pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0)) )


# ## Sample application of 1D electrochemical cell model:

# We want to fill a gap of 3 nm between gold electrodes with 0.2 wt % NaCl aqueous solution, apply a small potential difference and generate an initial configuration for LAMMPS within a cubic box:

# In[119]:


box_Ang=np.array([50.,50.,50.]) # Angstrom


# In[120]:


box_m = box_Ang*sc.angstrom


# In[121]:


box_m


# In[122]:


vol_AngCube = box_Ang.prod() # Angstrom^3


# In[123]:


vol_mCube = vol_AngCube*sc.angstrom**3


# With a concentration of 0.2 wt %, we are close to NaCl's solubility limit in water.
# We estimate molar concentrations and atom numbers in our box:

# In[124]:


# enter number between 0 ... 0.2 
weight_concentration_NaCl = 0.2 # wt %
# calculate saline mass density g/cm³
saline_mass_density_kg_per_L  = 1 + weight_concentration_NaCl * 0.15 / 0.20 # g / cm^3, kg / L
# see https://www.engineeringtoolbox.com/density-aqueous-solution-inorganic-sodium-salt-concentration-d_1957.html


# In[125]:


saline_mass_density_g_per_L = saline_mass_density_kg_per_L*sc.kilo


# In[126]:


molar_mass_H2O = 18.015 # g / mol
molar_mass_NaCl  = 58.44 # g / mol


# In[127]:


cNaCl_M = weight_concentration_NaCl*saline_mass_density_g_per_L/molar_mass_NaCl # mol L^-1


# In[128]:


cNaCl_mM = np.round(cNaCl_M/sc.milli) # mM


# In[129]:


cNaCl_mM


# In[130]:


n_NaCl = np.round(cNaCl_mM*vol_mCube*sc.value('Avogadro constant'))


# In[131]:


n_NaCl


# In[132]:


c = [cNaCl_mM,cNaCl_mM]
z = [1,-1]
L=box_m[2]
lamda_S = 2.0e-10
delta_u  = 0.5


# In[133]:


pnp = PoissonNernstPlanckSystem(c,z,L, lambda_S=lambda_S, delta_u=delta_u, N=200, maxit=20, e=1e-6)


# In[134]:


pnp.useSternLayerCellBC()


# In[135]:


pnp.init()


# In[136]:


pnp.output = True
xij = pnp.solve()


# In[137]:


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


# #### Potential at left and right hand side of domain

# In[138]:


(pnp.potential[0],pnp.potential[-1])


# #### Residual cation flux at interfaces

# In[139]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Residual anion flux at interfaces

# In[140]:


(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Cation concentration at interfaces

# In[141]:


(pnp.concentration[0,0],pnp.concentration[0,-1])


# #### Anion concentration at interfaces

# In[142]:


(pnp.concentration[1,0],pnp.concentration[1,-1])


# #### Equilibrium cation and anion amount

# In[143]:


( pnp.numberConservationConstraint(pnp.xij1,0,0), pnp.numberConservationConstraint(pnp.xij1,1,0) )


# #### Initial cation and anion amount

# In[144]:


( pnp.numberConservationConstraint(pnp.xi0,0,0), pnp.numberConservationConstraint(pnp.xi0,1,0) )


# #### Species conservation

# In[145]:


(pnp.numberConservationConstraint(pnp.xij1,0,
                                 pnp.numberConservationConstraint(pnp.xi0,0,0)), 
 pnp.numberConservationConstraint(pnp.xij1,1,
                                 pnp.numberConservationConstraint(pnp.xi0,1,0)) )


# ## Sampling
# First, convert the physical concentration distributions into a callable "probability density":

# In[146]:


pnp.concentration.shape


# In[147]:


distributions = [interpolate.interp1d(pnp.grid,pnp.concentration[i,:]) for i in range(pnp.concentration.shape[0])]


# Normalization is not necessary here. Now we can sample the distribution of our $Na^+$ ions in z-direction.

# In[148]:


na_coordinate_sample = continuous2discrete(
    distribution=distributions[0], box=box_m, count=n_NaCl)
histx, histy, histz = get_histogram(na_coordinate_sample, box=box_m, n_bins=51)
plot_dist(histz, 'Distribution of Na+ ions in z-direction', reference_distribution=distributions[0])


# In[149]:


cl_coordinate_sample = continuous2discrete(
    distributions[1], box=box_m, count=n_NaCl)
histx, histy, histz = get_histogram(cl_coordinate_sample, box=box_m, n_bins=51)
plot_dist(histx, 'Distribution of Cl- ions in x-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box[0])
plot_dist(histy, 'Distribution of Cl- ions in y-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box[1])
plot_dist(histz, 'Distribution of Cl- ions in z-direction', reference_distribution=distributions[1])


# ## Write to file
# To visualize our sampled coordinates, we utilize ASE to export it to some standard format, i.e. .xyz or LAMMPS data file.
# ASE speaks Ångström per default, thus we convert SI units:

# In[150]:


sample_size = int(n_NaCl)


# In[151]:


sample_size


# In[152]:


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


# In[153]:


# LAMMPS data format, units 'real', atom style 'full'
# before ASE 3.19.0b1, ASE had issues with exporting atom style 'full' in LAMMPS data file format, so do not expect this line to work for older ASE versions
ase.io.write('NaCl_c_4_M_u_0.5_V_box_5x5x10nm_lambda_S_2_Ang.lammps',system,format='lammps-data',units="real",atom_style='full')

