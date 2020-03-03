#!/usr/bin/env python
# coding: utf-8

# # pnp, c2d, stericify
# 
# *Johannes Hörmann, 2020*
# 
# from continuous electrochemical double layer theory to discrete coordinate sets

# In[1]:


# for dynamic module reload during testing, code modifications take immediate effect
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# stretching notebook width across whole window
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[4]:


import fenics as fn


# In[5]:


# basics
import logging
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

# sampling
from scipy import interpolate
from matscipy.electrochemistry import continuous2discrete
from matscipy.electrochemistry import get_histogram
from matscipy.electrochemistry.utility import plot_dist

# electrochemistry basics
from matscipy.electrochemistry import debye, ionic_strength

# Poisson-Bolzmann distribution
from matscipy.electrochemistry.poisson_boltzmann_distribution import gamma, potential, concentration, charge_density

# Poisson-Nernst-Planck solver
from matscipy.electrochemistry import PoissonNernstPlanckSystem
from matscipy.electrochemistry.poisson_nernst_planck_solver_logc import PoissonNernstPlanckSystemLogC

from scipy.optimize import minimize

# 3rd party file output
import ase
import ase.io


# In[12]:


mesh = fn.UnitSquareMesh(8,8)

V = fn.FunctionSpace(mesh, 'P', 1)

u_D = fn.Expression('1 +x[0]*x[0] + 2*x[1]*x[1]',degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = fn.DirichletBC(V, u_D, boundary)

u = fn.TrialFunction(V)
v = fn.TestFunction(V)
f = fn.Constant(-6.0)
a = fn.dot(fn.grad(u), fn.grad(v))*fn.dx
L = f*v*fn.dx
# Compute solution
u = fn.Function(V)
fn.solve(a == L, u, bc)


# In[13]:


fn.plot(u)
fn.plot(mesh)


# In[22]:


import sympy as sym


# In[23]:


def q(u):
    return 1+u**2


# In[24]:


x, y = sym.symbols('x[0], x[1]')


# In[25]:


u = 1 + x + 2*y


# In[26]:


f = - sym.diff(q(u)*sym.diff(u,x),x) - sym.diff(q(u)*sym.diff(u,y),y)


# In[27]:


f = sym.simplify(f)


# In[30]:


u_code = sym.printing.ccode(u)


# In[31]:


u_code


# In[32]:


f_code = sym.printing.ccode(f)


# In[33]:


f_code


# In[34]:


mesh = fn.UnitSquareMesh(8,8)
V = fn.FunctionSpace(mesh, 'P', 1)

u_D = fn.Expression(u_code,degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = fn.DirichletBC(V, u_D, boundary)

u = fn.Function(V)
v = fn.TestFunction(V)
f = fn.Expression(f_code,degree=1)
F = q(u)*fn.dot(fn.grad(u),fn.grad(v))*fn.dx - f*v*fn.dx
a = fn.dot(fn.grad(u), fn.grad(v))*fn.dx
fn.solve(F == 0, u, bc)


# In[35]:


fn.plot(u)
fn.plot(mesh)


# In[ ]:





# In[258]:


# Test case parameters   oö
c=[1.0, 1.0]
z=[ 1, -1]
L=20e-8 # 200 nm
a=28e-9 # 28 x 28 nm area
# delta_u=0.05
# delta_u=1.2 # V
delta_u=0.2 # V


# In[259]:


# define desired system
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u,e=1e-15, solver="hybr", options={'xtol':1e-15})
# constructor takes keyword arguments
#   c=array([0.1, 0.1]), z=array([ 1, -1]), L=1e-07, T=298.15, delta_u=0.05, relative_permittivity=79, vacuum_permittivity=8.854187817620389e-12, R=8.3144598, F=96485.33289
# with default values set for 0.1 mM NaCl aqueous solution across 100 nm  and 0.05 V potential drop


# In[260]:


pnp.c_scaled


# In[261]:


pnp.delta_u_scaled


# In[264]:


pnp.L_scaled


# In[265]:


mesh = fn.IntervalMesh(1000,0,pnp.L_scaled)


# In[266]:


fn.plot(mesh)


# In[267]:


P1 = fn.FiniteElement('P',fn.interval,1)


# In[268]:


H = fn.MixedElement([P1,P1,P1])


# In[269]:


W = fn.FunctionSpace(mesh,element)


# In[270]:


w = fn.Function(W)


# In[271]:


u,*p = fn.split(w)


# In[272]:


v,*q = fn.TestFunctions(W)


# In[273]:


# rho = pnp.z[0]*p[0] + pnp.z[1]*p[1]

# nernst_planck_0 = fn.dot(- fn.grad(p[0]) - pnp.z[0]*p[0]*fn.grad(u), fn.grad(q[0]) )*fn.dx

# nernst_planck_1 = fn.dot(- fn.grad(p[1]) - pnp.z[1]*p[1]*fn.grad(u), fn.grad(q[1]) )*fn.dx

# F = poisson + nernst_planck_0 + nernst_planck_1


# In[274]:


rho = 0
for i in range(2):
    rho += pnp.z[i]*p[i]


# In[275]:


source = - 0.5 * rho * v *fn.dx


# In[276]:


laplace = fn.dot(fn.grad(u), fn.grad(v))*fn.dx


# In[277]:


poisson  = laplace + source


# In[278]:


nernst_planck = 0
for i in range(pnp.M):
    nernst_planck += fn.dot(- fn.grad(p[i]) - pnp.z[i]*p[i]*fn.grad(u), fn.grad(q[i]) )*fn.dx


# In[279]:


F = poisson + nernst_planck


# In[280]:


u_L = fn.Constant(pnp.delta_u_scaled)


# In[281]:


def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and fn.near(x[0], 0, tol)


# In[282]:


u_R = fn.Constant(0)


# In[283]:


def boundary_R(x, on_boundary):
    tol = 1E-14
    return on_boundary and fn.near(x[0], pnp.L_scaled, tol)


# In[284]:


bcU_L = fn.DirichletBC(W.sub(0),u_L,boundary_L)

bcU_R = fn.DirichletBC(W.sub(0),u_R,boundary_R)

p_R = pnp.c_scaled

bcP_R = [
    fn.DirichletBC(W.sub(1),p_R[0],boundary_R),
    fn.DirichletBC(W.sub(2),p_R[1],boundary_R) ]

bcs = [bcU_L,bcU_R,*bcP_R]


# In[285]:


bcs


# In[286]:


fn.solve(F==0,w,bcs)


# In[205]:


coordinates = mesh.coordinates()


# In[206]:


coordinates.shape


# In[207]:


wij = np.array([ w(c) for c in coordinates ]).T
    


# In[208]:


fn.plot(u)


# In[209]:


fn.plot(p[0])


# In[210]:


fn.plot(p[1])


# In[304]:


from matscipy.electrochemistry.poisson_nernst_planck_solver_fenics import PoissonNernstPlanckSystemFEniCS


# In[308]:


# define desired system
pnp = PoissonNernstPlanckSystemFEniCS(c, z, L, delta_u=delta_u,N=1000)

pnp.useStandardInterfaceBC()

uij, nij, _ = pnp.solve()


# In[314]:


plt.plot(nij[0,:])


# In[297]:


pnp.mesh = fn.IntervalMesh(pnp.N,pnp.x0_scaled,pnp.x1_scaled)

# construct test function space
P1 = fn.FiniteElement('P',fn.interval,1)

# build function space
H = fn.MixedElement([P1]*(pnp.M+1))
pnp.W = fn.FunctionSpace(pnp.mesh,H)


# In[298]:


w = fn.Function(pnp.W)
# u represents voltage , p concentrations
u, *p = fn.split(w)
# v and q represent respetive test functions
v, *q = fn.TestFunctions(pnp.W)


# In[299]:


# TODO: implement Neumann and Robin BC:
rho = 0
for i in range(pnp.M):
    rho += pnp.z[i]*p[i]

source = - 0.5 * rho * v * fn.dx

laplace = fn.dot(fn.grad(u), fn.grad(v))*fn.dx

poisson = laplace + source

nernst_planck = 0
for i in range(pnp.M):
    nernst_planck += fn.dot(
            - fn.grad(p[i]) - pnp.z[i]*p[i]*fn.grad(u), fn.grad(q[i])
        )*fn.dx

F = poisson + nernst_planck
# for i in range pnp.M:


# In[300]:


pnp.boundary_conditions = [
    fn.DirichletBC(pnp.W.sub(0),pnp.delta_u_scaled,boundary_L),
    fn.DirichletBC(pnp.W.sub(0),0,boundary_R),
    fn.DirichletBC(pnp.W.sub(1),pnp.c_scaled[0],boundary_R),
    fn.DirichletBC(pnp.W.sub(2),pnp.c_scaled[1],boundary_R) ]


# In[301]:


pnp.W


# In[302]:


fn.solve(F==0,w,pnp.boundary_conditions)


# In[303]:


pnp.solve()


# In[213]:


fn.DirichletBC(
    pnp.W.sub(0), pnp.u0, pnp.boundary_L)


# In[186]:


PoissonNernstPlanckSystemFEniCS()


# In[112]:


u


# In[113]:


fn.plot(u)


# In[122]:


W


# In[125]:


u


# In[127]:


V


# In[129]:


coordinates = mesh.coordinates()


# In[135]:


fn.interpolate(u,W)


# In[128]:


fn.interpolate(u,V)


# In[ ]:


u_D 

def boundary(x, on_boundary):
    return on_boundary


# In[ ]:


mesh = fn.UnitSquareMesh(8,8)
V = fn.FunctionSpace(mesh, 'P', 1)

u_D = fn.Expression(u_code,degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = fn.DirichletBC(V, u_D, boundary)

u = fn.Function(V)
v = fn.TestFunction(V)
f = fn.Expression(f_code,degree=1)
F = q(u)*fn.dot(fn.grad(u),fn.grad(v))*fn.dx - f*v*fn.dx
a = fn.dot(fn.grad(u), fn.grad(v))*fn.dx
fn.solve(F == 0, u, bc)


# In[ ]:





# In[10]:


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


# In[11]:


# Test 1
logging.info("Root logger")


# In[12]:


# Test 2
logger.info("Root Logger")


# In[13]:


# Debug Test
logging.debug("Root logger")


# In[14]:


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# # Fenics trial

# In[23]:


from fenics import *


# ## Test case 1: PNP interface system, 0.1 mM NaCl, positive potential u = 0.05 V

# In[20]:


# Test case parameters   oö
c=[1000.0, 1000.0]
z=[ 1, -1]
L=20e-9 # 20 nm
a=28e-9 # 28 x 28 nm area
# delta_u=0.05
# delta_u=1.2 # V
delta_u=1.0 # V


# In[21]:


# expected number of ions in volume:
V = L*a**2
Nref = c[0]*V*sc.Avogadro


# In[187]:


# define desired system
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u,e=1e-15, solver="hybr", options={'xtol':1e-15})
# constructor takes keyword arguments
#   c=array([0.1, 0.1]), z=array([ 1, -1]), L=1e-07, T=298.15, delta_u=0.05, relative_permittivity=79, vacuum_permittivity=8.854187817620389e-12, R=8.3144598, F=96485.33289
# with default values set for 0.1 mM NaCl aqueous solution across 100 nm  and 0.05 V potential drop

pnp.useStandardCellBC()

pnp.init()

# pnp.output = True # let's Newton solver display convergence plots
uij, nij, lamj = pnp.solve()


# In[19]:


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


# In[210]:


# Test case parameters   oö
c=[1000.0, 1000.0]
z=[ 1, -1]
L=20e-9 # 20 nm
a=28e-9 # 28 x 28 nm area
# delta_u=0.05
delta_u=1.1 # V


# In[211]:


# define desired system
pnp_logc = PoissonNernstPlanckSystemLogC(c, z, L, delta_u=delta_u,e=1e-12) #, solver="hybr", options={'xtol':1e-16})
# constructor takes keyword arguments
#   c=array([0.1, 0.1]), z=array([ 1, -1]), L=1e-07, T=298.15, delta_u=0.05, relative_permittivity=79, vacuum_permittivity=8.854187817620389e-12, R=8.3144598, F=96485.33289
# with default values set for 0.1 mM NaCl aqueous solution across 100 nm  and 0.05 V potential drop


# In[212]:


pnp_logc.useStandardCellBC()


# In[213]:


pnp_logc.init()


# In[214]:


pnp_logc.ui0 = pnp.uij
pnp_logc.Ni0 = np.log(pnp.nij)


# In[215]:


# pnp.output = True # let's Newton solver display convergence plots
uij, Nij, lamj = pnp_logc.solve()


# In[216]:


# analytic Poisson-Boltzmann distribution and numerical solution to full Poisson-Nernst-Planck system
x = np.linspace(0,L,100)

deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[16,10])
ax1.set_xlabel('z [nm]')
ax1.plot(pnp_logc.grid/sc.nano, pnp_logc.potential, marker='', color='tab:red', label='potential, PNP', linewidth=1, linestyle='-')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')
ax2.plot(pnp_logc.grid/sc.nano, pnp_logc.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')

ax2.plot(pnp_logc.grid/sc.nano, pnp_logc.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')
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
ax3.plot(pnp_logc.grid/sc.nano, pnp_logc.charge_density, label='charge density, PNP', color='grey', linewidth=1, linestyle='-')

ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')
ax4.semilogy(pnp_logc.grid/sc.nano, pnp_logc.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')
ax4.semilogy(pnp_logc.grid/sc.nano, pnp_logc.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')

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


# ## Interface

# In[397]:


# Test case parameters
c=[1, 1]
z=[ 1, -1] 
L=1e-7
delta_u=1.0


# In[398]:


# expected number of ions in volume:
V = L*a**2
Nref = c[0]*V*sc.Avogadro


# In[399]:


# define desired system
pnp = PoissonNernstPlanckSystem(c, z, L, delta_u=delta_u,e=1e-12,maxit=20, N=200)#, solver="hybr", options={'xtol':1e-16},maxit=1)
# constructor takes keyword arguments
#   c=array([0.1, 0.1]), z=array([ 1, -1]), L=1e-07, T=298.15, delta_u=0.05, relative_permittivity=79, vacuum_permittivity=8.854187817620389e-12, R=8.3144598, F=96485.33289
# with default values set for 0.1 mM NaCl aqueous solution across 100 nm  and 0.05 V potential drop


# In[400]:


pnp.useStandardInterfaceBC()


# In[401]:


pnp.init()


# In[402]:


# pnp.output = True # let's Newton solver display convergence plots
uij, nij, lamj = pnp.solve()


# In[403]:


delta_u=1.2


# In[404]:


# define desired system
pnp_logc = PoissonNernstPlanckSystemLogC(c, z, L, delta_u=delta_u,e=1e-12,maxit=20, N=200)#, solver="hybr", options={'xtol':1e-12})
# constructor takes keyword arguments
#   c=array([0.1, 0.1]), z=array([ 1, -1]), L=1e-07, T=298.15, delta_u=0.05, relative_permittivity=79, vacuum_permittivity=8.854187817620389e-12, R=8.3144598, F=96485.33289
# with default values set for 0.1 mM NaCl aqueous solution across 100 nm  and 0.05 V potential drop


# In[405]:


pnp_logc.useStandardInterfaceBC()


# In[406]:


pnp_logc.init()


# In[407]:


pnp_logc.ui0 = pnp.uij


# In[408]:


pnp_logc.Ni0 = np.log(pnp.nij)


# In[409]:


# pnp.output = True # let's Newton solver display convergence plots
uij, Nij, lamj = pnp_logc.solve()


# In[410]:


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
ax1.plot(pnp_logc.grid/sc.nano, pnp_logc.potential, marker='', color='tab:red', label='potential, PNP, logc', linewidth=1, linestyle=':')

ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='bulk concentration', color='grey', linestyle=':')
ax2.plot(x/sc.nano, C[0], marker='', color='bisque', label='Na+, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[0], marker='', color='tab:orange', label='Na+, PNP', linewidth=2, linestyle='-')
ax2.plot(pnp_logc.grid/sc.nano, pnp_logc.concentration[0], marker='', color='tab:cyan', label='Na+, PNP, log c', linewidth=2, linestyle=':')

ax2.plot(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax2.plot(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')
ax2.plot(pnp_logc.grid/sc.nano, pnp_logc.concentration[1], marker='', color='tab:pink', label='Cl-, PNP, log c', linewidth=2, linestyle=':')

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
ax4.semilogy(pnp_logc.grid/sc.nano, pnp_logc.concentration[0], marker='', color='tab:cyan', label='Na+, PNP, log c', linewidth=2, linestyle=':')

ax4.semilogy(x/sc.nano, C[1], marker='', color='lightskyblue', label='Cl-, PB',linestyle='--')
ax4.semilogy(pnp.grid/sc.nano, pnp.concentration[1], marker='', color='tab:blue', label='Cl-, PNP', linewidth=2, linestyle='-')
ax4.semilogy(pnp_logc.grid/sc.nano, pnp_logc.concentration[1], marker='', color='tab:pink', label='Cl-, PNP, log c', linewidth=2, linestyle=':')

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


# In[366]:


pnp_logc.Nij[1,0]


# In[83]:


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


# In[95]:


F = pnp_logc.G(pnp_logc.xi0)


# In[96]:


H = pnp.G(pnp.xi0)


# In[97]:


from matscipy.electrochemistry.poisson_nernst_planck_solver import jacobian


# In[98]:


J = jacobian(pnp.G, pnp.xi0)


# In[99]:


np.linalg.matrix_rank(J)


# In[100]:


K = jacobian(pnp_logc.G, pnp_logc.xi0)


# In[101]:


np.linalg.matrix_rank(K)


# In[70]:


J


# In[71]:


K


# In[72]:


J-K


# In[98]:


from matplotlib.pyplot import imshow


# In[110]:


(J-K)[-1,:]


# In[99]:


imshow(J-K)


# In[116]:


pnp_logc.Ni0.shape


# In[137]:


imshow(J[:pnp.Ni,:pnp.Ni],extent=(0,100,0,100))


# In[138]:


imshow((J - K) != 0)


# In[145]:


imshow((J - K)[pnp.Ni:2*pnp.Ni+5,pnp.Ni:2*pnp.Ni+5] != 0)


# In[119]:


imshow(np.where(J - K))


# In[78]:


np.linalg.matrix_rank(F)


# In[79]:


np.linalg.matrix_rank(H)


# In[ ]:





# ### Validation: Analytical half-space solution & Numerical finite-size PNP system

# #### Potential at left and right hand side of domain

# In[502]:


(pnp.potential[0],pnp.potential[-1])


# #### Residual cation flux at interfaces

# In[503]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Residual anion flux at interfaces

# In[504]:


(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Cation concentration at interfaces

# In[505]:


(pnp.concentration[0,0],pnp.concentration[0,-1])


# #### Anion concentration at interfaces

# In[506]:


(pnp.concentration[1,0],pnp.concentration[1,-1])


# #### Equilibrium cation and anion amount

# In[507]:


( pnp.numberConservationConstraint(pnp.xij1,0,0), pnp.numberConservationConstraint(pnp.xij1,1,0) )


# #### Initial cation and anion amount

# In[508]:


( pnp.numberConservationConstraint(pnp.xi0,0,0), pnp.numberConservationConstraint(pnp.xi0,1,0) )


# #### Species conservation

# In[509]:


(pnp.numberConservationConstraint(pnp.xij1,0,
                                 pnp.numberConservationConstraint(pnp.xi0,0,0)), 
 pnp.numberConservationConstraint(pnp.xij1,1,
                                 pnp.numberConservationConstraint(pnp.xi0,1,0)) )


# ### Different Stern layers

# In[556]:


# Test case parameters
c=[1000.0, 1000.0]
z=[ 1, -1]
L=20e-9 # 20 nm
a=28e-9 # 28 x 28 nm area
#delta_u=0.05
delta_u=0.8 # V


# In[557]:


# expected number of ions in volume:
V = L*a**2
c[0]*V*sc.Avogadro


# In[558]:


lambda_S =  8e-10 # 0.8 nm Stern layer


# In[559]:


pnp_no_compact_layer = PoissonNernstPlanckSystem(c,z,L,delta_u=delta_u, e=1e-12, solver="hybr", options={'xtol':1e-16})


# In[560]:


pnp_with_explicit_compact_layer = PoissonNernstPlanckSystem(c,z,L, delta_u=delta_u,lambda_S=lambda_S, e=1e-12, solver="hybr", options={'xtol':1e-16})


# In[561]:


pnp_with_implicit_compact_layer = PoissonNernstPlanckSystem(c,z,L, delta_u=delta_u,lambda_S=lambda_S, e=1e-12, solver="hybr", options={'xtol':1e-16})


# In[562]:


pnp_no_compact_layer.useStandardCellBC()


# In[563]:


pnp_with_explicit_compact_layer.useSternLayerCellBC(implicit=False)


# In[564]:


pnp_with_implicit_compact_layer.useSternLayerCellBC(implicit=True)


# In[565]:


pnp_no_compact_layer.init()


# In[566]:


pnp_with_explicit_compact_layer.init()


# In[567]:


pnp_with_implicit_compact_layer.init()


# In[568]:


xij_no_compact_layer = pnp_no_compact_layer.solve()


# In[569]:


xij_with_explicit_compact_layer = pnp_with_explicit_compact_layer.solve()


# In[570]:


xij_with_implicit_compact_layer = pnp_with_implicit_compact_layer.solve()


# In[571]:


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
ax2.legend(loc='center rigcht', bbox_to_anchor=(-0.1,0.5),  fontsize=12)
ax3.legend(loc='lower right',  bbox_to_anchor=(-0.1,-0.02), fontsize=12)

fig.tight_layout()
plt.show()


# In[555]:


#### Potential at left and right hand side of domain

logger.info("Potential at left and right hand side of domain")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.potential[0],pnp_no_compact_layer.potential[-1]))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.potential[0],pnp_with_explicit_compact_layer.potential[-1]))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.potential[0],pnp_with_implicit_compact_layer.potential[-1]))


#### Residual cation flux at interfaces
logger.info("")
logger.info("Residual cation flux at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0) ))

#### Residual anion flux at interfaces
logger.info("")
logger.info("Residual anion flux at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format( pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1) ))


#### Cation concentration at interfaces
logger.info("")
logger.info("Cation concentration at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.concentration[0,0],pnp_no_compact_layer.concentration[0,-1]))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.concentration[0,0],pnp_with_explicit_compact_layer.concentration[0,-1]))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.concentration[0,0],pnp_with_implicit_compact_layer.concentration[0,-1]))

#### Anion concentration at interfaces
logger.info("")
logger.info("Anion concentration at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.concentration[1,0],pnp_no_compact_layer.concentration[1,-1]))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.concentration[1,0],pnp_with_explicit_compact_layer.concentration[1,-1]))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.concentration[1,0],pnp_with_implicit_compact_layer.concentration[1,-1]))

#### Equilibrium cation and anion amount
logger.info("")
logger.info("Equilibrium cation and anion amount")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,0) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,0) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,0) ))

#### Initial cation and anion amount
logger.info("")
logger.info("Equilibrium cation and anion amount")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0) ))

#### Species conservation
logger.info("")
logger.info("Species conservation (cations, anions)")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(
    pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,
                                 pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0)), 
    pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,
                                 pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0)) ) )
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( 
    pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,
                                 pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0)), 
    pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,
                                 pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0)) ) )
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(
    pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,
                                 pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0)), 
    pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,
                                 pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0)) ) )



# ### Own and scipy solver

# In[121]:


# Test case parameters
c=[1000.0, 1000.0]
z=[ 1, -1]
L=20e-9 # 20 nm
a=28e-9 # 28 x 28 nm area
# delta_u=0.05
delta_u=0.5 # V


# In[122]:


# expected number of ions in volume:
V = L*a**2
c[0]*V*sc.Avogadro


# In[123]:


lambda_S =  8e-10 # 0.5 nm Stern layer


# In[124]:


pnp_no_compact_layer = PoissonNernstPlanckSystem(c,z,L,delta_u=delta_u, e=1e-12)


# In[125]:


pnp_no_compact_layer.useStandardCellBC()


# In[126]:


pnp_no_compact_layer.init()


# In[127]:


xij_no_compact_layer = pnp_no_compact_layer.solve()


# In[531]:


np.finfo('float64').resolution


# In[129]:


pnp_no_compact_layer_scipy = PoissonNernstPlanckSystem(c,z,L,delta_u=delta_u, e=1e-12,solver="L-BFGS-B",
                                                       options= {'gtol':1.e-5,'maxfun':1e6, 'maxiter':20,'disp':True,'eps':1e-4})


# In[205]:


pnp_no_compact_layer_scipy = PoissonNernstPlanckSystem(c,z,L,delta_u=delta_u, e=1e-12,solver="hybr")


# In[206]:


pnp_no_compact_layer_scipy.useStandardCellBC()


# In[207]:


pnp_no_compact_layer_scipy.init()


# In[208]:


# finds sol: hybr, linearmixing
# does not find or too slow with standard paremeters: broyden1, broyden2, anderson, lm, krylov, diagbroyden, excitingmixing, df-sane


# In[209]:


xij_no_compact_layer_scipy = pnp_no_compact_layer_scipy.solve()


# In[532]:


pnp_no_compact_layer_scipy.xi0


# In[533]:


x = np.linspace(0,L,100)

deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[18,10])

# 1 - potentials
ax1.axvline(x=deb/sc.nano, label='Debye Length', color='grey', linestyle=':')
ax1.plot(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.potential, marker='', color='tab:red', label='potential, without compact layer, scipy optimizer', linewidth=1, linestyle=':')

# 2 - conencentratiosn
ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')

ax2.plot(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer, scipy', linewidth=2, linestyle=':')
ax2.plot(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer, scipy', linewidth=2, linestyle=':')


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


# 4 - concentrations, semi log
ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')

ax4.semilogy(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer', linewidth=2, linestyle=':')
ax4.semilogy(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer', linewidth=2, linestyle=':')


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


# In[534]:


x = np.linspace(0,L,100)

deb = debye(c, z)

fig, (ax1,ax4) = plt.subplots(nrows=2,ncols=1,figsize=[18,10])

# 1 - potentials
ax1.axvline(x=deb/sc.nano, label='Debye Length', color='grey', linestyle=':')
ax1.plot(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.potential, marker='', color='tab:red', label='potential, without compact layer', linewidth=1, linestyle='-')
ax1.plot(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.potential, marker='', color='tab:red', label='potential, without compact layer, scipy optimizer', linewidth=1, linestyle=':')

# 2 - conencentratiosn
ax2 = ax1.twinx()
ax2.plot(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')

ax2.plot(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer', linewidth=2, linestyle='-')
ax2.plot(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer', linewidth=2, linestyle='-')
ax2.plot(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer, scipy', linewidth=2, linestyle=':')
ax2.plot(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer, scipy', linewidth=2, linestyle=':')


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


# 4 - concentrations, semi log
ax4.semilogy(x/sc.nano, np.ones(x.shape)*c[0], label='average concentration', color='grey', linestyle=':')

ax4.semilogy(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer', linewidth=2, linestyle='-')
ax4.semilogy(pnp_no_compact_layer.grid/sc.nano, pnp_no_compact_layer.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer', linewidth=2, linestyle='-')
ax4.semilogy(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[0], marker='', color='tab:orange', label='Na+, without compact layer', linewidth=2, linestyle=':')
ax4.semilogy(pnp_no_compact_layer_scipy.grid/sc.nano, pnp_no_compact_layer_scipy.concentration[1], marker='', color='tab:blue', label='Cl-, without compact layer', linewidth=2, linestyle=':')


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


# In[535]:


from scipy import interpolate, integrate


# In[536]:


D_no_compact_layer = [ interpolate.interp1d(pnp_no_compact_layer.grid,c) for c in pnp_no_compact_layer.concentration ]


# In[537]:


integral_no_compact_layer = [ integrate.quad( D, pnp_no_compact_layer.grid[0] , pnp_no_compact_layer.grid[-1] )[0] for D in D_no_compact_layer]


# In[93]:


cave_no_compact_layer = [ integral / L for integral in integral_no_compact_layer ]


# In[94]:


cave_no_compact_layer # mM


# In[95]:


D_with_explicit_compact_layer = [ interpolate.interp1d(pnp_with_explicit_compact_layer.grid,c) for c in pnp_with_explicit_compact_layer.concentration ]


# In[96]:


integral_with_implicict_compact_layer = [ integrate.quad( D, pnp_with_explicit_compact_layer.grid[0] , pnp_with_explicit_compact_layer.grid[-1] )[0] for D in D_with_explicit_compact_layer]


# In[97]:


cave_with_explicit_compact_layer = [ integral / L for integral in integral_with_implicict_compact_layer ]


# In[98]:


cave_with_explicit_compact_layer # mM


# In[99]:


D_with_implicit_compact_layer = [ interpolate.interp1d(pnp_with_implicit_compact_layer.grid,c) for c in pnp_with_implicit_compact_layer.concentration ]


# In[100]:


integral_with_implicict_compact_layer = [ integrate.quad( D, pnp_with_implicit_compact_layer.grid[0] , pnp_with_implicit_compact_layer.grid[-1] )[0] for D in D_with_implicit_compact_layer]


# In[101]:


cave_with_implicit_compact_layer = [ integral / (L-2*lambda_S) for integral in integral_with_implicict_compact_layer ]


# In[102]:


cave_with_implicit_compact_layer # mM


# In[230]:


#### Potential at left and right hand side of domain

logger.info("Potential at left and right hand side of domain")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.potential[0],pnp_no_compact_layer.potential[-1]))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.potential[0],pnp_with_explicit_compact_layer.potential[-1]))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.potential[0],pnp_with_implicit_compact_layer.potential[-1]))


#### Residual cation flux at interfaces
logger.info("")
logger.info("Residual cation flux at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,0) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,0) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,0) ))

#### Residual anion flux at interfaces
logger.info("")
logger.info("Residual anion flux at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format( pnp_no_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1), pnp_no_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_no_compact_layer.xij1,1) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_explicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1), pnp_with_explicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_explicit_compact_layer.xij1,1) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_implicit_compact_layer.leftControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1), pnp_with_implicit_compact_layer.rightControlledVolumeSchemeFluxBC(pnp_with_implicit_compact_layer.xij1,1) ))


#### Cation concentration at interfaces
logger.info("")
logger.info("Cation concentration at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.concentration[0,0],pnp_no_compact_layer.concentration[0,-1]))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.concentration[0,0],pnp_with_explicit_compact_layer.concentration[0,-1]))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.concentration[0,0],pnp_with_implicit_compact_layer.concentration[0,-1]))

#### Anion concentration at interfaces
logger.info("")
logger.info("Anion concentration at interfaces")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(pnp_no_compact_layer.concentration[1,0],pnp_no_compact_layer.concentration[1,-1]))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_explicit_compact_layer.concentration[1,0],pnp_with_explicit_compact_layer.concentration[1,-1]))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(pnp_with_implicit_compact_layer.concentration[1,0],pnp_with_implicit_compact_layer.concentration[1,-1]))

#### Equilibrium cation and anion amount
logger.info("")
logger.info("Equilibrium cation and anion amount")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,0) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,0) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,0) ))

#### Initial cation and anion amount
logger.info("")
logger.info("Equilibrium cation and anion amount")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format( pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0), pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0) ))
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0), pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0) ))
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format( pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0), pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0) ))

#### Species conservation
logger.info("")
logger.info("Species conservation (cations, anions)")
logger.info("No compact layer:       {: 8.4e}, {: 8.4e}".format(
    pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,0,
                                 pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,0,0)), 
    pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xij1,1,
                                 pnp_no_compact_layer.numberConservationConstraint(pnp_no_compact_layer.xi0,1,0)) ) )
logger.info("Explicit compact layer: {: 8.4e}, {: 8.4e}".format( 
    pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,0,
                                 pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,0,0)), 
    pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xij1,1,
                                 pnp_with_explicit_compact_layer.numberConservationConstraint(pnp_with_explicit_compact_layer.xi0,1,0)) ) )
logger.info("Implicit compact layer: {: 8.4e}, {: 8.4e}".format(
    pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,0,
                                 pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,0,0)), 
    pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xij1,1,
                                 pnp_with_implicit_compact_layer.numberConservationConstraint(pnp_with_implicit_compact_layer.xi0,1,0)) ) )



# In[389]:


pnp_with_explicit_compact_layer.Ni


# In[390]:


np.sum(pnp_with_explicit_compact_layer.nij[0]*pnp_with_explicit_compact_layer.dx)


# In[391]:


pnp_with_explicit_compact_layer.L_scaled


# In[392]:


np.sum(pnp_with_explicit_compact_layer.nij[0]*pnp_with_explicit_compact_layer.dx)  / (pnp_with_explicit_compact_layer.L_scaled+pnp_with_explicit_compact_layer.dx)


# In[393]:


dx = np.mean((np.roll(pnp_with_explicit_compact_layer.grid,-1)-pnp_with_explicit_compact_layer.grid)[:-1])


# In[394]:


np.sum(pnp_with_explicit_compact_layer.concentration[0])/pnp_with_explicit_compact_layer.Ni


# In[395]:


pnp_with_explicit_compact_layer.c_unit*V*sc.Avogadro


# In[396]:


Nref


# In[397]:


np.sum(pnp_with_explicit_compact_layer.nij[0]*pnp_with_explicit_compact_layer.dx) * pnp_with_explicit_compact_layer.N / pnp_with_explicit_compact_layer.Ni


# In[ ]:





# In[398]:


D_with_explicit_compact_layer = [ interpolate.interp1d(pnp_with_explicit_compact_layer.grid,c) for c in pnp_with_explicit_compact_layer.concentration ]


# In[399]:


plt.plot( pnp_with_explicit_compact_layer.grid, D_with_explicit_compact_layer[0](pnp_with_explicit_compact_layer.grid) )


# In[444]:


integral_with_explicit_compact_layer = [ 
    integrate.quad( D, pnp_with_explicit_compact_layer.grid[0] , pnp_with_explicit_compact_layer.grid[-1], limit=1000, epsrel=1e-12 )[0] for D in D_with_explicit_compact_layer]


# In[447]:


integral_with_explicit_compact_layer[0]


# In[446]:


cave_with_explicit_compact_layer = [ integral / L for integral in integral_with_explicit_compact_layer ]


# In[430]:


cave_with_explicit_compact_layer # mM


# In[441]:


integral_with_explicit_compact_layer[0]


# In[448]:


L


# In[449]:


pnp_with_explicit_compact_layer.grid[-1]


# In[457]:


L/dx


# In[470]:


np.sum(D_with_explicit_compact_layer[0](pnp_with_explicit_compact_layer.grid)*dx)


# ## Sample application of 1D electrochemical cell model:

# We want to fill a gap of 3 nm between gold electrodes with 0.2 wt % NaCl aqueous solution, apply a small potential difference and generate an initial configuration for LAMMPS within a cubic box:

# In[404]:


box_Ang=np.array([50.,50.,50.]) # Angstrom


# In[405]:


box_m = box_Ang*sc.angstrom


# In[109]:


box_m


# In[110]:


vol_AngCube = box_Ang.prod() # Angstrom^3


# In[111]:


vol_mCube = vol_AngCube*sc.angstrom**3


# With a concentration of 0.2 wt %, we are close to NaCl's solubility limit in water.
# We estimate molar concentrations and atom numbers in our box:

# In[112]:


# enter number between 0 ... 0.2 
weight_concentration_NaCl = 0.2 # wt %
# calculate saline mass density g/cm³
saline_mass_density_kg_per_L  = 1 + weight_concentration_NaCl * 0.15 / 0.20 # g / cm^3, kg / L
# see https://www.engineeringtoolbox.com/density-aqueous-solution-inorganic-sodium-salt-concentration-d_1957.html


# In[113]:


saline_mass_density_g_per_L = saline_mass_density_kg_per_L*sc.kilo


# In[114]:


molar_mass_H2O = 18.015 # g / mol
molar_mass_NaCl  = 58.44 # g / mol


# In[115]:


cNaCl_M = weight_concentration_NaCl*saline_mass_density_g_per_L/molar_mass_NaCl # mol L^-1


# In[116]:


cNaCl_mM = np.round(cNaCl_M/sc.milli) # mM


# In[117]:


cNaCl_mM


# In[118]:


n_NaCl = np.round(cNaCl_mM*vol_mCube*sc.value('Avogadro constant'))


# In[119]:


n_NaCl


# In[120]:


c = [cNaCl_mM,cNaCl_mM]
z = [1,-1]
L=box_m[2]
lamda_S = 2.0e-10
delta_u  = 0.5


# In[121]:


pnp = PoissonNernstPlanckSystem(c,z,L, lambda_S=lambda_S, delta_u=delta_u, N=200, maxit=20, e=1e-6)


# In[122]:


pnp.useSternLayerCellBC()


# In[123]:


pnp.init()


# In[124]:


pnp.output = True
xij = pnp.solve()


# In[125]:


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

# In[126]:


(pnp.potential[0],pnp.potential[-1])


# #### Residual cation flux at interfaces

# In[127]:


( pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,0), pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Residual anion flux at interfaces

# In[128]:


(pnp.leftControlledVolumeSchemeFluxBC(pnp.xij1,1),  pnp.rightControlledVolumeSchemeFluxBC(pnp.xij1,0) )


# #### Cation concentration at interfaces

# In[129]:


(pnp.concentration[0,0],pnp.concentration[0,-1])


# #### Anion concentration at interfaces

# In[130]:


(pnp.concentration[1,0],pnp.concentration[1,-1])


# #### Equilibrium cation and anion amount

# In[131]:


( pnp.numberConservationConstraint(pnp.xij1,0,0), pnp.numberConservationConstraint(pnp.xij1,1,0) )


# #### Initial cation and anion amount

# In[132]:


( pnp.numberConservationConstraint(pnp.xi0,0,0), pnp.numberConservationConstraint(pnp.xi0,1,0) )


# #### Species conservation

# In[133]:


(pnp.numberConservationConstraint(pnp.xij1,0,
                                 pnp.numberConservationConstraint(pnp.xi0,0,0)), 
 pnp.numberConservationConstraint(pnp.xij1,1,
                                 pnp.numberConservationConstraint(pnp.xi0,1,0)) )


# ## Sampling
# First, convert the physical concentration distributions into a callable "probability density":

# In[134]:


pnp.concentration.shape


# In[135]:


distributions = [interpolate.interp1d(pnp.grid,pnp.concentration[i,:]) for i in range(pnp.concentration.shape[0])]


# Normalization is not necessary here. Now we can sample the distribution of our $Na^+$ ions in z-direction.

# In[136]:


na_coordinate_sample = continuous2discrete(
    distribution=distributions[0], box=box_m, count=n_NaCl)
histx, histy, histz = get_histogram(na_coordinate_sample, box=box_m, n_bins=51)
plot_dist(histz, 'Distribution of Na+ ions in z-direction', reference_distribution=distributions[0])


# In[138]:


cl_coordinate_sample = continuous2discrete(
    distributions[1], box=box_m, count=n_NaCl)
histx, histy, histz = get_histogram(cl_coordinate_sample, box=box_m, n_bins=51)
plot_dist(histx, 'Distribution of Cl- ions in x-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box_m[0])
plot_dist(histy, 'Distribution of Cl- ions in y-direction', reference_distribution=lambda x: np.ones(x.shape)*1/box_m[1])
plot_dist(histz, 'Distribution of Cl- ions in z-direction', reference_distribution=distributions[1])


# ## Write to file
# To visualize our sampled coordinates, we utilize ASE to export it to some standard format, i.e. .xyz or LAMMPS data file.
# ASE speaks Ångström per default, thus we convert SI units:

# In[139]:


sample_size = int(n_NaCl)


# In[140]:


sample_size


# In[141]:


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


# In[142]:


# LAMMPS data format, units 'real', atom style 'full'
# before ASE 3.19.0b1, ASE had issues with exporting atom style 'full' in LAMMPS data file format, so do not expect this line to work for older ASE versions
ase.io.write('NaCl_c_4_M_u_0.5_V_box_5x5x10nm_lambda_S_2_Ang.lammps',system,format='lammps-data',units="real",atom_style='full')

