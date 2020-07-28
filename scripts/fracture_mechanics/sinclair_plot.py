import sys
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
plt.rcParams["figure.figsize"] = (10, 8)

from ase.units import GPa

from matscipy import parameter
from matscipy.elasticity import  fit_elastic_constants
from matscipy.fracture_mechanics.crack import CubicCrystalCrack, SinclairCrack


args = sys.argv[1:]
if not args:
    args = ['.'] # current directory only

sys.path.insert(0, args[0])
import params

calc = parameter('calc')
fmax = parameter('fmax', 1e-3)
vacuum = parameter('vacuum', 10.0)
flexible = parameter('flexible', True)
extended_far_field = parameter('extended_far_field', False)

k0 = parameter('k0', 1.0)
alpha0 = parameter('alpha0', 0.0) # initial guess for crack position

colors = parameter('colors',
                   ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"])

# compute elastic constants
cryst = params.cryst.copy()
cryst.pbc = True
cryst.calc = calc
C, C_err = fit_elastic_constants(cryst,
                                 symmetry=parameter('elastic_symmetry',
                                                    'triclinic'))

crk = CubicCrystalCrack(parameter('crack_surface'),
                        parameter('crack_front'),
                        Crot=C/GPa)

# Get Griffith's k1.
k1g = crk.k1g(parameter('surface_energy'))
print('Griffith k1 = %f' % k1g)

cluster = params.cluster.copy()
sc = SinclairCrack(crk, cluster, calc, k0 * k1g,
                   alpha=alpha0,
                   variable_alpha=flexible,
                   variable_k=True,
                   vacuum=vacuum,
                   extended_far_field=extended_far_field)

for directory, color in zip(args, colors):
    x = np.loadtxt(f'{directory}/x_traj.txt')
    u, alpha, k = sc.unpack(x.T)
    if flexible:
        plt.plot(k / k1g, alpha, label=directory)
    else:
        plt.plot(k / k1g, np.linalg.norm(u, axis=0),
                 label=directory)


plt.axvline(1.0, color='k', label=r'$K_G$')
plt.xlabel(r'Stress intensity factor $K/K_{G}$')
if flexible:
    plt.ylabel(r'Crack position $\alpha$')
else:
    plt.ylabel(r'Norm of corrector $\||u\||$ / $\mathrm{\AA{}}$')
plt.legend()

pdffile = parameter('pdffile', 'plot.pdf')
plt.savefig(pdffile)