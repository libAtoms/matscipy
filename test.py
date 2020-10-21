from ase.io import read
from matscipy.calculators.AbellTersoffBrenner.calculator import AbellTersoffBrenner, KumagaiTersoff

a = read('../aSi.structure_minimum_65atoms_pot_energy.nc')
hessian = AbellTersoffBrenner(**KumagaiTersoff()).calculate_hessian_matrix(a).todense()
print(hessian)
