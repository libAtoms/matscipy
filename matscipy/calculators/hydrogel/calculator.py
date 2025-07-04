from ase.calculators.calculator import Calculator


class Hydrogel(Calculator):
    implemented_properties = ['energy', 'stress', 'forces']
    default_parameters = {}
    name = 'Embedding'

    def __init__(self,
                 F=None,
                 f=None,
                 cutoff=None,
                 pair=None,
                 spring_topology=None):
        super().__init__()
        self._F = F
        self._f = f
        self._pair = pair
        self._cutoff = cutoff
        self._spring_i, self._spring_j, self._spring_S = spring_topology

        # Derivative of spline interpolation
        self._dF = F.derivative(1)
        self._df = f.derivative(1)
        self._dpair = pair.derivative(1)

    def energy_virial_and_forces(self, nat, cell, pbc, r_ic, i_n, j_n, dr_nc, abs_dr_n):
        # Local density
        density_i = np.bincount(i_n, weights=self._f(abs_dr_n), minlength=nat)

        # Energy
        spring_dr_nc = r_ic[self._spring_j] - r_ic[self._spring_i] + self._spring_S.dot(cell)
        spring_abs_dr_n = np.sqrt((spring_dr_nc**2).sum(axis=1))
        #print(spring_abs_dr_n.min(), spring_abs_dr_n.max())
        epot = self._F(density_i).sum() + 0.5*self._pair(spring_abs_dr_n).sum()
        demb_i = self._dF(density_i)

        # Forces
        df_n = self._df(abs_dr_n)
        dpair_n = self._dpair(spring_abs_dr_n)
        df_nc = -(demb_i[i_n]*df_n*dr_nc.T/abs_dr_n).T - 0.5 * (dpair_n * spring_dr_nc.T / spring_abs_dr_n).T

        # Sum for each atom
        f_ic = mabincount(j_n, weights=df_nc, minlength=nat) - \
            mabincount(i_n, weights=df_nc, minlength=nat)

        # Virial
        virial_v = -np.array([dr_nc[:,0]*df_nc[:,0],               # xx
                              dr_nc[:,1]*df_nc[:,1],               # yy
                              dr_nc[:,2]*df_nc[:,2],               # zz
                              dr_nc[:,1]*df_nc[:,2],               # yz
                              dr_nc[:,0]*df_nc[:,2],               # xz
                              dr_nc[:,0]*df_nc[:,1]]).sum(axis=1)  # xy

        return epot, virial_v, f_ic

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', atoms, self._cutoff)

        epot, virial_v, forces_ic = self.energy_virial_and_forces(len(atoms), atoms.cell, atoms.pbc, atoms.get_positions(), i_n, j_n, dr_nc, abs_dr_n)

        self.results.update({'energy': epot, 'free_energy': epot,
                             'stress': virial_v/atoms.get_volume(),
                             'forces': forces_ic})
