# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2014) James Kermode, King's College London
#                  Lars Pastewka, Karlsruhe Institute of Technology
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
# ======================================================================

"""
Embedded-atom method potential.
"""

import os

import numpy as np

import ase
from ase.calculators.calculator import Calculator

try:
    from scipy.interpolate import InterpolatedUnivariateSpline
except:
    InterpolatedUnivariateSpline = None

from scipy.sparse import bsr_matrix
from matscipy.calculators.eam.io import read_eam
from matscipy.neighbours import neighbour_list, first_neighbours, find_indices_of_reversed_pairs

###

def _make_splines(dx, y):
    if len(np.asarray(y).shape) > 1:
        return [_make_splines(dx, yy) for yy in y]
    else:
        return InterpolatedUnivariateSpline(np.arange(len(y))*dx, y)

def _make_derivative(x, n=1):
    if type(x) == list:
        return [_make_derivative(xx, n=n) for xx in x]
    else:
        return x.derivative(n=n)


###

class EAM(Calculator):
    implemented_properties = ['energy', 'stress', 'forces']
    default_parameters = {}
    name = 'EAM'
       
    def __init__(self, fn=None, atomic_numbers=None, F=None, f=None, rep=None,
                 cutoff=None, kind='eam/alloy'):
        Calculator.__init__(self)
        if fn is not None:
            source, parameters, F, f, rep = read_eam(fn, kind=kind)
            self._db_atomic_numbers = parameters.atomic_numbers
            self._db_cutoff = parameters.cutoff
            dr = parameters.distance_grid_spacing
            dF = parameters.density_grid_spacing

            # Create spline interpolation
            self.F = _make_splines(dF, F)
            self.f = _make_splines(dr, f)
            self.rep = _make_splines(dr, rep)
        else:
            self._db_atomic_numbers = atomic_numbers
            self.F = F
            self.f = f
            self.rep = rep
            self._db_cutoff = cutoff

        self.atnum_to_index = -np.ones(np.max(self._db_atomic_numbers)+1, dtype=int)
        self.atnum_to_index[self._db_atomic_numbers] = \
            np.arange(len(self._db_atomic_numbers))

        # Derivative of spline interpolation
        self.dF = _make_derivative(self.F)
        self.df = _make_derivative(self.f)
        self.drep = _make_derivative(self.rep)

        # Second derivative of spline interpolation
        self.ddF = _make_derivative(self.F, n=2)
        self.ddf = _make_derivative(self.f, n=2)
        self.ddrep = _make_derivative(self.rep, n=2)

    def energy_virial_and_forces(self, atomic_numbers_i, i_n, j_n, dr_nc, abs_dr_n):
        """
        Compute the potential energy, the virial and the forces.

        Parameters
        ----------
        atomic_numbers_i : array_like
            Atomic number for each atom in the system
        i_n, j_n : array_like
            Neighbor pairs
        dr_nc : array_like
            Distance vectors between neighbors
        abd_dr_n : array_like
            Length of distance vectors between neighbors

        Returns
        -------
        epot : float
            Potential energy
        virial_v : array
            Virial
        forces_ic : array
            Forces acting on each atom
        """
        nat = len(atomic_numbers_i)
        atnums_in_system = set(atomic_numbers_i)
        for atnum in atnums_in_system:
            if atnum not in self._db_atomic_numbers:
                raise RuntimeError('Element with atomic number {} found, but '
                                   'this atomic number has no EAM '
                                   'parametrization'.format(atnum))

        # Density
        f_n = np.zeros_like(abs_dr_n)
        df_n = np.zeros_like(abs_dr_n)
        for atidx1, atnum1 in enumerate(self._db_atomic_numbers):
            f1 = self.f[atidx1]
            df1 = self.df[atidx1]
            mask1 = atomic_numbers_i[j_n]==atnum1
            if mask1.sum() > 0:
                if type(f1) == list:
                    for atidx2, atnum2 in enumerate(self._db_atomic_numbers):
                        f = f1[atidx2]
                        df = df1[atidx2]
                        mask = np.logical_and(mask1, atomic_numbers_i[i_n]==atnum2)
                        if mask.sum() > 0:
                            f_n[mask] = f(abs_dr_n[mask])
                            df_n[mask] = df(abs_dr_n[mask])
                else:
                    f_n[mask1] = f1(abs_dr_n[mask1])
                    df_n[mask1] = df1(abs_dr_n[mask1])

        density_i = np.bincount(i_n, weights=f_n, minlength=nat)

        # Repulsion
        rep_n = np.zeros_like(abs_dr_n)
        drep_n = np.zeros_like(abs_dr_n)
        for atidx1, atnum1 in enumerate(self._db_atomic_numbers):
            rep1 = self.rep[atidx1]
            drep1 = self.drep[atidx1]
            mask1 = atomic_numbers_i[i_n]==atnum1
            if mask1.sum() > 0:
                for atidx2, atnum2 in enumerate(self._db_atomic_numbers):
                    rep = rep1[atidx2]
                    drep = drep1[atidx2]
                    mask = np.logical_and(mask1, atomic_numbers_i[j_n]==atnum2)
                    if mask.sum() > 0:
                        r = rep(abs_dr_n[mask])/abs_dr_n[mask]
                        rep_n[mask] = r
                        drep_n[mask] = (drep(abs_dr_n[mask])-r)/abs_dr_n[mask]

        # Energy
        epot = 0.5*np.sum(rep_n)
        demb_i = np.zeros(nat)
        for atidx, atnum in enumerate(self._db_atomic_numbers):
            F = self.F[atidx]
            dF = self.dF[atidx]
            mask = atomic_numbers_i==atnum
            if mask.sum() > 0:
                epot += np.sum(F(density_i[mask]))
                demb_i[mask] += dF(density_i[mask])

        # Forces
        reverse = find_indices_of_reversed_pairs(i_n, j_n, abs_dr_n)
        df_i_n = np.take(df_n, reverse)
        df_nc = -0.5*((demb_i[i_n]*df_n+demb_i[j_n]*df_i_n)+drep_n).reshape(-1,1)*dr_nc/abs_dr_n.reshape(-1,1)

        # Sum for each atom
        fx_i = np.bincount(j_n, weights=df_nc[:,0], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,0], minlength=nat)
        fy_i = np.bincount(j_n, weights=df_nc[:,1], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,1], minlength=nat)
        fz_i = np.bincount(j_n, weights=df_nc[:,2], minlength=nat) - \
            np.bincount(i_n, weights=df_nc[:,2], minlength=nat)

        # Virial
        virial_v = -np.array([dr_nc[:,0]*df_nc[:,0],               # xx
                              dr_nc[:,1]*df_nc[:,1],               # yy
                              dr_nc[:,2]*df_nc[:,2],               # zz
                              dr_nc[:,1]*df_nc[:,2],               # yz
                              dr_nc[:,0]*df_nc[:,2],               # xz
                              dr_nc[:,0]*df_nc[:,1]]).sum(axis=1)  # xy

        return epot, virial_v, np.transpose([fx_i, fy_i, fz_i])

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', self.atoms,
                                                   self._db_cutoff)

        epot, virial_v, forces_ic = self.energy_virial_and_forces(self.atoms.numbers, i_n, j_n, dr_nc, abs_dr_n)

        self.results = {'energy': epot, 'free_energy': epot,
                        'stress': virial_v/self.atoms.get_volume(),
                        'forces': forces_ic}

    def calculate_hessian_matrix(self, atoms, divide_by_masses=False):
        r"""Compute the Hessian matrix

        The Hessian matrix is the matrix of second derivatives 
        of the potential energy :math:`\mathcal{V}_\mathrm{int}` 
        with respect to coordinates, i.e.\

        .. math:: 
        
            \frac{\partial^2 \mathcal{V}_\mathrm{int}}
                 {\partial r_{\nu{}i}\partial r_{\mu{}j}},

        where the indices :math:`\mu` and :math:`\nu` refer to atoms and
        the indices :math:`i` and :math:`j` refer to the components of the
        position vector :math:`r_\nu` along the three spatial directions.

        The Hessian matrix has contributions from the pair potential
        and the embedding energy, 

        .. math::

            \frac{\partial^2 \mathcal{V}_\mathrm{int}}{\partial r_{\nu{}i}\partial r_{\mu{}j}} = 
            \frac{\partial^2 \mathcal{V}_\mathrm{pair}}{ \partial r_{\nu i} \partial r_{\mu j}} +
            \frac{\partial^2 \mathcal{V}_\mathrm{embed}}{\partial r_{\nu i} \partial r_{\mu j}}. 	


        The contribution from the pair potential is

        .. math::

            \frac{\partial^2 \mathcal{V}_\mathrm{pair}}{ \partial r_{\nu i} \partial r_{\mu j}} &= 
            -\phi_{\nu\mu}'' \left(
            \frac{r_{\nu\mu i}}{r_{\nu\mu}} 
            \frac{r_{\nu\mu j}}{r_{\nu\mu}} 
            \right)
            -\frac{\phi_{\nu\mu}'}{r_{\nu\mu}}\left(
            \delta_{ij}-
            \frac{r_{\nu\mu i}}{r_{\nu\mu}} 
            \frac{r_{\nu\mu j}}{r_{\nu\mu}} 
            \right) \\ 
            &+\delta_{\nu\mu}\sum_{\gamma\neq\nu}^{N}
            \phi_{\nu\gamma}'' \left(
            \frac{r_{\nu\gamma i}}{r_{\nu\gamma}} 
            \frac{r_{\nu\gamma j}}{r_{\nu\gamma}} 
            \right)
            +\delta_{\nu\mu}\sum_{\gamma\neq\nu}^{N}\frac{\phi_{\nu\gamma}'}{r_{\nu\gamma}}\left(
            \delta_{ij}-
            \frac{r_{\nu\gamma i}}{r_{\nu\gamma}} 
            \frac{r_{\nu\gamma j}}{r_{\nu\gamma}} 
            \right).

        The contribution of the embedding energy to the Hessian matrix is a sum of eight terms,
        
        .. math::

            \frac{\mathcal{V}_\mathrm{embed}}{\partial r_{\mu j} \partial r_{\nu i}} 
            	&= T_1 + T_2 + T_3 + T_4 + T_5 + T_6 + T_7 + T_8 \\ 
            T_1 &= 
            \delta_{\nu\mu}U_\nu''
            \sum_{\gamma\neq\nu}^{N}g_{\nu\gamma}'\frac{r_{\nu\gamma i}}{r_{\nu\gamma}}
            \sum_{\gamma\neq\nu}^{N}g_{\nu\gamma}'\frac{r_{\nu\gamma j}}{r_{\nu\gamma}} \\
            T_2 &= 
            -u_\nu''g_{\nu\mu}' \frac{r_{\nu\mu j}}{r_{\nu\mu}} \sum_{\gamma\neq\nu}^{N} 
            G_{\nu\gamma}' \frac{r_{\nu\gamma i}}{r_{\nu\gamma}} \\
            T_3 &=
            +u_\mu''g_{\mu\nu}' \frac{r_{\nu\mu i}}{r_{\nu\mu}} \sum_{\gamma\neq\mu}^{N} 
            G_{\mu\gamma}' \frac{r_{\mu\gamma j}}{r_{\mu\gamma}} \\
            T_4 &= -\left(u_\mu'g_{\mu\nu}'' + u_\nu'g_{\nu\mu}''\right)
            \left(
            \frac{r_{\nu\mu i}}{r_{\nu\mu}} 
            \frac{r_{\nu\mu j}}{r_{\nu\mu}}
            \right)\\
            T_5 &= \delta_{\nu\mu} \sum_{\gamma\neq\nu}^{N}
            \left(U_\gamma'g_{\gamma\nu}'' + U_\nu'g_{\nu\gamma}''\right)
            \left(
            \frac{r_{\nu\gamma i}}{r_{\nu\gamma}}
            \frac{r_{\nu\gamma j}}{r_{\nu\gamma}}
            \right) \\
            T_6 &= -\left(U_\mu'g_{\mu\nu}' + U_\nu'g_{\nu\mu}'\right) \frac{1}{r_{\nu\mu}}
            \left(
            \delta_{ij}- 
            \frac{r_{\nu\mu i}}{r_{\nu\mu}} 
            \frac{r_{\nu\mu j}}{r_{\nu\mu}}
            \right) \\
            T_7 &= \delta_{\nu\mu} \sum_{\gamma\neq\nu}^{N}
            \left(U_\gamma'g_{\gamma\nu}' + U_\nu'g_{\nu\gamma}'\right) \frac{1}{r_{\nu\gamma}}
            \left(\delta_{ij}-
            \frac{r_{\nu\gamma i}}{r_{\nu\gamma}} 
            \frac{r_{\nu\gamma j}}{r_{\nu\gamma}}
            \right) \\
            T_8 &= \sum_{\substack{\gamma\neq\nu \\ \gamma \neq \mu}}^{N}
            U_\gamma'' g_{\gamma\nu}'g_{\gamma\mu}' 
            \frac{r_{\gamma\nu i}}{r_{\gamma\nu}}
            \frac{r_{\gamma\mu j}}{r_{\gamma\mu}} 


        Parameters
        ----------
        atoms : ase.Atoms
        divide_by_masses : bool
            Divide block :math:`\nu\mu` by :math:`m_\num_\mu` to obtain the dynamical matrix

        Returns
        -------
        D : numpy.matrix
            Block Sparse Row matrix with the nonzero blocks

        Notes
        -----
        Notation:
         * :math:`N` Number of atoms 
         * :math:`\mathcal{V}_\mathrm{int}`  Total potential energy 
         * :math:`\mathcal{V}_\mathrm{pair}` Pair potential 
         * :math:`\mathcal{V}_\mathrm{embed}` Embedding energy 
         * :math:`r_{\nu{}i}`  Component :math:`i` of the position vector of atom :math:`\nu` 
         * :math:`r_{\nu\mu{}i} = r_{\mu{}i}-r_{\nu{}i}` 
         * :math:`r_{\nu\mu{}}` Norm of :math:`r_{\nu\mu{}i}`, i.e.\ :math:`\left(r_{\nu\mu{}1}^2+r_{\nu\mu{}2}^2+r_{\nu\mu{}3}^2\right)^{1/2}`
         * :math:`\phi_{\nu\mu}(r_{\nu\mu{}})` Pair potential energy of atoms :math:`\nu` and :math:`\mu` 
         * :math:`\rho_nu` Total electron density of atom :math:`\nu`  
         * :math:`U_\nu(\rho_nu)` Embedding energy of atom :math:`\nu` 
         * :math:`g_{\delta}\left(r_{\gamma\delta}\right) \equiv g_{\gamma\delta}` Contribution from atom :math:`\delta` to :math:`\rho_\gamma`
         * :math:`m_\nu` mass of atom :math:`\nu`
        """

        nat = len(atoms)
        atnums = atoms.numbers

        atnums_in_system = set(atnums)
        for atnum in atnums_in_system:
            if atnum not in atnums:
                raise RuntimeError('Element with atomic number {} found, but '
                                   'this atomic number has no EAM '
                                   'parametrization'.format(atnum))

        # i_n: index of the central atom
        # j_n: index of the neighbor atom
        # dr_nc: distance vector between the two
        # abs_dr_n: norm of distance vector
        # Variable name ending with _n indicate arrays that contain
        # one element for each pair in the neighbor list. Names ending
        # with _i indicate arrays containing one element for each atom.
        i_n, j_n, dr_nc, abs_dr_n = neighbour_list('ijDd', atoms,
                                                   self._db_cutoff)
        first_i = first_neighbours(nat, i_n)
        # Make sure that the neighborlist does not contain the same pairs twice.
        # Reoccuring entries may be due to small system size. In this case, the
        # Hessian matrix will not be symmetric.
        unique_pairs = set((i, j) for (i, j) in zip(i_n, j_n))
        if len(unique_pairs) != len(i_n):
            raise ValueError("neighborlist contains some pairs more than once") 
        assert(np.all(i_n != j_n))

        if divide_by_masses: 
            masses_i = atoms.get_masses().reshape(-1, 1, 1)
            geom_mean_mass_n = np.sqrt(
                np.take(masses_i, i_n) * np.take(masses_i, j_n)
            ).reshape(-1, 1, 1)

        # Calculate the derivatives of the pair energy
        drep_n = np.zeros_like(abs_dr_n)  # first derivative
        ddrep_n = np.zeros_like(abs_dr_n) # second derivative
        for atidx1, atnum1 in enumerate(self._db_atomic_numbers):
            rep1 = self.rep[atidx1]
            drep1 = self.drep[atidx1]
            ddrep1 = self.ddrep[atidx1]
            mask1 = atnums[i_n]==atnum1
            if mask1.sum() > 0:
                for atidx2, atnum2 in enumerate(self._db_atomic_numbers):
                    rep = rep1[atidx2]
                    drep = drep1[atidx2]
                    ddrep = ddrep1[atidx2]
                    mask = np.logical_and(mask1, atnums[j_n]==atnum2)
                    if mask.sum() > 0:
                        r = rep(abs_dr_n[mask])/abs_dr_n[mask]
                        drep_n[mask] = (drep(abs_dr_n[mask])-r) / abs_dr_n[mask]
                        ddrep_n[mask] = (ddrep(abs_dr_n[mask]) - 2.0 * drep_n[mask]) / abs_dr_n[mask]

        # Calculate the total electron density at each atom, and the 
        # derivatives of pairwise contributions
        f_n = np.zeros_like(abs_dr_n)
        df_n = np.zeros_like(abs_dr_n)  # first derivative
        ddf_n = np.zeros_like(abs_dr_n) # second derivative
        for atidx1, atnum1 in enumerate(self._db_atomic_numbers):
            f1 = self.f[atidx1]     
            df1 = self.df[atidx1]   
            ddf1 = self.ddf[atidx1] 
            mask1 = atnums[j_n]==atnum1
            if mask1.sum() > 0:
                if type(f1) == list:
                    for atidx2, atnum2 in enumerate(self._db_atomic_numbers):
                        f = f1[atidx2]
                        df = df1[atidx2]
                        ddf = ddf1[atidx2]
                        mask = np.logical_and(mask1, atnums[i_n]==atnum2)
                        if mask.sum() > 0:
                            f_n[mask] = f(abs_dr_n[mask])
                            df_n[mask] = df(abs_dr_n[mask])
                            ddf_n[mask] = ddf(abs_dr_n[mask])
                else:
                    f_n[mask1] = f1(abs_dr_n[mask1])
                    df_n[mask1] = df1(abs_dr_n[mask1])
                    ddf_n[mask1] = ddf1(abs_dr_n[mask1])
        # Accumulate density contributions
        density_i = np.bincount(i_n, weights=f_n, minlength=nat)

        # Calculate the derivatives of the embedding energy
        demb_i = np.zeros(nat)   # first derivative
        ddemb_i = np.zeros(nat)  # second derivative
        for atidx, atnum in enumerate(self._db_atomic_numbers):
            F = self.F[atidx]
            dF = self.dF[atidx]
            ddF = self.ddF[atidx]
            mask = atnums==atnum
            if mask.sum() > 0:
                demb_i[mask] += dF(density_i[mask])
                ddemb_i[mask] += ddF(density_i[mask])

        symmetry_check = False # check symmetry of individual terms?
        #------------------------------------------------------------------------ 
        # Calculate pair contribution to the Hessian matrix
        #------------------------------------------------------------------------ 
        e_nc = (dr_nc.T / abs_dr_n).T # normalized distance vectors r_i^{\mu\nu}
        outer_1 = e_nc.reshape(-1, 3, 1) * e_nc.reshape(-1, 1, 3)
        outer_2 = np.eye(3, dtype=e_nc.dtype) - outer_1
        D_ncc = -(ddrep_n * outer_1.T).T
        D_ncc += -(drep_n / abs_dr_n * outer_2.T).T
        if divide_by_masses:
            D = bsr_matrix(
                (D_ncc / geom_mean_mass_n, j_n, first_i), 
                shape=(3*nat, 3*nat)
            )
        else:
            D = bsr_matrix((D_ncc, j_n, first_i), shape=(3*nat, 3*nat))
        Ddiag = np.empty((nat, 3, 3))
        for x in range(3):
            for y in range(3):
                Ddiag[:, x, y] = -np.bincount(i_n, weights=D_ncc[:, x, y]) # summation
        if divide_by_masses:
            Ddiag /= masses_i
        # put 3x3 blocks on diagonal (Kronecker Delta delta_{\mu\nu})
        D += bsr_matrix((Ddiag, np.arange(nat), np.arange(nat+1)), shape=(3*nat, 3*nat))

        #------------------------------------------------------------------------ 
        # Calculate contribution of embedding term
        #------------------------------------------------------------------------ 
        # For each pair in the neighborlist, create arrays which store the 
        # derivatives of the embedding energy of the corresponding atoms.
        demb_i_n = np.take(demb_i, i_n)
        demb_j_n = np.take(demb_i, j_n)
        ddemb_i_n = np.take(ddemb_i, i_n)
        ddemb_j_n = np.take(ddemb_i, j_n)

        # Let r be an index into the neighbor list. df_n[r] contains the the
        # contribution from atom j_n[r] to the derivative of the electron
        # density of atom i_n[r]. We additionally need the contribution of
        # i_n[r] to the derivative of j_n[r]. This value is also in df_n,
        # but at a different position. reverse[r] gives the new index s
        # where we find this value. The same indexing applies to ddf_n.
        reverse = find_indices_of_reversed_pairs(i_n, j_n, abs_dr_n)
        df_i_n = np.take(df_n, reverse)
        ddf_i_n = np.take(ddf_n, reverse)
        #we already have ddf_j_n = ddf_n 

        # Term 1: 
        # \delta_{\nu\mu}U_\nu''
        # \sum_{\gamma\neq\nu}^{\natoms}g_{\nu\gamma}'\frac{r_{\nu\gamma i}}{r_{\nu\gamma}} 
        # \sum_{\gamma\neq\nu}^{\natoms}g_{\nu\gamma}'\frac{r_{\nu\gamma j}}{r_{\nu\gamma}} 
        # Likely zero in equilibrium because the sum is zero (appears in the force vector)
        df_n_e_nc_outer_product = (df_n * e_nc.T).T
        df_n_e_nc_i = np.empty((nat, 3), dtype=df_n.dtype)
        for x in range(3):
            df_n_e_nc_i[:, x] = np.bincount(i_n, weights=df_n_e_nc_outer_product[:, x], minlength=nat)
        term_1_ncc = ((ddemb_i * df_n_e_nc_i.T).T).reshape(-1,3,1) * df_n_e_nc_i.reshape(-1,1,3)
        if divide_by_masses:
            term_1_ncc /= masses_i
        term_1 = bsr_matrix((term_1_ncc, np.arange(nat), np.arange(nat+1)), shape=(3*nat, 3*nat)) 
        D += term_1
        if symmetry_check: 
            print("check term 1", np.linalg.norm(term_1.todense() - term_1.todense().T))

        # Term 2: 
        # -u_\nu''g_{\nu\mu}' \frac{r_{\nu\mu j}}{r_{\nu\mu}} \sum_{\gamma\neq\nu}^{\natoms} 
        #      g_{\nu\gamma}' \frac{r_{\nu\gamma i}}{r_{\nu\gamma}}
        # Likely zero in equilibrium because the sum is zero (appears in the force vector)
        df_n_e_nc_j_n = np.take(df_n_e_nc_i, j_n, axis=0)
        term_2_ncc = ((ddemb_j_n * df_i_n * e_nc.T).T).reshape(-1,3,1) * df_n_e_nc_j_n.reshape(-1,1,3)
        if divide_by_masses:
            term_2_ncc /= geom_mean_mass_n
        term_2 = bsr_matrix((term_2_ncc, j_n, first_i), shape=(3*nat, 3*nat))
        D += term_2 
        if symmetry_check: 
            print("check term 2", np.linalg.norm(term_2.todense() - term_2.todense().T))

        # Term 3: 
        # +u_\mu''g_{\mu\nu}' \frac{r_{\nu\mu i}}{r_{\nu\mu}} \sum_{\gamma\neq\mu}^{\natoms} 
        #      g_{\mu\gamma}' \frac{r_{\mu\gamma j}}{r_{\mu\gamma}} 
        # Likely zero in equilibrium because the sum is zero (appears in the force vector)
        df_n_e_nc_i_n = np.take(df_n_e_nc_i, i_n, axis=0)
        term_3_ncc = -((ddemb_i_n * df_n * df_n_e_nc_i_n.T).T).reshape(-1,3,1) * e_nc.reshape(-1,1,3)
        if divide_by_masses:
            term_3_ncc /= geom_mean_mass_n
        term_3 = bsr_matrix((term_3_ncc, j_n, first_i), shape=(3*nat, 3*nat))
        D += term_3
        if symmetry_check: 
            print("check term 3", np.linalg.norm(term_3.todense() - term_3.todense().T))

        # Term 4:
        # -\left(u_\mu'g_{\mu\nu}'' + u_\nu'g_{\nu\mu}''\right)
        # \left(
        # \frac{r_{\nu\mu i}}{r_{\nu\mu}} 
        # \frac{r_{\nu\mu j}}{r_{\nu\mu}}
        # \right)
        tmp_1 = -((demb_j_n * ddf_i_n + demb_i_n * ddf_n) * outer_1.T).T 
        # We don't immediately add term 4 to the matrix, because it would have 
        # to be normalized by the masses if divide_by_masses is true. However,
        # for construction of term 5, we need term 4 without normalization

        # Term 5:
        # \delta_{\nu\mu} \sum_{\gamma\neq\nu}^{\natoms}
        #\left(U_\gamma'g_{\gamma\nu}'' + U_\nu'g_{\nu\gamma}''\right)
        #\left(
        #\frac{r_{\nu\gamma i}}{r_{\nu\gamma}}
        #\frac{r_{\nu\gamma j}}{r_{\nu\gamma}}
        #\right)
        tmp_1_summed = np.empty((nat, 3, 3), dtype=tmp_1.dtype)
        for x in range(3):
            for y in range(3):
                tmp_1_summed[:, x, y] = -np.bincount(i_n, weights=tmp_1[:, x, y]) 
        if divide_by_masses:
            tmp_1_summed /= masses_i
        term_5 = bsr_matrix((tmp_1_summed, np.arange(nat), np.arange(nat+1)), shape=(3*nat, 3*nat))
        D += term_5
        if symmetry_check: 
            print("check term 5", np.linalg.norm(term_5.todense() - term_5.todense().T))
        if divide_by_masses:
            tmp_1 /= geom_mean_mass_n
        term_4 = bsr_matrix((tmp_1, j_n, first_i), shape=(3*nat, 3*nat))
        D += term_4 
        if symmetry_check: 
            print("check term 4", np.linalg.norm(term_4.todense() - term_4.todense().T))

        # Term 6:
        # -\left(U_\mu'g_{\mu\nu}' + U_\nu'g_{\nu\mu}'\right) \frac{1}{r_{\nu\mu}}
        #\left(
        #\delta_{ij}- 
        #\frac{r_{\nu\mu i}}{r_{\nu\mu}} 
        #\frac{r_{\nu\mu j}}{r_{\nu\mu}}
        #\right)
        # Like term 4, which was needed to construct term 5, we don't add 
        # term 6 immediately, because it is needed for construction of term 7
        tmp_2 = -((demb_j_n * df_i_n + demb_i_n * df_n) / abs_dr_n * outer_2.T).T

        # Term 7:
        # \delta_{\nu\mu} \sum_{\gamma\neq\nu}^{\natoms}
        #\left(U_\gamma'g_{\gamma\nu}' + U_\nu'g_{\nu\gamma}'\right) \frac{1}{r_{\nu\gamma}}
        #\left(\delta_{ij}-
        #\frac{r_{\nu\gamma i}}{r_{\nu\gamma}}
        #\frac{r_{\nu\gamma j}}{r_{\nu\gamma}}
        #\right)
        tmp_2_summed = np.empty((nat, 3, 3), dtype=tmp_2.dtype)
        for x in range(3):
            for y in range(3):
                tmp_2_summed[:, x, y] = -np.bincount(i_n, weights=tmp_2[:, x, y]) 
        if divide_by_masses:
            tmp_2_summed /= masses_i
        term_7 = bsr_matrix((tmp_2_summed, np.arange(nat), np.arange(nat+1)), shape=(3*nat, 3*nat))
        D += term_7 
        if symmetry_check: 
            print("check term 7", np.linalg.norm(term_7.todense() - term_7.todense().T))
        if divide_by_masses:
            tmp_2 /= geom_mean_mass_n
        term_6 = bsr_matrix((tmp_2, j_n, first_i), shape=(3*nat, 3*nat))
        D += term_6 
        if symmetry_check: 
            print("check term 6", np.linalg.norm(term_6.todense() - term_6.todense().T))

        #Term 8: 
        #  \sum_{\substack{\gamma\neq\nu \\ \gamma \neq \mu}}^{\natoms}
        #U_\gamma'' g_{\gamma\nu}'g_{\gamma\mu}' 
        #\frac{r_{\gamma\nu i}}{r_{\gamma\nu}}
        #\frac{r_{\gamma\mu j}}{r_{\gamma\mu}} 
        #-----------------------------------------------------------------------
        # This term requires knowledge of common neighbors of pairs of atoms.
        # Construction of a common neighbor list in Python is likely a
        # performance bottleneck; it should be implemented in C++ instead.
        #-----------------------------------------------------------------------

        # For each (i1, j2)-pair in the neighbor list, find all other
        # pairs (i2, j) which share the same j. This includes (i1, j)
        # itself. In this way, create a list with n blocks of rows, where
        # n is the length of the neighbor list. All rows in a block have
        # the same j. Each row corresponds to one triplet i1, j, i2.
        # The number of rows in the block is equal to the total number
        # of neighbors of j. This is the common neighbor list (cnl).
        # Besides the block number and i1, j, i2, the cnl stores the
        # index into the neighbor list where the pair i2-j can be found.
        j_order = np.argsort(j_n)
        i_n_2 = i_n[j_order]
        j_n_2 = j_n[j_order]
        first_j = first_neighbours(nat, j_n_2)
        num_rows_per_j = first_j[j_n+1] - first_j[j_n]
        total_num_rows = np.sum(num_rows_per_j)
        # The common neighbor information could be stored as
        # a 2D array. However, multiple 1D arrays are likely
        # better for performance (fewer cache misses later).
        cnl_block_number = np.empty(total_num_rows, dtype=i_n.dtype)
        cnl_j1 = np.empty(total_num_rows, dtype=i_n.dtype)
        cnl_j_order = np.empty(total_num_rows, dtype=i_n.dtype)
        cnl_i1_i2 = np.empty((total_num_rows, 2), dtype=i_n.dtype)
        block_start = np.r_[0, np.cumsum(num_rows_per_j)]
        slice_for_j1 = {j1: slice(first_j[j1], first_j[j1+1]) for j1 in np.arange(nat)}
        for block_number, (i1, j1) in enumerate(zip(i_n, j_n)):
            slice1 = slice(block_start[block_number], block_start[block_number+1])
            slice2 = slice_for_j1[j1] 
            cnl_block_number[slice1] = block_number
            cnl_j1[slice1] = j1 
            cnl_j_order[slice1] = j_order[slice2]
            cnl_i1_i2[slice1, 0] = i1
            cnl_i1_i2[slice1, 1] = i_n_2[slice2]
        # Determine bins for accumulation by bincount
        unique_pairs_i1_i2, bincount_bins = np.unique(cnl_i1_i2, axis=0, return_inverse=True)
        e_nu_cnl = np.take(e_nc, cnl_block_number, axis=0)
        e_mu_cnl = np.take(e_nc, cnl_j_order, axis=0)
        ddemb_cnl = np.take(ddemb_i, cnl_j1)
        df_nu_cnl = np.take(df_i_n, cnl_block_number)
        df_mu_cnl = np.take(df_i_n, cnl_j_order)
        tmp_3 = ((ddemb_cnl * df_nu_cnl * df_mu_cnl * e_nu_cnl.T).T).reshape(-1, 3, 1) * e_mu_cnl.reshape(-1, 1, 3)
        tmp_3_summed = np.empty((unique_pairs_i1_i2.shape[0], 3, 3), dtype=e_nc.dtype)
        for x in range(3):
            for y in range(3):
                tmp_3_summed[:, x, y] = np.bincount(
                        bincount_bins, weights=tmp_3[:, x, y], 
                        minlength=unique_pairs_i1_i2.shape[0]
                ) 
        if divide_by_masses:
            geom_mean_mass_i1_i2 = np.sqrt(
                np.take(masses_i, unique_pairs_i1_i2[:, 0]) * np.take(masses_i, unique_pairs_i1_i2[:, 1])
            )
            tmp_3_summed /= geom_mean_mass_i1_i2[:, np.newaxis, np.newaxis]
        index_ptr = first_neighbours(nat, unique_pairs_i1_i2[:, 0])
        term_8 = bsr_matrix((tmp_3_summed, unique_pairs_i1_i2[:, 1], index_ptr), shape=(3*nat, 3*nat))
        if symmetry_check:
            print("check term 8", np.linalg.norm(term_8.todense() - term_8.todense().T))
        D += term_8
        return D

    @property
    def cutoff(self):
        return self._db_cutoff
