from ase.calculators.calculator import Calculator
from matscipy.neighbours import neighbour_list, first_neighbours, triplet_list, find_indices_of_reversed_pairs
from scipy.sparse import bsr_matrix
import numpy as np


class AbellTersoffBrenner(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {}
    name = 'ThreeBodyPotential'

    def __init__(self, F, G,
                 d1F, d2F, d11F, d22F, d12F,
                 d1xG, d1yG, d1zG,
                 d1x1xG, d1y1yG, d1z1zG, d1y1zG, d1x1zG, d1x1yG,
                 d2xG, d2yG, d2zG,
                 d2x2xG, d2y2yG, d2z2zG, d2y2zG, d2x2zG, d2x2yG,
                 d1x2xG, d1y2yG, d1z2zG, d1y2zG, d1x2zG, d1x2yG, d1z2yG, d1z2xG, d1y2xG,
                 cutoff):
        Calculator.__init__(self)
        self.F = F
        self.G = G
        self.d1F = d1F
        self.d2F = d2F
        self.d11F = d11F
        self.d22F = d22F
        self.d12F = d12F
        self.d1xG = d1xG
        self.d1yG = d1yG
        self.d1zG = d1zG
        self.d1x1xG = d1x1xG
        self.d1y1yG = d1y1yG
        self.d1z1zG = d1z1zG
        self.d1y1zG = d1y1zG
        self.d1x1zG = d1x1zG
        self.d1x1yG = d1x1yG
        self.d2xG = d2xG
        self.d2yG = d2yG
        self.d2zG = d2zG
        self.d2x2xG = d2x2xG
        self.d2y2yG = d2y2yG
        self.d2z2zG = d2z2zG
        self.d2y2zG = d2y2zG
        self.d2x2zG = d2x2zG
        self.d2x2yG = d2x2yG
        self.d1x2xG = d1x2xG
        self.d1y2yG = d1y2yG
        self.d1z2zG = d1z2zG
        self.d1y2zG = d1y2zG
        self.d1x2zG = d1x2zG
        self.d1x2yG = d1x2yG
        self.d1z2yG = d1z2yG
        self.d1z2xG = d1z2xG
        self.d1y2xG = d1y2xG
        self.cutoff = cutoff

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # construct neighbor list
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms,
                                             cutoff=self.cutoff)

        nb_atoms = len(self.atoms)
        nb_pairs = len(i_p)

        # normal vectors
        n_pc = (r_pc.T / r_p).T
        nx_p, ny_p, nz_p = n_pc.T

        # construct triplet list
        first_i = first_neighbours(nb_atoms, i_p)
        ij_t, ik_t = triplet_list(first_i)

        # calculate energy
        G_t = self.G(r_pc[ij_t], r_pc[ik_t])
        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)
        F_p = self.F(r_p, xi_p)
        epot = 0.5 * np.sum(F_p)

        # calculate forces (per pair)
        fx_p = \
            self.d1F(r_p, xi_p) * nx_p + \
            self.d2F(r_p, xi_p) * np.bincount(ij_t, weights=self.d1xG(r_pc[ij_t], r_pc[ik_t]), minlength=nb_pairs) + \
            np.bincount(ik_t, self.d2F(r_p[ij_t], xi_p[ij_t]) * self.d2xG(r_pc[ij_t], r_pc[ik_t]), minlength=nb_pairs)
        fy_p = \
            self.d1F(r_p, xi_p) * ny_p + \
            self.d2F(r_p, xi_p) * np.bincount(ij_t, weights=self.d1yG(r_pc[ij_t], r_pc[ik_t]), minlength=nb_pairs) + \
            np.bincount(ik_t, self.d2F(r_p[ij_t], xi_p[ij_t]) * self.d2yG(r_pc[ij_t], r_pc[ik_t]), minlength=nb_pairs)
        fz_p = \
            self.d1F(r_p, xi_p) * nz_p + \
            self.d2F(r_p, xi_p) * np.bincount(ij_t, weights=self.d1zG(r_pc[ij_t], r_pc[ik_t]), minlength=nb_pairs) + \
            np.bincount(ik_t, self.d2F(r_p[ij_t], xi_p[ij_t]) * self.d2zG(r_pc[ij_t], r_pc[ik_t]), minlength=nb_pairs)

        # collect atomic forces
        fx_n = 0.5*(np.bincount(i_p, weights=fx_p) -
                    np.bincount(j_p, weights=fx_p))
        fy_n = 0.5*(np.bincount(i_p, weights=fy_p) -
                    np.bincount(j_p, weights=fy_p))
        fz_n = 0.5*(np.bincount(i_p, weights=fz_p) -
                    np.bincount(j_p, weights=fz_p))

        f_n = np.transpose([fx_n, fy_n, fz_n])

        self.results = {'energy': epot, 'forces': f_n}

    def calculate_hessian_matrix(self, atoms, divide_by_masses=False):
        """
        Calculate the Hessian matrix for a bond order potential.
        For an atomic configuration with N atoms in d dimensions the hessian matrix is a symmetric, hermitian matrix
        with a shape of (d*N,d*N). The matrix is in general a sparse matrix, which consists of dense blocks of shape (d,d), which
        are the mixed second derivatives.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.

        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems
        """

        # construct neighbor list
        i_p, j_p, r_p, r_pc = neighbour_list('ijdD', atoms=atoms,
                                             cutoff=2*self.cutoff)
        mask_p = r_p > self.cutoff

        nb_atoms = len(atoms)
        nb_pairs = len(i_p)

        # reverse pairs
        tr_p = find_indices_of_reversed_pairs(i_p, j_p, r_p)

        # normal vectors
        n_pc = (r_pc.T / r_p).T
        nx_p, ny_p, nz_p = n_pc.T

        # construct triplet list
        first_i = first_neighbours(nb_atoms, i_p)
        ij_t, ik_t, jk_t = triplet_list(first_i, r_p, self.cutoff, i_p, j_p)
        first_ij = first_neighbours(len(i_p), ij_t)

        nb_triplets = len(ij_t)

        # basic triplet and pair terms
        G_t = self.G(r_pc[ij_t], r_pc[ik_t])
        xi_p = np.bincount(ij_t, weights=G_t, minlength=nb_pairs)
        F_p = self.F(r_p, xi_p)

        # Hessian term #4
        d1F_p = self.d1F(r_p, xi_p)
        d1F_p[mask_p] = 0.0  # we need to explicitly exclude everything with r > cutoff
        Hxx_p = - d1F_p * (1 - nx_p * nx_p) / r_p
        Hyy_p = - d1F_p * (1 - ny_p * ny_p) / r_p
        Hzz_p = - d1F_p * (1 - nz_p * nz_p) / r_p
        Hyz_p = d1F_p * ny_p * nz_p / r_p
        Hxz_p = d1F_p * nx_p * nz_p / r_p
        Hxy_p = d1F_p * nx_p * ny_p / r_p
        Hzy_p = d1F_p * nz_p * ny_p / r_p
        Hzx_p = d1F_p * nz_p * nx_p / r_p
        Hyx_p = d1F_p * ny_p * nx_p / r_p

        # Hessian term #1
        d11F_p = self.d11F(r_p, xi_p)
        d11F_p[mask_p] = 0.0
        Hxx_p -= d11F_p * nx_p * nx_p
        Hyy_p -= d11F_p * ny_p * ny_p
        Hzz_p -= d11F_p * nz_p * nz_p
        Hyz_p -= d11F_p * ny_p * nz_p
        Hxz_p -= d11F_p * nx_p * nz_p
        Hxy_p -= d11F_p * nx_p * ny_p
        Hzy_p -= d11F_p * nz_p * ny_p
        Hzx_p -= d11F_p * nz_p * nx_p
        Hyx_p -= d11F_p * ny_p * nx_p

        # Hessian term #2
        d12F_p = self.d12F(r_p, xi_p)
        d12F_p[mask_p] = 0.0
        d2xG_t = self.d2xG(r_pc[ij_t], r_pc[ik_t])
        d2yG_t = self.d2yG(r_pc[ij_t], r_pc[ik_t])
        d2zG_t = self.d2zG(r_pc[ij_t], r_pc[ik_t])
        Hxx_t = d12F_p[ij_t] * d2xG_t * nx_p[ij_t]
        Hyy_t = d12F_p[ij_t] * d2yG_t * ny_p[ij_t]
        Hzz_t = d12F_p[ij_t] * d2zG_t * nz_p[ij_t]
        Hyz_t = d12F_p[ij_t] * d2yG_t * nz_p[ij_t]
        Hxz_t = d12F_p[ij_t] * d2xG_t * nz_p[ij_t]
        Hxy_t = d12F_p[ij_t] * d2xG_t * ny_p[ij_t]
        Hzy_t = d12F_p[ij_t] * d2zG_t * ny_p[ij_t]
        Hzx_t = d12F_p[ij_t] * d2zG_t * nx_p[ij_t]
        Hyx_t = d12F_p[ij_t] * d2yG_t * nx_p[ij_t]

        Hxx_p += np.bincount(tr_p[jk_t], weights=Hxx_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hxx_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hxx_t, minlength=nb_pairs)
        Hyy_p += np.bincount(tr_p[jk_t], weights=Hyy_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hyy_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hyy_t, minlength=nb_pairs)
        Hzz_p += np.bincount(tr_p[jk_t], weights=Hzz_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hzz_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hzz_t, minlength=nb_pairs)
        Hyz_p += np.bincount(tr_p[jk_t], weights=Hyz_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hyz_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hyz_t, minlength=nb_pairs)
        Hxz_p += np.bincount(tr_p[jk_t], weights=Hxz_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hxz_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hxz_t, minlength=nb_pairs)
        Hxy_p += np.bincount(tr_p[jk_t], weights=Hxy_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hxy_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hxy_t, minlength=nb_pairs)
        Hzy_p += np.bincount(tr_p[jk_t], weights=Hzy_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hzy_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hzy_t, minlength=nb_pairs)
        Hzx_p += np.bincount(tr_p[jk_t], weights=Hzx_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hzx_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hzx_t, minlength=nb_pairs)
        Hyx_p += np.bincount(tr_p[jk_t], weights=Hyx_t, minlength=nb_pairs) - np.bincount(ij_t, weights=Hyx_t, minlength=nb_pairs) - np.bincount(tr_p[ik_t], weights=Hyx_t, minlength=nb_pairs)

        d1xG_t = self.d1xG(r_pc[ij_t], r_pc[ik_t])
        d1yG_t = self.d1yG(r_pc[ij_t], r_pc[ik_t])
        d1zG_t = self.d1zG(r_pc[ij_t], r_pc[ik_t])
        Hxx_t = d12F_p[ij_t] * d1xG_t * nx_p[ij_t]
        Hyy_t = d12F_p[ij_t] * d1yG_t * ny_p[ij_t]
        Hzz_t = d12F_p[ij_t] * d1zG_t * nz_p[ij_t]
        Hyz_t = d12F_p[ij_t] * d1yG_t * nz_p[ij_t]
        Hxz_t = d12F_p[ij_t] * d1xG_t * nz_p[ij_t]
        Hxy_t = d12F_p[ij_t] * d1xG_t * ny_p[ij_t]
        Hzy_t = d12F_p[ij_t] * d1zG_t * ny_p[ij_t]
        Hzx_t = d12F_p[ij_t] * d1zG_t * nx_p[ij_t]
        Hyx_t = d12F_p[ij_t] * d1yG_t * nx_p[ij_t]

        Hxx_p += - np.bincount(ij_t, weights=Hxx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxx_t, minlength=nb_pairs)
        Hyy_p += - np.bincount(ij_t, weights=Hyy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyy_t, minlength=nb_pairs)
        Hzz_p += - np.bincount(ij_t, weights=Hzz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzz_t, minlength=nb_pairs)
        Hyz_p += - np.bincount(ij_t, weights=Hyz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyz_t, minlength=nb_pairs)
        Hxz_p += - np.bincount(ij_t, weights=Hxz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxz_t, minlength=nb_pairs)
        Hxy_p += - np.bincount(ij_t, weights=Hxy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxy_t, minlength=nb_pairs)
        Hzy_p += - np.bincount(ij_t, weights=Hzy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzy_t, minlength=nb_pairs)
        Hzx_p += - np.bincount(ij_t, weights=Hzx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzx_t, minlength=nb_pairs)
        Hyx_p += - np.bincount(ij_t, weights=Hyx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyx_t, minlength=nb_pairs)

        # Hessian term #5
        d2F_p = self.d2F(r_p, xi_p)
        d2F_p[mask_p] = 0.0
        d1x1xG_t = self.d1x1xG(r_pc[ij_t], r_pc[ik_t])
        d1y1yG_t = self.d1y1yG(r_pc[ij_t], r_pc[ik_t])
        d1z1zG_t = self.d1z1zG(r_pc[ij_t], r_pc[ik_t])
        d1y1zG_t = self.d1y1zG(r_pc[ij_t], r_pc[ik_t])
        d1x1zG_t = self.d1x1zG(r_pc[ij_t], r_pc[ik_t])
        d1x1yG_t = self.d1x1yG(r_pc[ij_t], r_pc[ik_t])
        Hxx_t = d2F_p[ij_t] * d1x1xG_t
        Hyy_t = d2F_p[ij_t] * d1y1yG_t
        Hzz_t = d2F_p[ij_t] * d1z1zG_t
        Hyz_t = d2F_p[ij_t] * d1y1zG_t
        Hxz_t = d2F_p[ij_t] * d1x1zG_t
        Hxy_t = d2F_p[ij_t] * d1x1yG_t
        Hzy_t = d2F_p[ij_t] * d1y1zG_t
        Hzx_t = d2F_p[ij_t] * d1x1zG_t
        Hyx_t = d2F_p[ij_t] * d1x1yG_t

        Hxx_p -= np.bincount(ij_t, weights=Hxx_t, minlength=nb_pairs)
        Hyy_p -= np.bincount(ij_t, weights=Hyy_t, minlength=nb_pairs)
        Hzz_p -= np.bincount(ij_t, weights=Hzz_t, minlength=nb_pairs)
        Hyz_p -= np.bincount(ij_t, weights=Hyz_t, minlength=nb_pairs)
        Hxz_p -= np.bincount(ij_t, weights=Hxz_t, minlength=nb_pairs)
        Hxy_p -= np.bincount(ij_t, weights=Hxy_t, minlength=nb_pairs)
        Hzy_p -= np.bincount(ij_t, weights=Hzy_t, minlength=nb_pairs)
        Hzx_p -= np.bincount(ij_t, weights=Hzx_t, minlength=nb_pairs)
        Hyx_p -= np.bincount(ij_t, weights=Hyx_t, minlength=nb_pairs)

        d2x2xG_t = self.d2x2xG(r_pc[ij_t], r_pc[ik_t])
        d2y2yG_t = self.d2y2yG(r_pc[ij_t], r_pc[ik_t])
        d2z2zG_t = self.d2z2zG(r_pc[ij_t], r_pc[ik_t])
        d2y2zG_t = self.d2y2zG(r_pc[ij_t], r_pc[ik_t])
        d2x2zG_t = self.d2x2zG(r_pc[ij_t], r_pc[ik_t])
        d2x2yG_t = self.d2x2yG(r_pc[ij_t], r_pc[ik_t])
        Hxx_t = d2F_p[ij_t] * d2x2xG_t
        Hyy_t = d2F_p[ij_t] * d2y2yG_t
        Hzz_t = d2F_p[ij_t] * d2z2zG_t
        Hyz_t = d2F_p[ij_t] * d2y2zG_t
        Hxz_t = d2F_p[ij_t] * d2x2zG_t
        Hxy_t = d2F_p[ij_t] * d2x2yG_t
        Hzy_t = d2F_p[ij_t] * d2y2zG_t
        Hzx_t = d2F_p[ij_t] * d2x2zG_t
        Hyx_t = d2F_p[ij_t] * d2x2yG_t

        Hxx_p -= np.bincount(ik_t, weights=Hxx_t, minlength=nb_pairs)
        Hyy_p -= np.bincount(ik_t, weights=Hyy_t, minlength=nb_pairs)
        Hzz_p -= np.bincount(ik_t, weights=Hzz_t, minlength=nb_pairs)
        Hyz_p -= np.bincount(ik_t, weights=Hyz_t, minlength=nb_pairs)
        Hxz_p -= np.bincount(ik_t, weights=Hxz_t, minlength=nb_pairs)
        Hxy_p -= np.bincount(ik_t, weights=Hxy_t, minlength=nb_pairs)
        Hzy_p -= np.bincount(ik_t, weights=Hzy_t, minlength=nb_pairs)
        Hzx_p -= np.bincount(ik_t, weights=Hzx_t, minlength=nb_pairs)
        Hyx_p -= np.bincount(ik_t, weights=Hyx_t, minlength=nb_pairs)

        d1x2xG_t = self.d1x2xG(r_pc[ij_t], r_pc[ik_t])
        d1y2yG_t = self.d1y2yG(r_pc[ij_t], r_pc[ik_t])
        d1z2zG_t = self.d1z2zG(r_pc[ij_t], r_pc[ik_t])
        d1y2zG_t = self.d1y2zG(r_pc[ij_t], r_pc[ik_t])
        d1x2zG_t = self.d1x2zG(r_pc[ij_t], r_pc[ik_t])
        d1x2yG_t = self.d1x2yG(r_pc[ij_t], r_pc[ik_t])
        d1z2yG_t = self.d1z2yG(r_pc[ij_t], r_pc[ik_t])
        d1z2xG_t = self.d1z2xG(r_pc[ij_t], r_pc[ik_t])
        d1y2xG_t = self.d1y2xG(r_pc[ij_t], r_pc[ik_t])
        Hxx_t = d2F_p[ij_t] * d1x2xG_t
        Hyy_t = d2F_p[ij_t] * d1y2yG_t
        Hzz_t = d2F_p[ij_t] * d1z2zG_t
        Hyz_t = d2F_p[ij_t] * d1y2zG_t
        Hxz_t = d2F_p[ij_t] * d1x2zG_t
        Hxy_t = d2F_p[ij_t] * d1x2yG_t
        Hzy_t = d2F_p[ij_t] * d1z2yG_t
        Hzx_t = d2F_p[ij_t] * d1z2xG_t
        Hyx_t = d2F_p[ij_t] * d1y2xG_t

        Hxx_p += np.bincount(jk_t, weights=Hxx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxx_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hxx_t, minlength=nb_pairs)
        Hyy_p += np.bincount(jk_t, weights=Hyy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyy_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hyy_t, minlength=nb_pairs)
        Hzz_p += np.bincount(jk_t, weights=Hzz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzz_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hzz_t, minlength=nb_pairs)
        Hyz_p += np.bincount(jk_t, weights=Hyz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyz_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hyz_t, minlength=nb_pairs)
        Hxz_p += np.bincount(jk_t, weights=Hxz_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxz_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hxz_t, minlength=nb_pairs)
        Hxy_p += np.bincount(jk_t, weights=Hxy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hxy_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hxy_t, minlength=nb_pairs)
        Hzy_p += np.bincount(jk_t, weights=Hzy_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzy_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hzy_t, minlength=nb_pairs)
        Hzx_p += np.bincount(jk_t, weights=Hzx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hzx_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hzx_t, minlength=nb_pairs)
        Hyx_p += np.bincount(jk_t, weights=Hyx_t, minlength=nb_pairs) - np.bincount(tr_p[ij_t], weights=Hyx_t, minlength=nb_pairs) - np.bincount(ik_t, weights=Hyx_t, minlength=nb_pairs)

        # Hessian term #3

        ## Terms involving D_1 * D_1
        d22F_p = self.d22F(r_p, xi_p)
        d22F_p[mask_p] = 0.0
        d1xG_p = np.bincount(ij_t, weights=d1xG_t, minlength=nb_pairs)
        d1yG_p = np.bincount(ij_t, weights=d1yG_t, minlength=nb_pairs)
        d1zG_p = np.bincount(ij_t, weights=d1zG_t, minlength=nb_pairs)

        Hxx_p -= d22F_p * d1xG_p * d1xG_p
        Hyy_p -= d22F_p * d1yG_p * d1yG_p
        Hzz_p -= d22F_p * d1zG_p * d1zG_p
        Hyz_p -= d22F_p * d1yG_p * d1zG_p
        Hxz_p -= d22F_p * d1xG_p * d1zG_p
        Hxy_p -= d22F_p * d1xG_p * d1yG_p
        Hzy_p -= d22F_p * d1zG_p * d1yG_p
        Hzx_p -= d22F_p * d1zG_p * d1xG_p
        Hyx_p -= d22F_p * d1yG_p * d1xG_p

        ## Terms involving D_2 * D_2
        d2xG_p = np.bincount(ij_t, weights=d2xG_t, minlength=nb_pairs)
        d2yG_p = np.bincount(ij_t, weights=d2yG_t, minlength=nb_pairs)
        d2zG_p = np.bincount(ij_t, weights=d2zG_t, minlength=nb_pairs)

        Hxx_p -= np.bincount(ik_t, weights=(d22F_p * d2xG_p)[ij_t] * d2xG_t, minlength=nb_pairs)
        Hyy_p -= np.bincount(ik_t, weights=(d22F_p * d2yG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzz_p -= np.bincount(ik_t, weights=(d22F_p * d2zG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hyz_p -= np.bincount(ik_t, weights=(d22F_p * d2yG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hxz_p -= np.bincount(ik_t, weights=(d22F_p * d2xG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hxy_p -= np.bincount(ik_t, weights=(d22F_p * d2xG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzy_p -= np.bincount(ik_t, weights=(d22F_p * d2zG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzx_p -= np.bincount(ik_t, weights=(d22F_p * d2zG_p)[ij_t] * d2xG_t, minlength=nb_pairs)
        Hyx_p -= np.bincount(ik_t, weights=(d22F_p * d2yG_p)[ij_t] * d2xG_t, minlength=nb_pairs)

        for il_im in range(nb_triplets):
            il = ij_t[il_im]
            im = ik_t[il_im]
            lm = jk_t[il_im]
            for t in range(first_ij[il], first_ij[il+1]):
                ij = ik_t[t]
                if ij != il and ij != im:
                    r_p_ij = np.array([r_pc[ij]])
                    r_p_il = np.array([r_pc[il]])
                    r_p_im = np.array([r_pc[im]])
                    Hxx_p[lm] += 0.5 * d22F_p[ij] * self.d2xG(r_p_ij, r_p_il)[0] * self.d2xG(r_p_ij, r_p_im)[0]
                    Hyy_p[lm] += 0.5 * d22F_p[ij] * self.d2yG(r_p_ij, r_p_il)[0] * self.d2yG(r_p_ij, r_p_im)[0]
                    Hzz_p[lm] += 0.5 * d22F_p[ij] * self.d2zG(r_p_ij, r_p_il)[0] * self.d2zG(r_p_ij, r_p_im)[0]
                    Hyz_p[lm] += 0.5 * d22F_p[ij] * self.d2yG(r_p_ij, r_p_il)[0] * self.d2zG(r_p_ij, r_p_im)[0]
                    Hxz_p[lm] += 0.5 * d22F_p[ij] * self.d2xG(r_p_ij, r_p_il)[0] * self.d2zG(r_p_ij, r_p_im)[0]
                    Hxy_p[lm] += 0.5 * d22F_p[ij] * self.d2xG(r_p_ij, r_p_il)[0] * self.d2yG(r_p_ij, r_p_im)[0]
                    Hzy_p[lm] += 0.5 * d22F_p[ij] * self.d2zG(r_p_ij, r_p_il)[0] * self.d2yG(r_p_ij, r_p_im)[0]
                    Hzx_p[lm] += 0.5 * d22F_p[ij] * self.d2zG(r_p_ij, r_p_il)[0] * self.d2xG(r_p_ij, r_p_im)[0]
                    Hyx_p[lm] += 0.5 * d22F_p[ij] * self.d2yG(r_p_ij, r_p_il)[0] * self.d2xG(r_p_ij, r_p_im)[0]

        ## Terms involving D_1 * D_2
        # Was d2*d1
        Hxx_p += np.bincount(jk_t, weights=(d22F_p * d1xG_p)[ij_t] * d2xG_t, minlength=nb_pairs)
        Hyy_p += np.bincount(jk_t, weights=(d22F_p * d1yG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzz_p += np.bincount(jk_t, weights=(d22F_p * d1zG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hyz_p += np.bincount(jk_t, weights=(d22F_p * d1yG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hxz_p += np.bincount(jk_t, weights=(d22F_p * d1xG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hxy_p += np.bincount(jk_t, weights=(d22F_p * d1xG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzy_p += np.bincount(jk_t, weights=(d22F_p * d1zG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzx_p += np.bincount(jk_t, weights=(d22F_p * d1zG_p)[ij_t] * d2xG_t, minlength=nb_pairs)
        Hyx_p += np.bincount(jk_t, weights=(d22F_p * d1yG_p)[ij_t] * d2xG_t, minlength=nb_pairs)

        # Was d2*d1
        Hxx_p -= np.bincount(ik_t, weights=(d22F_p * d1xG_p)[ij_t] * d2xG_t, minlength=nb_pairs)
        Hyy_p -= np.bincount(ik_t, weights=(d22F_p * d1yG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzz_p -= np.bincount(ik_t, weights=(d22F_p * d1zG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hyz_p -= np.bincount(ik_t, weights=(d22F_p * d1yG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hxz_p -= np.bincount(ik_t, weights=(d22F_p * d1xG_p)[ij_t] * d2zG_t, minlength=nb_pairs)
        Hxy_p -= np.bincount(ik_t, weights=(d22F_p * d1xG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzy_p -= np.bincount(ik_t, weights=(d22F_p * d1zG_p)[ij_t] * d2yG_t, minlength=nb_pairs)
        Hzx_p -= np.bincount(ik_t, weights=(d22F_p * d1zG_p)[ij_t] * d2xG_t, minlength=nb_pairs)
        Hyx_p -= np.bincount(ik_t, weights=(d22F_p * d1yG_p)[ij_t] * d2xG_t, minlength=nb_pairs)

        Hxx_p -= d22F_p * d2xG_p * d1xG_p
        Hyy_p -= d22F_p * d2yG_p * d1yG_p
        Hzz_p -= d22F_p * d2zG_p * d1zG_p
        Hyz_p -= d22F_p * d2yG_p * d1zG_p
        Hxz_p -= d22F_p * d2xG_p * d1zG_p
        Hxy_p -= d22F_p * d2xG_p * d1yG_p
        Hzy_p -= d22F_p * d2zG_p * d1yG_p
        Hzx_p -= d22F_p * d2zG_p * d1xG_p
        Hyx_p -= d22F_p * d2yG_p * d1xG_p

        # Add the conjugate terms (symmetrize Hessian)
        Hxx_p += Hxx_p[tr_p]
        Hyy_p += Hyy_p[tr_p]
        Hzz_p += Hzz_p[tr_p]
        tmp = Hyz_p.copy()
        Hyz_p += Hzy_p[tr_p]
        Hzy_p += tmp[tr_p]
        tmp = Hxz_p.copy()
        Hxz_p += Hzx_p[tr_p]
        Hzx_p += tmp[tr_p]
        tmp = Hxy_p.copy()
        Hxy_p += Hyx_p[tr_p]
        Hyx_p += tmp[tr_p]

        # Construct diagonal terms from off-diagonal terms
        Hxx_a = -np.bincount(i_p, weights=Hxx_p)
        Hyy_a = -np.bincount(i_p, weights=Hyy_p)
        Hzz_a = -np.bincount(i_p, weights=Hzz_p)
        Hyz_a = -np.bincount(i_p, weights=Hyz_p)
        Hxz_a = -np.bincount(i_p, weights=Hxz_p)
        Hxy_a = -np.bincount(i_p, weights=Hxy_p)
        Hzy_a = -np.bincount(i_p, weights=Hzy_p)
        Hzx_a = -np.bincount(i_p, weights=Hzx_p)
        Hyx_a = -np.bincount(i_p, weights=Hyx_p)

        # Construct full off-diagonal term
        H_pcc = np.transpose([[Hxx_p, Hyx_p, Hzx_p],
                              [Hxy_p, Hyy_p, Hzy_p],
                              [Hxz_p, Hyz_p, Hzz_p]])

        # Construct full diagonal term
        H_acc = np.transpose([[Hxx_a, Hxy_a, Hxz_a],
                              [Hyx_a, Hyy_a, Hyz_a],
                              [Hzx_a, Hzy_a, Hzz_a]])

        if divide_by_masses:
            mass_nat = atoms.get_masses()
            geom_mean_mass_n = np.sqrt(mass_nat[i_p]*mass_nat[j_p])
            return \
                bsr_matrix(((H_pcc.T/(2 * geom_mean_mass_n)).T, j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms)) \
                + bsr_matrix(((H_acc.T/(2 * mass_nat)).T, np.arange(nb_atoms), np.arange(nb_atoms+1)),
                     shape=(3*nb_atoms, 3*nb_atoms))
        else:
            return \
                bsr_matrix((H_pcc/2, j_p, first_i), shape=(3*nb_atoms, 3*nb_atoms)) \
                + bsr_matrix((H_acc/2, np.arange(nb_atoms), np.arange(nb_atoms+1)),
                     shape=(3*nb_atoms, 3*nb_atoms))


def ab(x):
    """Compute absolute value (norm) of an array of vectors"""
    return np.linalg.norm(x, axis=1)


def TersoffIII():
    A = 1.8308e3
    B = 4.7118e2
    chi = 1.0
    lam = 2.4799e0
    mu = 1.7322e0
    beta = 1.1000e-6
    n = 7.8734e-1
    c = 1.0039e5
    d = 1.6217e1
    h = -5.9825e-1
    r1 = 2.70
    r2 = 3.00
    #lam3 = 5.19745
    lam3 = 0.0
    delta = 3

    f = lambda r: np.where(
        r < r1,
        np.ones_like(r),
        np.where(r > r2,
            np.zeros_like(r),
            (1 + np.cos((np.pi*(r-r1)/(r2-r1))))/2
            )
        )
    df = lambda r: np.where(
        r < r1,
        np.zeros_like(r),
        np.where(r > r2,
            np.zeros_like(r),
            -np.pi * np.sin(np.pi * (r - r1)/(r2 - r1)) / (2*(r2 - r1))
            )
        )
    ddf = lambda r: np.where(
        r < r1,
        np.zeros_like(r),
        np.where(r > r2,
            np.zeros_like(r),
            -np.pi**2 * np.cos(np.pi * (r - r1)/(r2 - r1)) / (2*(r2 - r1)**2)
            )
        )

    fR = lambda r: A * np.exp(-lam * r)
    dfR = lambda r: -lam * fR(r)
    ddfR = lambda r: lam**2 * fR(r)

    fA = lambda r: -B * np.exp(-mu * r)
    dfA = lambda r: -mu * fA(r)
    ddfA = lambda r: mu**2 * fA(r)

    b = lambda xi: (1 + (beta * xi)**n)**(-1 / (2 * n))
    db = lambda xi: -0.5 * beta * (beta * xi)**(-1 + n) * (1 + (beta * xi)**n)**(-1-1/(2*n))
    ddb = lambda xi: -0.5 * beta**2 * (n-1) * (beta * xi)**(-2 + n) * (1 + (beta * xi)**n)**(-1-1/(2*n)) - \
        0.5 * beta**2 * n * (beta * xi)**(-2 + 2*n) * ( -1 - 1/(2*n)) * (1 + (beta * xi)**n)**(-2-1/(2*n))

    F = lambda r, xi: f(r) * (fR(r) + b(xi) * fA(r))
    d1F = lambda r, xi: df(r) * (fR(r) + b(xi) * fA(r)) + f(r) * (dfR(r) + b(xi) * dfA(r))
    d2F = lambda r, xi: f(r) * fA(r) * db(xi)
    d11F = lambda r, xi: f(r) * (ddfR(r) + b(xi) * ddfA(r)) + 2 * df(r) * (dfR(r) + b(xi) * dfA(r)) + ddf(r) * (fR(r) + b(xi) * fA(r))
    d22F = lambda r, xi:  f(r) * fA(r) * ddb(xi)
    d12F = lambda r, xi: f(r) * dfA(r) * db(xi) + fA(r) * df(r) * db(xi)

    g = lambda cost: 1 + c**2 / d**2 - c**2 / (d**2 + (h - cost)**2)
    dg = lambda cost: -2 * c**2 * (h - cost) / (d**2 + (h - cost)**2)**2
    ddg = lambda cost: 2 * c**2 / (d**2 + (h - cost)**2)**2 - 8 * c**2 * (h - cost)**2 / (d**2 + (h - cost)**2)**3

    hf = lambda rij, rik: f(ab(rik)) * np.exp(lam3*(ab(rij)-ab(rik))**delta)
    d1h = lambda rij, rik: lam3 * hf(rij, rik)
    d2h = lambda rij, rik: -lam3 * hf(rij, rik) + \
        df(ab(rik)) * np.exp(lam3*(ab(rij)-ab(rik))**delta)
    d11h = lambda rij, rik: lam3**2*hf(rij, rik)
    d12h = lambda rij, rik: (df(ab(rik))*(lam3*delta)
                             * np.exp(lam3*(ab(rij)-ab(rik))**delta)
                             - lam3 * hf(rij, rik))
    d22h = lambda rij, rik: \
        (ddf(ab(rik))*np.exp(lam3*(ab(rij)-ab(rik))**delta)
         + 2 * (lam3*delta)*np.exp(lam3*(ab(rij)-ab(rik))**delta)*df(ab(rik))
         + lam3**2 * hf(rij, rik))

    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))
    c1q = lambda rij, rik, q: (rik[:, q]/ab(rik) - rij[:, q]/ab(rij) * costh(rij, rik)) / ab(rij)
    c2q = lambda rij, rik, q: (rij[:, q]/ab(rij) - rik[:, q]/ab(rik) * costh(rij, rik)) / ab(rik)

    dc1q1t = lambda rij, rik, q, t: \
        (- c1q(rij, rik, q) * rij[:, t] \
         - rij[:, q] * c1q(rij, rik, t) \
         - costh(rij, rik) * (int(q == t) - rij[:, q]*rij[:, t]/ab(rij)**2) \
        )/ab(rij)**2
    dc2q2t = lambda rij, rik, q, t: \
        (- c2q(rij, rik, q) * rik[:, t] \
         - rik[:, q] * c2q(rij, rik, t) \
         - costh(rij, rik) * (int(q == t) - rik[:, q]*rik[:, t]/ab(rik)**2) \
        )/ab(rik)**2
    dc1q2t = lambda rij, rik, q, t: \
        ((int(q == t) - rij[:, q]*rij[:, t]/ab(rij)**2)/ab(rij)
         - c1q(rij, rik, q) * rik[:, t]/ab(rik) \
        )/ab(rik)

    Dh1q = lambda rij, rik, q: d1h(rij, rik) * (rij[:, q] / ab(rij))
    Dh2q = lambda rij, rik, q: d2h(rij, rik) * (rik[:, q] / ab(rik))

    Dh1q1t = lambda rij, rik, q, t: \
        d11h(rij, rik) * rij[:, q]/ab(rij) * rij[:, t]/ab(rij) \
         + d1h(rij, rik) * (int(q == t) - rij[:, q]/ab(rij) * rij[:, t]/ab(rij))/ab(rij)
    Dh2q2t = lambda rij, rik, q, t: \
        d22h(rij, rik) * rik[:, q]/ab(rik) * rik[:, t]/ab(rik) \
         + d2h(rij, rik) * (int(q == t) - rik[:, q]/ab(rik) * rik[:, t]/ab(rik))/ab(rik)
    Dh1q2t = lambda rij, rik, q, t: \
        d12h(rij, rik) * rij[:, q]/ab(rij) * rik[:, t]/ab(rik)

    Dg1q = lambda rij, rik, q: dg(costh(rij, rik)) * c1q(rij, rik, q)
    Dg2q = lambda rij, rik, q: dg(costh(rij, rik)) * c2q(rij, rik, q)

    Dg1q1t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c1q(rij, rik, q) * c1q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q1t(rij, rik, q, t))
    Dg2q2t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c2q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc2q2t(rij, rik, q, t))
    Dg1q2t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c1q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q2t(rij, rik, q, t))

    G = lambda rij, rik: g(costh(rij, rik)) * hf(rij, rik)
    d1qG = lambda rij, rik, q: Dh1q(rij, rik, q) * g(costh(rij, rik)) + hf(rij, rik) * Dg1q(rij, rik, q)
    d2qG = lambda rij, rik, q: Dh2q(rij, rik, q) * g(costh(rij, rik)) + hf(rij, rik) * Dg2q(rij, rik, q)

    d1q1tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh1q(rij, rik, t) + Dg1q(rij, rik, t) * Dh1q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh1q1t(rij, rik, q, t) + hf(rij, rik) * Dg1q1t(rij, rik, q, t)
    d2q2tG = lambda rij, rik, q, t: \
        Dg2q(rij, rik, q) * Dh2q(rij, rik, t) + Dg2q(rij, rik, t) * Dh2q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh2q2t(rij, rik, q, t) + hf(rij, rik) * Dg2q2t(rij, rik, q, t)
    d1q2tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh2q(rij, rik, t) + Dg2q(rij, rik, t) * Dh1q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh1q2t(rij, rik, q, t) + hf(rij, rik) * Dg1q2t(rij, rik, q, t)

    return {
        'F': F,
        'G': G,
        'd1F': d1F,
        'd2F': d2F,
        'd11F': d11F,
        'd12F': d12F,
        'd22F': d22F,
        'd1xG': lambda rij, rik: d1qG(rij, rik, 0),
        'd1yG': lambda rij, rik: d1qG(rij, rik, 1),
        'd1zG': lambda rij, rik: d1qG(rij, rik, 2),
        'd2xG': lambda rij, rik: d2qG(rij, rik, 0),
        'd2yG': lambda rij, rik: d2qG(rij, rik, 1),
        'd2zG': lambda rij, rik: d2qG(rij, rik, 2),
        'd1x1xG': lambda rij, rik: d1q1tG(rij, rik, 0, 0),
        'd1y1yG': lambda rij, rik: d1q1tG(rij, rik, 1, 1),
        'd1z1zG': lambda rij, rik: d1q1tG(rij, rik, 2, 2),
        'd1y1zG': lambda rij, rik: d1q1tG(rij, rik, 1, 2),
        'd1x1zG': lambda rij, rik: d1q1tG(rij, rik, 0, 2),
        'd1x1yG': lambda rij, rik: d1q1tG(rij, rik, 0, 1),
        'd2x2xG': lambda rij, rik: d2q2tG(rij, rik, 0, 0),
        'd2y2yG': lambda rij, rik: d2q2tG(rij, rik, 1, 1),
        'd2z2zG': lambda rij, rik: d2q2tG(rij, rik, 2, 2),
        'd2y2zG': lambda rij, rik: d2q2tG(rij, rik, 1, 2),
        'd2x2zG': lambda rij, rik: d2q2tG(rij, rik, 0, 2),
        'd2x2yG': lambda rij, rik: d2q2tG(rij, rik, 0, 1),
        'd1x2xG': lambda rij, rik: d1q2tG(rij, rik, 0, 0),
        'd1y2yG': lambda rij, rik: d1q2tG(rij, rik, 1, 1),
        'd1z2zG': lambda rij, rik: d1q2tG(rij, rik, 2, 2),
        'd1y2zG': lambda rij, rik: d1q2tG(rij, rik, 1, 2),
        'd1x2zG': lambda rij, rik: d1q2tG(rij, rik, 0, 2),
        'd1x2yG': lambda rij, rik: d1q2tG(rij, rik, 0, 1),
        'd1z2yG': lambda rij, rik: d1q2tG(rij, rik, 2, 1),
        'd1z2xG': lambda rij, rik: d1q2tG(rij, rik, 2, 0),
        'd1y2xG': lambda rij, rik: d1q2tG(rij, rik, 1, 0),
        'cutoff': r2
    }


def KumagaiTersoff():
    A = 3281.5905
    B = 121.00047
    lambda_1 = 3.2300135
    lambda_2 = 1.3457970
    eta = 1.0000000
    delta = 0.53298909
    alpha = 2.3890327
    c_1 = 0.20173476
    c_2 = 730418.72
    c_3 = 1000000.0
    c_4 = 1.0000000
    c_5 = 26.000000
    h = -0.36500000
    R_1 = 2.70
    R_2 = 3.30

    f = lambda r: np.where(
            r <= R_1, 1.0,
            np.where(r >= R_2, 0.0,
                     (1/2+(9/16) * np.cos(np.pi*(r - R_1)/(R_2 - R_1))
                      - (1/16) * np.cos(3*np.pi*(r - R_1)/(R_2 - R_1)))
                     )
                           )
    df = lambda r: np.where(
            r >= R_2, 0.0,
            np.where(r <= R_1, 0.0,
                     (3*np.pi*(3*np.sin(np.pi * (R_1 - r) / (R_1 - R_2))
                      - np.sin(3*np.pi*(R_1 - r) / (R_1 - R_2))))/(16*(R_1 - R_2))
                     )
                            )
    ddf = lambda r: np.where(
            r >= R_2, 0.0,
            np.where(r <= R_1, 0.0,
                     ((9*np.pi**2*(np.cos(3*np.pi*(R_1 - r)/(R_1 - R_2))
                       - np.cos(np.pi*(R_1 - r)/(R_1 - R_2))))/(16*(R_1 - R_2)**2))
                     )
                            )

    fR = lambda r:  A*np.exp(-lambda_1 * r)
    dfR = lambda r: -lambda_1 * fR(r)
    ddfR = lambda r: lambda_1**2 * fR(r)

    fA = lambda r: -B*np.exp(-lambda_2 * r)
    dfA = lambda r: -lambda_2 * fA(r)
    ddfA = lambda r: lambda_2**2 * fA(r)

    b = lambda xi: 1/((1+xi**eta)**(delta))
    db = lambda xi: -delta*eta*xi**(eta-1)*(xi**eta+1)**(-delta-1)
    ddb = lambda xi: delta*eta*xi**(eta - 1)*(delta + 1)*(xi**eta + 1)**(-delta - 2)

    F = lambda r, xi: f(r) * (fR(r) + b(xi) * fA(r))
    d1F = lambda r, xi: df(r) * (fR(r) + b(xi) * fA(r)) + f(r) * (dfR(r) + b(xi) * dfA(r))
    d2F = lambda r, xi: f(r) * fA(r) * db(xi)
    d11F = lambda r, xi: f(r) * (ddfR(r) + b(xi) * ddfA(r)) + 2 * df(r) * (dfR(r) + b(xi) * dfA(r)) + ddf(r) * (fR(r) + b(xi) * fA(r))
    d22F = lambda r, xi:  f(r) * fA(r) * ddb(xi)
    d12F = lambda r, xi: f(r) * dfA(r) * db(xi) + fA(r) * df(r) * db(xi)

    g = lambda cost: c_1 + (1 + c_4*np.exp(-c_5*(h-cost)**2)) * \
                           ((c_2*(h-cost)**2)/(c_3 + (h-cost)**2))
    dg = lambda cost: 2*c_2*(cost - h)*(
            (c_3 + (cost - h)**2) *
            (-c_4*c_5*(cost - h)**2 + c_4 +
             np.exp(c_5*(cost - h)**2)) -
            (c_4 + np.exp(c_5*(cost - h)**2))
            * (cost - h)**2) * np.exp(-c_5*(cost - h)**2)/(c_3 + (cost - h)**2)**2
    ddg = lambda cos_theta: \
        (2*c_2*((c_3 + (cos_theta - h)**2)**2
                * (2*c_4*c_5**2*(cos_theta - h)**4
                - 5*c_4*c_5*(cos_theta - h)**2 + c_4
                + np.exp(c_5*(cos_theta - h)**2))
                + (c_3 + (cos_theta - h)**2)*(cos_theta - h)**2
                * (4*c_4*c_5*(cos_theta - h)**2
                - 5*c_4 - 5*np.exp(c_5*(cos_theta - h)**2))
                + 4*(c_4 + np.exp(c_5*(cos_theta - h)**2))*(cos_theta - h)**4)
              * np.exp(-c_5*(cos_theta - h)**2)/(c_3 + (cos_theta - h)**2)**3
         )

    hf = lambda rij, rik: f(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik)))
    d1h = lambda rij, rik: alpha * hf(rij, rik)
    d2h = lambda rij, rik: \
        - alpha * hf(rij, rik) \
        + df(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik)))
    d11h = lambda rij, rik: alpha**2 * hf(rij, rik)
    d12h = lambda rij, rik: alpha * d2h(rij, rik)
    d22h = lambda rij, rik: \
         - alpha * ( 2 * df(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik))) \
         - alpha * hf(rij, rik)) \
         + ddf(ab(rik)) * np.exp(alpha * (ab(rij) - ab(rik)))

    costh = lambda rij, rik: np.sum(rij*rik, axis=1) / (ab(rij)*ab(rik))
    c1q = lambda rij, rik, q: (rik[:, q]/ab(rik) - rij[:, q]/ab(rij) * costh(rij, rik)) / ab(rij)
    c2q = lambda rij, rik, q: (rij[:, q]/ab(rij) - rik[:, q]/ab(rik) * costh(rij, rik)) / ab(rik)

    dc1q1t = lambda rij, rik, q, t: \
        (- c1q(rij, rik, q) * rij[:, t] \
         - rij[:, q] * c1q(rij, rik, t) \
         - costh(rij, rik) * (int(q == t) - rij[:, q]*rij[:, t]/ab(rij)**2) \
        )/ab(rij)**2
    dc2q2t = lambda rij, rik, q, t: \
        (- c2q(rij, rik, q) * rik[:, t] \
         - rik[:, q] * c2q(rij, rik, t) \
         - costh(rij, rik) * (int(q == t) - rik[:, q]*rik[:, t]/ab(rik)**2) \
        )/ab(rik)**2
    dc1q2t = lambda rij, rik, q, t: \
        ((int(q == t) - rij[:, q]*rij[:, t]/ab(rij)**2)/ab(rij)
         - c1q(rij, rik, q) * rik[:, t]/ab(rik) \
        )/ab(rik)

    Dh1q = lambda rij, rik, q: d1h(rij, rik) * (rij[:, q] / ab(rij))
    Dh2q = lambda rij, rik, q: d2h(rij, rik) * (rik[:, q] / ab(rik))

    Dh1q1t = lambda rij, rik, q, t: \
        d11h(rij, rik) * rij[:, q]/ab(rij) * rij[:, t]/ab(rij) \
         + d1h(rij, rik) * (int(q == t) - rij[:, q]/ab(rij) * rij[:, t]/ab(rij))/ab(rij)
    Dh2q2t = lambda rij, rik, q, t: \
        d22h(rij, rik) * rik[:, q]/ab(rik) * rik[:, t]/ab(rik) \
         + d2h(rij, rik) * (int(q == t) - rik[:, q]/ab(rik) * rik[:, t]/ab(rik))/ab(rik)
    Dh1q2t = lambda rij, rik, q, t: \
        d12h(rij, rik) * rij[:, q]/ab(rij) * rik[:, t]/ab(rik)

    Dg1q = lambda rij, rik, q: dg(costh(rij, rik)) * c1q(rij, rik, q)
    Dg2q = lambda rij, rik, q: dg(costh(rij, rik)) * c2q(rij, rik, q)

    Dg1q1t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c1q(rij, rik, q) * c1q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q1t(rij, rik, q, t))
    Dg2q2t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c2q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc2q2t(rij, rik, q, t))
    Dg1q2t = lambda rij, rik, q, t: \
        (ddg(costh(rij, rik)) * c1q(rij, rik, q) * c2q(rij, rik, t)
         + dg(costh(rij, rik)) * dc1q2t(rij, rik, q, t))

    G = lambda rij, rik: g(costh(rij, rik)) * hf(rij, rik)
    d1qG = lambda rij, rik, q: Dh1q(rij, rik, q) * g(costh(rij, rik)) + hf(rij, rik) * Dg1q(rij, rik, q)
    d2qG = lambda rij, rik, q: Dh2q(rij, rik, q) * g(costh(rij, rik)) + hf(rij, rik) * Dg2q(rij, rik, q)

    d1q1tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh1q(rij, rik, t) + Dg1q(rij, rik, t) * Dh1q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh1q1t(rij, rik, q, t) + hf(rij, rik) * Dg1q1t(rij, rik, q, t)
    d2q2tG = lambda rij, rik, q, t: \
        Dg2q(rij, rik, q) * Dh2q(rij, rik, t) + Dg2q(rij, rik, t) * Dh2q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh2q2t(rij, rik, q, t) + hf(rij, rik) * Dg2q2t(rij, rik, q, t)
    d1q2tG = lambda rij, rik, q, t: \
        Dg1q(rij, rik, q) * Dh2q(rij, rik, t) + Dg2q(rij, rik, t) * Dh1q(rij, rik, q) \
        + g(costh(rij, rik)) * Dh1q2t(rij, rik, q, t) + hf(rij, rik) * Dg1q2t(rij, rik, q, t)

    return {
        'F': F,
        'G': G,
        'd1F': d1F,
        'd2F': d2F,
        'd11F': d11F,
        'd12F': d12F,
        'd22F': d22F,
        'd1xG': lambda rij, rik: d1qG(rij, rik, 0),
        'd1yG': lambda rij, rik: d1qG(rij, rik, 1),
        'd1zG': lambda rij, rik: d1qG(rij, rik, 2),
        'd2xG': lambda rij, rik: d2qG(rij, rik, 0),
        'd2yG': lambda rij, rik: d2qG(rij, rik, 1),
        'd2zG': lambda rij, rik: d2qG(rij, rik, 2),
        'd1x1xG': lambda rij, rik: d1q1tG(rij, rik, 0, 0),
        'd1y1yG': lambda rij, rik: d1q1tG(rij, rik, 1, 1),
        'd1z1zG': lambda rij, rik: d1q1tG(rij, rik, 2, 2),
        'd1y1zG': lambda rij, rik: d1q1tG(rij, rik, 1, 2),
        'd1x1zG': lambda rij, rik: d1q1tG(rij, rik, 0, 2),
        'd1x1yG': lambda rij, rik: d1q1tG(rij, rik, 0, 1),
        'd2x2xG': lambda rij, rik: d2q2tG(rij, rik, 0, 0),
        'd2y2yG': lambda rij, rik: d2q2tG(rij, rik, 1, 1),
        'd2z2zG': lambda rij, rik: d2q2tG(rij, rik, 2, 2),
        'd2y2zG': lambda rij, rik: d2q2tG(rij, rik, 1, 2),
        'd2x2zG': lambda rij, rik: d2q2tG(rij, rik, 0, 2),
        'd2x2yG': lambda rij, rik: d2q2tG(rij, rik, 0, 1),
        'd1x2xG': lambda rij, rik: d1q2tG(rij, rik, 0, 0),
        'd1y2yG': lambda rij, rik: d1q2tG(rij, rik, 1, 1),
        'd1z2zG': lambda rij, rik: d1q2tG(rij, rik, 2, 2),
        'd1y2zG': lambda rij, rik: d1q2tG(rij, rik, 1, 2),
        'd1x2zG': lambda rij, rik: d1q2tG(rij, rik, 0, 2),
        'd1x2yG': lambda rij, rik: d1q2tG(rij, rik, 0, 1),
        'd1z2yG': lambda rij, rik: d1q2tG(rij, rik, 2, 1),
        'd1z2xG': lambda rij, rik: d1q2tG(rij, rik, 2, 0),
        'd1y2xG': lambda rij, rik: d1q2tG(rij, rik, 1, 0),
        'cutoff': R_2
    }