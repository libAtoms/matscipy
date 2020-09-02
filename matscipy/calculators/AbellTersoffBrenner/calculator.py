from ase.calculators.calculator import Calculator
from matscipy.neighbours import neighbour_list, first_neighbours, triplet_list
import numpy as np


class AbellTersoffBrenner(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {}
    name = 'ThreeBodyPotential'

    def __init__(self, g, f_c, xi, b, psi_rep, psi_att, exp,
                 df_c_r, dpsi_rep_r, dpsi_att_r, db_xi, dg_costheta,
                 dexp_r_ij, dexp_r_ik,
                 ddf_c_r_r, ddb_xi_xi, ddexp_r_ij_r_ik, ddexp_r_ij_r_ij,
                 ddg_costheta_costheta, ddU_r_r, ddV_r_r):
        Calculator.__init__(self)
        self.g = g
        self.f_c = f_c
        self.xi = xi
        self.b = b
        self.psi_rep = psi_rep
        self.psi_att = psi_att
        self.cos_theta = lambda abs_dr_ij, abs_dr_ik, dr_ijc, dr_ikc: \
            np.einsum('ij,ij->i', dr_ijc, dr_ikc)/(abs_dr_ij*abs_dr_ik)
        self.h = lambda r_ij, r_ik:  self.f_c(r_ik) * self.exp(r_ij, r_ik)

        # forces
        self.exp = exp
        self.dg_costheta = dg_costheta
        self.df_c_r = df_c_r
        self.dpsi_rep_r = dpsi_rep_r
        self.dpsi_att_r = dpsi_att_r
        self.dexp_r_ij = dexp_r_ij
        self.dexp_r_ik = dexp_r_ik
        self.db_xi = db_xi
        self.dha = lambda r_ij, r_ik: self.f_c(r_ik)*self.dexp_r_ij(r_ij, r_ik)
        self.dhb = lambda r_ij, r_ik: self.f_c(r_ik)*self.dexp_r_ik(r_ij, r_ik)\
                + self.df_c_r(r_ik) * self.exp(r_ij, r_ik)

        # hessian
        self.ddf_c_r_r = ddf_c_r_r
        self.ddb_xi_xi = ddb_xi_xi
        self.ddexp_r_ij_r_ik = ddexp_r_ij_r_ik
        self.ddexp_r_ij_r_ij = ddexp_r_ij_r_ij
        self.ddg_costheta_costheta = ddg_costheta_costheta
        self.ddU_r_r = ddU_r_r
        self.ddV_r_r = ddV_r_r
        self.ddhaa = lambda r_ij, r_ik: \
            self.f_c(r_ik)*self.ddexp_r_ij_r_ik(r_ij, r_ik)
        self.ddhab = lambda r_ij, r_ik: self.df_c_r(r_ik)*self.dexp_r_ij(r_ij, r_ik)\
                + self.f_c(r_ik)*self.ddexp_r_ij_r_ik(r_ij, r_ik)
        self.ddhbb = lambda r_ij, r_ik: self.ddf_c_r_r(r_ik)*self.exp(r_ij, r_ik)\
                +2*self.dexp_r_ij(r_ij, r_ik)*self.df_c_r(r_ik)\
                +self.f_c(r_ik)*self.ddexp_r_ij_r_ij(r_ij, r_ik)

    def __call__(self):
        return None

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        nat = len(self.atoms)

        # get neighbour and triplet lists
        i_n, j_n, self.r_p, dr_pc = neighbour_list('ijdD', atoms=atoms,
                                                   cutoff=self.cutoff)
        first_i = first_neighbours(nat, i_n)

        ij_t, ik_t = triplet_list(first_i)

        # calculate energy

        # pair terms
        f_c_p = self.f_c(self.r_p)

        U_p = self.psi_att(self.r_p, f_c_p)
        V_p = self.psi_rep(self.r_p, f_c_p)

        # triplet terms
        if ij_t.size > 0:
            self.cos_theta_t = self.cos_theta(self.r_p[ij_t], self.r_p[ik_t],
                                              dr_pc[ij_t], dr_pc[ik_t])
            g_t = self.g(self.cos_theta_t)

            xi_t = self.xi(self.r_p[ij_t], self.r_p[ik_t], f_c_p[ik_t], g_t)
            xi_p = np.bincount(ij_t, weights=xi_t, minlength=nat)
            b_p = self.b(xi_p)
        else:
            b_p = np.ones_like(i_n)

        e_n = V_p + b_p * U_p

        epot = 1/2*np.sum(e_n)

        # calculate forces

        # '-' is due to a different definition of \v{r}_ij
        norm_pc = -dr_pc/self.r_p.reshape(-1, 1)

        # pair terms
        df_c_r_p = self.df_c_r(self.r_p)

        f_att_p = self.dpsi_att_r(f_c_p, self.r_p, df_c_r_p)
        f_rep_p = self.dpsi_rep_r(f_c_p, self.r_p, df_c_r_p)

        f_pc = norm_pc*(f_rep_p + b_p*f_att_p).reshape(-1, 1)

        # triplet terms

        if ij_t.size > 0:

            db_p = self.db_xi(xi_p)

            f_aa_p = np.bincount(ij_t, weights=(g_t*U_p[ij_t]*db_p[ij_t] *
                                                self.dha(self.r_p[ij_t],
                                                         self.r_p[ik_t])))
            f_ab_p = np.bincount(ij_t, weights=(g_t*U_p[ik_t]*db_p[ik_t] *
                                                self.dhb(self.r_p[ik_t],
                                                         self.r_p[ij_t])))

            f_a_p_c = norm_pc*(f_aa_p + f_ab_p).reshape(-1, 1)

            c_tc = (norm_pc[ik_t] - self.cos_theta_t.reshape(-1, 1)
                    * norm_pc[ij_t])/(self.r_p[ij_t].reshape(-1, 1))

            dg_t = self.dg_costheta(self.cos_theta_t)

            f_ba_tc = c_tc*(dg_t*U_p[ij_t]*db_p[ij_t]*self.h(self.r_p[ij_t],
                            self.r_p[ik_t])).reshape(-1, 1)
            f_bb_tc = c_tc*(dg_t*U_p[ik_t]*db_p[ik_t]*self.h(self.r_p[ik_t],
                            self.r_p[ij_t])).reshape(-1, 1)
            # TODO: use symmetry; proabably not existent
            f_b_tc = f_ba_tc + f_bb_tc

            fx_b_p = np.bincount(ij_t, weights=f_b_tc[:, 0])
            fy_b_p = np.bincount(ij_t, weights=f_b_tc[:, 1])
            fz_b_p = np.bincount(ij_t, weights=f_b_tc[:, 2])

            fx_p = f_pc[:, 0] + f_a_p_c[:, 0] + fx_b_p
            fy_p = f_pc[:, 1] + f_a_p_c[:, 1] + fy_b_p
            fz_p = f_pc[:, 2] + f_a_p_c[:, 2] + fz_b_p
        else:
            fx_p = f_pc[:, 0]
            fy_p = f_pc[:, 1]
            fz_p = f_pc[:, 2]

        fx_n = -1/2*(np.bincount(i_n, weights=fx_p) -
                     np.bincount(j_n, weights=fx_p))
        fy_n = -1/2*(np.bincount(i_n, weights=fy_p) -
                     np.bincount(j_n, weights=fy_p))
        fz_n = -1/2*(np.bincount(i_n, weights=fz_p) -
                     np.bincount(j_n, weights=fz_p))

        f_n = np.transpose([fx_n, fy_n, fz_n])

        self.results = {'energy': epot, 'forces': f_n}

    def calculate_hessian_matrix(self, atoms, H_format="dense"):
        """
        Calculate the Hessian matrix for a bond order potential.
        For an atomic configuration with N atoms in d dimensions the hessian matrix is a symmetric, hermitian matrix
        with a shape of (d*N,d*N). The matrix is in general a sparse matrix, which consists of dense blocks of shape (d,d), which
        are the mixed second derivatives.

        Parameters
        ----------
        atoms: ase.Atoms
            Atomic configuration in a local or global minima.
        H_format: "dense" or "sparse"
            Output format of the hessian matrix.
            The format "sparse" is only possible if matscipy was build with scipy.
        limits: list [atomID_low, atomID_up]
            Calculate the Hessian matrix only for the given atom IDs.
            If limits=[5,10] the Hessian matrix is computed for atom IDs 5,6,7,8,9 only.
            The Hessian matrix will have the full shape dim(3*N,3*N) where N is the number of atoms.
            This ensures correct indexing of the data.
        Restrictions
        ----------
        This method is currently only implemented for three dimensional systems
        """

        if H_format == "sparse":
            try:
                from scipy.sparse import bsr_matrix, vstack, hstack
            except ImportError:
                raise ImportError(
                    "Import error: Can not output the hessian matrix since scipy.sparse could not be loaded!")

        nat = len(self.atoms)

        # get neighbour and triplet lists
        i_n, j_n, self.r_p, dr_pc = neighbour_list('ijdD', atoms=atoms,
                                                   cutoff=self.cutoff)
        first_i = first_neighbours(nat, i_n)

        ij_t, ik_t = triplet_list(first_i)

        norm_pc = -dr_pc/self.r_p.reshape(-1, 1)

        kron_pcc = norm_pc.reshape(-1, 3, 1)*norm_pc.reshape(-1, 1, 3)
        kron_tcc = norm_pc[ij_t].reshape(-1, 3, 1)*norm_pc[ik_t].reshape(-1, 1, 3)

        f_c_p = self.f_c(self.r_p)
        df_c_r_p = self.df_c_r(self.r_p)
        ddf_c_r_r_p = self.ddf_c_r_r(self.r_p)

        # print('ddfcr', ddf_c_r_r_p)

        f_att_p = self.dpsi_att_r(f_c_p, self.r_p, df_c_r_p)
        f_rep_p = self.dpsi_rep_r(f_c_p, self.r_p, df_c_r_p)

        k_att_p = self.ddU_r_r(self.r_p, f_c_p, df_c_r_p, ddf_c_r_r_p)
        k_rep_p = self.ddV_r_r(self.r_p, f_c_p, df_c_r_p, ddf_c_r_r_p)

        # print(k_att_p, k_rep_p)
        # print(kron_p)

        if ij_t.size > 0:
            # triplet term

            self.cos_theta_t = self.cos_theta(self.r_p[ij_t], self.r_p[ik_t],
                                              dr_pc[ij_t], dr_pc[ik_t])

            ca_tc = (norm_pc[ij_t] - self.cos_theta_t.reshape(-1, 1)  # il, ij
                     * norm_pc[ik_t])/(self.r_p[ik_t].reshape(-1, 1))
            cb_tc = (norm_pc[ik_t] - self.cos_theta_t.reshape(-1, 1)  # ij, il
                     * norm_pc[ij_t])/(self.r_p[ij_t].reshape(-1, 1))

            g_t = self.g(self.cos_theta_t)

            xi_t = self.xi(self.r_p[ij_t], self.r_p[ik_t], f_c_p[ik_t], g_t)
            xi_p = np.bincount(ij_t, weights=xi_t, minlength=nat)
            b_p = self.b(xi_p)
            db_p = self.db_xi(xi_p)

            dg_t = self.dg_costheta(self.cos_theta_t)

            f_c_p = self.f_c(self.r_p)
            U_p = self.psi_att(self.r_p, f_c_p)
            ddb_p = self.ddb_xi_xi(xi_p)
            ddg_t = self.ddg_costheta_costheta(self.cos_theta_t)
            h_t_ab = self.h(self.r_p[ij_t], self.r_p[ik_t])
            h_t_ba = self.h(self.r_p[ik_t], self.r_p[ij_t])
            dha_t_ab = self.dha(self.r_p[ij_t], self.r_p[ik_t])
            dha_t_ba = self.dha(self.r_p[ik_t], self.r_p[ij_t])
            dhb_t_ab = self.dhb(self.r_p[ij_t], self.r_p[ik_t])
            dhb_t_ba = self.dhb(self.r_p[ik_t], self.r_p[ij_t])
            ddhaa_t_ab = self.ddhaa(self.r_p[ij_t], self.r_p[ik_t])
            ddhab_t_ab = self.ddhab(self.r_p[ij_t], self.r_p[ik_t])
            ddhbb_t_ba = self.ddhbb(self.r_p[ik_t], self.r_p[ij_t])
            ddhab_t_ba = self.ddhab(self.r_p[ik_t], self.r_p[ij_t])
            
            # print('g', g_t, dg_t, ddg_t)

            A_t = U_p[ij_t] * db_p[ij_t] * dha_t_ab + U_p[ik_t] * db_p[ik_t] * dhb_t_ba
            B_t = U_p[ij_t] * ddb_p[ij_t] * dha_t_ab + U_p[ik_t] * ddb_p[ik_t] * dhb_t_ba
            C_t = U_p[ij_t] * db_p[ij_t] * h_t_ab + U_p[ik_t] * db_p[ik_t] * h_t_ba
            D_t = U_p[ij_t] * ddb_p[ij_t] * h_t_ab + U_p[ik_t] * ddb_p[ik_t] * h_t_ba
            E_t = U_p[ij_t] * db_p[ij_t] * dhb_t_ab + U_p[ik_t] * db_p[ik_t] * dha_t_ba
            F_t = U_p[ij_t] * db_p[ij_t] * ddhab_t_ab + U_p[ik_t] * db_p[ik_t] * ddhab_t_ba
            X_tc = (h_t_ab.reshape(-1, 1) * cb_tc
                    + dha_t_ab.reshape(-1, 1) * norm_pc[ij_t]) * dg_t.reshape(-1, 1)
            Xx_p = np.bincount(ij_t, weights=X_tc[:, 0])
            Xy_p = np.bincount(ij_t, weights=X_tc[:, 1])
            Xz_p = np.bincount(ij_t, weights=X_tc[:, 2])
            X_pc = np.transpose([Xx_p, Xy_p, Xz_p])
            Y_tc = (h_t_ab.reshape(-1, 1) * ca_tc
                    + dhb_t_ab.reshape(-1, 1) * norm_pc[ik_t]) * dg_t.reshape(-1, 1)
            Yx_p = np.bincount(ij_t, weights=Y_tc[:, 0])
            Yy_p = np.bincount(ij_t, weights=Y_tc[:, 1])
            Yz_p = np.bincount(ij_t, weights=Y_tc[:, 2])
            Y_pc = np.transpose([Yx_p, Yy_p, Yz_p])

            chi_tcc = ((cb_tc.reshape(-1, 3, 1)*norm_pc[ij_t].reshape(-1, 1, 3)).T*1/self.r_p[ij_t]
                       + (norm_pc[ik_t].reshape(-1, 3, 1)*norm_pc[ij_t].reshape(-1, 1, 3)).T*1/self.r_p[ij_t]**2
                       + self.cos_theta_t*((np.eye(3, dtype=kron_pcc.dtype) - kron_pcc)[ij_t].T*1/self.r_p[ij_t]
                       - kron_pcc[ij_t].T*1/self.r_p[ij_t]**2)).T

            zeta_tcc = ((np.eye(3, dtype=kron_pcc.dtype) - norm_pc[ik_t].reshape(-1, 3, 1)*norm_pc[ik_t].reshape(-1, 1, 3)).T*1/(self.r_p[ik_t]*self.r_p[ij_t])
                        - (ca_tc.reshape(-1, 3, 1)*norm_pc[ij_t].reshape(-1, 1, 3)).T*1/self.r_p[ij_t]).T

            """
            chi_tcc = ((cb_tc.reshape(-1, 3, 1)*norm_pc[ij_t].reshape(-1, 1, 3))*(1/self.r_p[ij_t]).reshape(-1, 1, 1) \
                       + (norm_pc[ik_t].reshape(-1, 3, 1)*norm_pc[ij_t].reshape(-1, 1, 3))*(1/self.r_p[ij_t]**2).reshape(-1, 1, 1) \
                       + ((np.eye(3, dtype=kron_pcc.dtype) - kron_pcc)[ij_t]*(self.cos_theta_t/self.r_p[ij_t]).reshape(-1, 1, 1) \
                       - kron_pcc[ij_t]*(1/self.r_p[ij_t]**2).reshape(-1, 1, 1)))

            zeta_tcc = ((np.eye(3, dtype=kron_pcc.dtype) - norm_pc[ik_t].reshape(-1,3,1)*norm_pc[ik_t].reshape(-1,1,3))*(1/(self.r_p[ik_t]*self.r_p[ij_t])).reshape(-1,1,1) \
                        - (ca_tc.reshape(-1,3,1)*norm_pc[ij_t].reshape(-1,1,3))*(1/self.r_p[ij_t]).reshape(-1,1,1))

            """
            dim_tcc = -((((np.eye(3, dtype=kron_pcc.dtype) + kron_pcc)[ij_t].T*(g_t/self.r_p[ij_t])
                          + dg_t*(norm_pc[ij_t].reshape(-1, 3, 1)*ca_tc.reshape(-1, 1, 3)
                          + norm_pc[ij_t].reshape(-1, 3, 1)*cb_tc.reshape(-1, 1, 3)
                          + cb_tc.reshape(-1, 3, 1)*norm_pc[ij_t].reshape(-1, 1, 3)).T)*A_t).T
                        + ((g_t*B_t).reshape(-1, 1)*norm_pc[ij_t]+(dg_t*D_t).reshape(-1, 1)*cb_tc).reshape(-1, 3, 1)*X_pc[ij_t].reshape(-1, 1, 3)
                        + ((g_t*B_t).reshape(-1, 1)*norm_pc[ij_t]+(dg_t*D_t).reshape(-1, 1)*cb_tc).reshape(-1, 3, 1)*Y_pc[ij_t].reshape(-1, 1, 3)
                        + (ddg_t*C_t*(cb_tc.reshape(-1, 3, 1)*ca_tc.reshape(-1, 1, 3)+cb_tc.reshape(-1, 3, 1)*cb_tc.reshape(-1, 1, 3)).T).T
                        + (g_t*(U_p[ij_t]*db_p[ij_t]*ddhaa_t_ab+U_p[ik_t]*db_p[ik_t]*ddhbb_t_ba)*kron_pcc[ij_t].T).T
                        + ((g_t*f_att_p[ij_t]*db_p[ij_t]*dha_t_ab)*kron_pcc[ij_t].T).T
                        + (g_t*F_t*kron_tcc.T).T
                        + (g_t*f_att_p[ik_t]*db_p[ik_t]*dhb_t_ba*kron_tcc.T).T \
                        + (dg_t*f_att_p[ij_t]*db_p[ij_t]*h_t_ab*(cb_tc.reshape(-1, 3, 1)*norm_pc[ij_t].reshape(-1, 1, 3)).T).T
                        + (dg_t*f_att_p[ik_t]*db_p[ik_t]*h_t_ba*(cb_tc.reshape(-1, 3, 1)*norm_pc[ik_t].reshape(-1, 1, 3)).T).T
                        + (dg_t*E_t*(cb_tc.reshape(-1, 3, 1)*norm_pc[ik_t].reshape(-1, 1, 3)).T).T \
                        # additional term from 'K_pair' derivative
                        + (f_att_p[ij_t]*db_p[ij_t]*(norm_pc[ij_t].reshape(-1, 3, 1)*X_pc[ij_t].reshape(-1, 1, 3)).T).T \
                        + (f_att_p[ij_t]*db_p[ij_t]*(norm_pc[ij_t].reshape(-1, 3, 1)*Y_pc[ij_t].reshape(-1, 1, 3)).T).T \
                        ) + ((chi_tcc-zeta_tcc).T*dg_t*C_t).T

            djm_tcc = ((dg_t*A_t*(norm_pc[ij_t].reshape(-1, 3, 1)*ca_tc.reshape(-1, 1, 3)).T).T
                       + ((g_t*B_t).reshape(-1, 1)*norm_pc[ij_t]+(dg_t*D_t).reshape(-1, 1)*cb_tc).reshape(-1, 3, 1)*Y_pc[ij_t].reshape(-1, 1, 3)
                       + (ddg_t*C_t*(cb_tc.reshape(-1, 3, 1)*ca_tc.reshape(-1, 1, 3)).T).T
                       + (g_t*F_t*kron_tcc.T).T
                       + (g_t*f_att_p[ik_t]*db_p[ik_t]*dhb_t_ba*kron_tcc.T).T
                       + (dg_t*f_att_p[ik_t]*db_p[ik_t]*h_t_ba*(cb_tc.reshape(-1, 3, 1)*norm_pc[ik_t].reshape(-1, 1, 3)).T).T
                       + (dg_t*E_t*(cb_tc.reshape(-1, 3, 1)*norm_pc[ik_t].reshape(-1, 1, 3)).T).T \
                       # additional term from 'K_pair' derivative
                       + (f_att_p[ij_t]*db_p[ij_t]*(norm_pc[ij_t].reshape(-1, 3, 1)*Y_pc[ij_t].reshape(-1, 1, 3)).T).T \
                       ) + (zeta_tcc.T*dg_t*C_t).T
            # print('jm', djm_tcc)
        else:
            b_p = np.ones_like(i_n)
            # dxi_pcc = np.zeros([nat, 3, 3])

        dF_pcc = - kron_pcc*(k_rep_p-(f_rep_p/self.r_p)+b_p*(k_att_p-(f_att_p/self.r_p))).reshape(-1, 1, 1)\
                 - np.eye(3, dtype=kron_pcc.dtype)*((b_p*f_att_p+f_rep_p)/self.r_p).reshape(-1, 1, 1) \

        # K_pcc = 1/2*(dF_pcc[i_n]-dF_pcc[j_n])

        # print('a',dF_pcc)
        # dF_tcc = dF_pcc[ij_t]
        # print('b',dF_pcc)

        # print(dxi_pcc.shape, dF_pcc.shape)

        # print(djm_tcc.T, dim_tcc.T)

        H_pcc = np.empty((i_n.shape[0], 3, 3))
        # H_pcc_a = np.empty((i_n.shape[0], 3, 3))

        if ij_t.size > 0:
            for x in range(3):
                for y in range(3):
                    H_pcc[:, x, y] = dF_pcc[:, x, y]+np.bincount(ij_t, weights=dim_tcc[:, x, y])+np.bincount(ik_t, weights=djm_tcc[:, x, y])
                    # H_pcc[:,x,y] = dF_pcc[:, x, y]+np.bincount(ij_t, weights=dim_tcc[:, x, y])+np.bincount(ik_t, weights=djm_tcc[:, x, y])
                    # H_pcc_a[:,x,y] = 1/2*((np.bincount(ij_t, weights=dF_tcc[:, x, y])+np.bincount(ij_t, weights=dim_tcc[:, x, y])+np.bincount(ik_t, weights=djm_tcc[:, x, y]))-(np.bincount(ik_t, weights=dF_tcc[:, x, y])+np.bincount(ik_t, weights=dim_tcc[:, x, y])+np.bincount(ij_t, weights=djm_tcc[:, x, y])))
                    # H_pcc[:,x,y] = 1/2*(H_pcc[:,x,y][i_n]-H_pcc[:,x,y][j_n])
                    # a = np.bincount(ij_t, weights=dim_tcc[:, x, y])
                    # b = np.bincount(ik_t, weights=djm_tcc[:, x, y])
                    # c = dF_pcc[:, x, y]
                    # H_pcc[:,x,y] = 1/2*(c[i_n]+c[j_n])+1/2*(a[i_n]+a[j_n])+1/2*(b[i_n]+b[j_n])

        else:
            for x in range(3):
                for y in range(3):
                    H_pcc[:, x, y] = dF_pcc[:, x, y]

        Hdiag_ncc = np.empty((nat, 3, 3))
        for x in range(3):
            for y in range(3):
                Hdiag_ncc[:, x, y] = - \
                        np.bincount(i_n, weights=H_pcc[:, x, y])  # 1/2*(np.bincount(i_n, weights=H_pcc[:, x, y])+np.bincount(j_n, weights=H_pcc[:, x, y]))

        H = np.zeros((3*nat, 3*nat))
        for atom in range(len(i_n)):
            H[3*i_n[atom]:3*i_n[atom]+3,
              3*j_n[atom]:3*j_n[atom]+3] += H_pcc[atom]  # 1/2*(H_pcc[atom]-H_pcc[j_n[atom]])

        Hdiag_pcc = np.zeros((3*nat, 3*nat))
        for atom in range(nat):
            Hdiag_pcc[3*atom:3*atom+3,
                      3*atom:3*atom+3] += Hdiag_ncc[atom]

        H += Hdiag_pcc

        return H


class KumagaiTersoff(AbellTersoffBrenner):
    def __init__(self, A, B, lambda_1, lambda_2, eta, delta, alpha, beta,
                 c_1, c_2, c_3, c_4, c_5, h, R_1, R_2, __ref__=None, el=None):
        self.__ref__ = __ref__
        self.cutoff = R_2
        super().__init__(
            g=lambda cos_theta: c_1 + (1 + c_4*np.exp(-c_5*(h-cos_theta)**2)) *\
                                ((c_2*(h-cos_theta)**2)/(c_3 + (h-cos_theta)**2)),

            f_c=lambda r: np.where(r <= R_1, 1.0,  np.where(r >= R_2, 0.0, (1/2+(9/16) \
                                                *np.cos(np.pi*(r - R_1)/(R_2 - R_1)) - (1/16) \
                                                    *np.cos(3*np.pi*(r-R_1)/(R_2-R_1))))),

            # f_c =  lambda r : np.where(r <= R_1, 1.0,  np.where(r >= R_2, 0.0, \
            #                    (1/2*(1+np.cos(np.pi*(r - R_1)/(R_2 - R_1)))))),


            xi=lambda r_ij, r_ik, f_c, g: f_c * g * np.exp(alpha*(r_ij-r_ik)**beta),
            exp=lambda r_ij, r_ik: np.exp(alpha*(r_ij-r_ik)**beta),
            b=lambda xi_ij:  1/((1+xi_ij**eta)**(delta)),
            psi_rep=lambda r_ij, f_c: f_c*(A*np.exp(-lambda_1 * r_ij)),
            psi_att=lambda r_ij, f_c: -f_c*(B*np.exp(-lambda_2 * r_ij)),

            # forces
            df_c_r=lambda r: np.where(r >= R_2, 0.0,\
                                      np.where(r <= R_1, 0.0,\
                                               (3*np.pi*(3*np.sin(np.pi *\
                                                         (R_1 - r)/(R_1 - R_2))\
                                        - np.sin(3*np.pi*(R_1 - r)/(R_1 - R_2))))/(16*(R_1 - R_2)))),

            # df_c_r = lambda r: np.where(r >= R_2, 0.0, np.where(r <= R_1, 0.0, np.pi*np.sin(np.pi*(R_1 - r)/(R_1 - R_2))/(2*(R_1 - R_2)))),

            dpsi_rep_r=lambda f_c, r, df_c_r:  A*np.exp(-lambda_1*r)*(df_c_r - f_c*lambda_1),

            dpsi_att_r=lambda f_c, r, df_c_r: -B*np.exp(-lambda_2*r)*(df_c_r - f_c*lambda_2),

            db_xi=lambda xi: -delta*eta*xi**(eta-1)*(xi**eta+1)**(-delta-1),

            dg_costheta=lambda cos_theta: 2*c_2*(cos_theta - h)*((c_3 + (cos_theta - h)**2) * \
                                                                 (-c_4*c_5*(cos_theta - h)**2 + c_4 + \
                                                                  np.exp(c_5*(cos_theta - h)**2)) - \
                                                                 (c_4 + np.exp(c_5*(cos_theta - h)**2)) \
                                                                 * (cos_theta - h)**2)*np.exp(-c_5*(cos_theta - \
                                                                                              h)**2)/(c_3 + (cos_theta - h)**2)**2,
            dexp_r_ij=lambda r_ij, r_ik: (alpha*beta)*np.exp(alpha*(r_ij-r_ik)**beta),  # (alpha*beta*(r_ij-r_ik)**(beta-1))*np.exp(alpha*(r_ij-r_ik)**beta),
            dexp_r_ik=lambda r_ij, r_ik: -(alpha*beta)*np.exp(alpha*(r_ij-r_ik)**beta),  # -(alpha*beta*(r_ij-r_ik)**(beta-1))*np.exp(alpha*(r_ij-r_ik)**beta),
            
            # hessian
            ddf_c_r_r=lambda r: np.where(r >= R_2, 0.0,\
                                         np.where(r <= R_1, 0.0,\
                                                  ((9*np.pi**2*(np.cos(3*np.pi*(R_1 - r)/(R_1 - R_2)) \
                                                                - np.cos(np.pi*(R_1 - r)/(R_1 - R_2))))/(16*(R_1 - R_2)**2)))),
            #ddb_xi_xi=lambda xi: delta*eta*xi**(eta - 2)*(xi**eta + 1)**(-delta)\
            #                    *(delta*eta*xi**eta - eta + xi**eta + 1)/(xi**(2*eta) + 2*xi**eta + 1),
            ddb_xi_xi=lambda xi: delta*eta**2*xi**(eta - 1)*(delta + 1)*(xi**eta + 1)**(-delta - 2),
            #ddexp_r_ij_r_ik=lambda r_ij, r_ik: alpha*beta*(r_ij - r_ik)**(beta - 2)\
            #                    *(-alpha*beta*(r_ij - r_ik)**beta - beta + 1)*np.exp(alpha*(r_ij - r_ik)**beta),
            #ddexp_r_ij_r_ij=lambda r_ij, r_ik: alpha*beta*(r_ij - r_ik)**(beta - 2)\
            #                    *(alpha*beta*(r_ij - r_ik)**beta + beta - 1)*np.exp(alpha*(r_ij - r_ik)**beta),
            ddexp_r_ij_r_ik=lambda r_ij, r_ik: -(alpha**2)*np.exp(alpha*(r_ij-r_ik)**beta),
            ddexp_r_ij_r_ij=lambda r_ij, r_ik: (alpha**2)*np.exp(alpha*(r_ij-r_ik)**beta),
            ddg_costheta_costheta=lambda cos_theta: 2*c_2*((c_3 + (cos_theta - h)**2)**2*(2*c_4*c_5**2*(cos_theta - h)**4\
                                        - 5*c_4*c_5*(cos_theta - h)**2 + c_4 + np.exp(c_5*(cos_theta - h)**2))\
                                        + (c_3 + (cos_theta - h)**2)*(cos_theta - h)**2*(4*c_4*c_5*(cos_theta - h)**2\
                                        - 5*c_4 - 5*np.exp(c_5*(cos_theta - h)**2))\
                                        + 4*(c_4 + np.exp(c_5*(cos_theta - h)**2))*(cos_theta - h)**4)*np.exp(-c_5*(cos_theta - h)**2)/(c_3 + (cos_theta - h)**2)**3,
            ddU_r_r=lambda r, f_c, df_c_r, ddf_c_r_r: -B*np.exp(-lambda_2*r)*(ddf_c_r_r-2*df_c_r*lambda_2+lambda_2**2*f_c),
            ddV_r_r=lambda r, f_c, df_c_r, ddf_c_r_r: A*np.exp(-lambda_1*r)*(ddf_c_r_r-2*df_c_r*lambda_1+lambda_1**2*f_c),

        )

        def get_cutoff(self):
            return self.cutoff


"""
Kumagai_CompMaterSci_39_457_Si_py_var = {
    "__ref__":  "T. Kumagai, S. Izumi, S. Hara, and S. Sakai, \
            Comput. Mater. Sci. 39, 457 (2007)",
    "el":        "Si"        ,
    "A":         3281.5905   ,
    "B":         121.00047   ,
    "lambda_1":  3.2300135   ,
    "lambda_2":  1.3457970   ,
    "eta":       1.0000000   ,
    "delta":     0.53298909  ,
    "alpha":     2.3890327   ,
    "beta":      1           ,
    "c_1":       0.20173476  ,
    "c_2":       730418.72   ,
    "c_3":       1000000.0   ,
    "c_4":       1.0000000   ,
    "c_5":       26.000000   ,
    "h":         -0.36500000 ,
    "R_1":       2.70        ,
    "R_2":       3.30        ,
}
"""
