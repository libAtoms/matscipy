# ======================================================================
# matscipy - Python materials science tools
# https://github.com/libAtoms/matscipy
#
# Copyright (2019) Johannes Hoermann, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ======================================================================"""
""" Compute ion concentrations consistent with general
Poisson-Nernst-Planck (PNP) equations.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging, os, sys, time
import numpy as np
import scipy.optimize

from matscipy.electrochemistry.poisson_nernst_planck_solver import PoissonNernstPlanckSystem, B

logger = logging.getLogger(__name__)

class PoissonNernstPlanckSystemLogC(PoissonNernstPlanckSystem):
    """Describes and solves a 1D Poisson-Nernst-Planck system,
    using log concentrations internally"""

    # properties "offer" the solution in physical units:
    @property
    def concentration(self):
        return np.exp(self.Nij)*self.c_unit

    def init(self):
        """Sets up discretization scheme and initial value"""
        # indices
        self.Ni = self.N+1
        I = np.arange(self.Ni)

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
          'discretization segments N', self.N, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
          'grid points N', self.Ni, lwidth=self.label_width))

        # discretization
        self.dx      = self.L_scaled / self.N # spatial step
        # maximum time step allowed
        # (irrelevant for our steady state case)
        # D * dt / dx^2 <= 1/2
        # dt <= dx^2 / ( 2*D )
        # dt_max = dx**2 / 2 # since D = 1
        # dt_max = dx**2 / (2*self.Dn_scaled)

        # dx2overtau = dx**2 / self.tau_scaled
        self.dx2overtau = 10.0

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
          'dx', self.dx, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
          'dx2overtau', self.dx2overtau, lwidth=self.label_width))

        # positions (scaled)
        self.X = self.x0_scaled + I*self.dx

        # Bounary & initial values

        # internally:
        #   n: dimensionless concentrations
        #   u: dimensionless potential
        #   i: spatial index
        #   j: temporal index
        #   k: species index
        # initial concentrations equal to bulk concentrations

        # Kronecker product, M rows (ion species), Ni cols (grid points),
        self.Ni0 = np.kron( np.log(self.c_scaled), np.ones((self.Ni,1)) ).T
        self.zi0 = np.kron( self.z, np.ones((self.Ni,1)) ).T # does not change

        self.initial_values()

    def initial_values(self):
        """
        Solves decoupled linear system to get inital potential distribution.
        """

        zini0 = self.zi0*np.exp(self.Ni0) # z*n

        # shape: ion species (rows), grid points (cols), sum over ion species (along rows)
        rhoi0 = 0.5*zini0.sum(axis=0)

        # system matrix of spatial poisson equation
        Au = np.zeros((self.Ni,self.Ni))
        bu = np.zeros(self.Ni)
        Au[0,0]   = 1
        Au[-1,-1] = 1
        for i in range(1,self.N):
            Au[i,[i-1,i,i+1]] = [1.0, -2.0, 1.0] # 1D Laplace operator, 2nd order

        bu = rhoi0*self.dx**2 # => Poisson equation
        bu[0]  = self.u0
        bu[-1] = self.u1

        # get initial potential distribution by solving Poisson equation
        self.ui0 = np.dot( np.linalg.inv(Au), bu) # A u - b = 0 <=> u = A^-1 b

        return self.ui0

    # evokes Newton solver
    def solve(self):
        """Evokes newton solver

        Returns
        -------
        uij : (Ni,) ndarray
            potential at Ni grid points
        Nij : (M,Nij) ndarray
            log concentrations of M species at Ni grid points
        lamj: (L,) ndarray
            value of L Lagrange multipliers (not implemented, empty)
        """

        if len(self.g) > 0:
            self.xi0 = np.concatenate([self.ui0, self.Ni0.flatten(), np.zeros(len(self.g))])
        else:
            self.xi0 = np.concatenate([self.ui0, self.Ni0.flatten()])

        self.callback_count = 0
        self.t0 = time.perf_counter()
        self.tj = self.t0 # previosu callback timer value

        # neat lecture on scipy optimizers
        # http://scipy-lectures.org/advanced/mathematical_optimization/
        if isinstance(self.solver, str) and self.solver in [
            'hybr','lm','broyden1','broyden2','anderson','linearmixing',
            'diagbroyden','excitingmixing','krylov','df-sane']:
            res = scipy.optimize.root(self.G,self.xi0,
                method=self.solver,callback=self.solver_callback,
                options = self.options)
            self.xij1 = res.x
            if not res.success:
                logger.warn(res.message)
        elif isinstance( self.solver, str):
            f = lambda x: np.linalg.norm(self.G(x))
            res = scipy.optimize.minimize(f,self.xi0.copy(),
                method=self.solver,callback=self.solver_callback,
                options=self.options)
            self.xij1 = res.x
            if not res.success:
                logger.warn(res.message)
        else:
            self.xij1 = self.solver(self.G,self.xi0.copy(),
                callback=self.solver_callback, options=self.options)

        # store results:
        self.uij  = self.xij1[:self.Ni] # potential
        self.Nij  = self.xij1[self.Ni:(self.M+1)*self.Ni].reshape(self.M, self.Ni) # concentrations
        self.lamj = self.xij1[(self.M+1)*self.Ni:] # Lagrange multipliers

        return self.uij, self.Nij, self.lamj

    # boundary conditions and constraints building blocks:
    def leftFiniteDifferenceSchemeFluxBC(self,x,k,j0=0):
        """
        Parameters
        ----------
        x : (Ni,) ndarray
            N-valued variable vector
        k : int
            ion species (-1 for potential)
        j0 : float
            flux of ion species `k` at left hand boundary

        Returns
        -------
        float: boundary condition residual
        """
        uij = x[:self.Ni]
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        # 2nd order right hand side finite difference scheme:
        # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2)
        # - dndx - z n dudx = j0
        dndx = -3.0*np.exp(Nijk[0]) + 4.0*np.exp(Nijk[1]) - np.exp(Nijk[2])
        dudx = -3.0*uij[0]  + 4.0*uij[1]  - uij[2]
        bcval = - dndx - self.zi0[k,0]*np.exp(Nijk[0])*dudx - 2.0*self.dx*j0

        self.logger.debug(
            'Flux BC F[0]  = - dndx - z n dudx - 2*dx*j0 = {:> 8.4g}'.format(bcval))
        self.logger.debug(
            '   = - ({:.2f}) - ({:.0f})*{:.2f}*({:.2f}) - 2*{:.2f}*({:.2f})'.format(
                dndx, self.zi0[k,0], np.exp(Nijk[0]), dudx, self.dx, j0))
        return bcval

    def rightFiniteDifferenceSchemeFluxBC(self,x,k,j0=0):
        """
        See ```leftFiniteDifferenceSchemeFluxBC```
        """
        uij = x[:self.Ni]
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        # 2nd order left hand side finite difference scheme:
        # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2)
        # - dndx - z n dudx = j0
        dndx = 3.0*np.exp(Nijk[-1]) - 4.0*np.exp(Nijk[-2]) + np.exp(Nijk[-3])
        dudx = 3.0*uij[-1]  - 4.0*uij[-2]  + uij[-3]
        bcval = - dndx - self.zi0[k,-1]*np.exp(Nijk[-1])*dudx - 2.0*self.dx*j0

        self.logger.debug(
            'FD flux BC F[-1]  = - dndx - z n dudx - 2*dx*j0 = {:> 8.4g}'.format(bcval))
        self.logger.debug(
            '  = - {:.2f} - {:.0f}*{:.2f}*{:.2f} - 2*{:.2f}*{:.2f}'.format(
                dndx, self.zi0[k,-1], np.exp(Nijk[-1]), dudx, self.dx, j0))
        return bcval

    def leftControlledVolumeSchemeFluxBC(self,x,k,j0=0):
        """
        Compute left hand side flux boundary condition residual in accord with
        controlled volume scheme.

        Parameters
        ----------
        x : (Ni,) ndarray
            N-valued variable vector
        k : int
            ion species (-1 for potential)
        j0 : float
            flux of ion species `k` at left hand boundary

        Returns
        -------
        float: boundary condition residual
        """
        uij = x[:self.Ni]
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]

        # flux by controlled volume scheme:

        bcval = ( + B(self.z[k]*(uij[0]-uij[1]))*np.exp(0.5*(Nijk[1]-Nijk[0]))
                  - B(self.z[k]*(uij[1]-uij[0]))*np.exp(0.5*(Nijk[0]-Nijk[1]))
                  - self.dx*j0 * np.exp(-0.5*(Nijk[0]+Nijk[1])) )

        self.logger.debug(
            'CV flux BC F[0]  = n1*B(z(u0-u1)) - n0*B(z(u1-u0)) - j0*dx = {:> 8.4g}'.format(bcval))
        return bcval

    def rightControlledVolumeSchemeFluxBC(self,x,k,j0=0):
        """
        Compute right hand side flux boundary condition residual in accord with
        controlled volume scheme. See ``leftControlledVolumeSchemeFluxBC``
        """
        uij  = x[:self.Ni]
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]

        # flux by controlled volume scheme:

        bcval = ( + B(self.z[k]*(uij[-2]-uij[-1]))*np.exp(0.5*(Nijk[-1]-Nijk[-2]))
                  - B(self.z[k]*(uij[-1]-uij[-2]))*np.exp(0.5*(Nijk[-2]-Nijk[-1]))
                  - self.dx*j0 * np.exp(-0.5*(Nijk[-1]+Nijk[-2])) )

        self.logger.debug(
            'CV flux BC F[-1]  = n[-1]*B(z(u[-2]-u[-1])) - n[-2]*B(z(u[-1]-u[-2])) - j0*dx = {:> 8.4g}'.format(bcval))
        return bcval

    def leftPotentialDirichletBC(self,x,u0=0):
        """Construct potential Dirichlet BC at left boundary"""
        uij = x[:self.Ni]
        return uij[0] - u0

    def leftDirichletBC(self,x,k,x0=0):
        """Construct Dirichlet BC at left boundary"""
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return Nijk[0] - np.log(x0)

    def rightPotentialDirichletBC(self,x,u0=0):
        """Construct potential Dirichlet BC at left boundary"""
        uij = x[:self.Ni]
        return uij[-1] - u0

    def rightDirichletBC(self,x,k,x0=0):
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return Nijk[-1] - np.log(x0)

    def leftPotentialRobinBC(self,x,lam,u0=0):
        """
        Compute left hand side Robin (u + lam*dudx = u0 ) BC at in accord with
        2nd order finite difference scheme.

        Parameters
        ----------
        x : (Ni,) ndarray
            N-valued variable vector
        lam: float
            BC coefficient, corresponds to Stern layer thickness
            if applied to potential variable in PNP problem. Here, this steric
            layer is assumed to constitute a region of uniform charge density
            and thus linear potential drop across the interface.
        x0 : float
            right hand side value of BC, corresponds to potential beyond Stern
            layer if applied to poential variable in PNP system.

        Returns
        -------
        float: boundary condition residual
        """
        uij = x[:self.Ni]
        return uij[0] + lam/(2*self.dx)* ( 3.0*uij[0] - 4.0*uij[1] + uij[2] ) - u0


    def leftRobinBC(self,x,k,lam,x0=0):
        """
        Compute left hand side Robin (u + lam*dudx = u0 ) BC at in accord with
        2nd order finite difference scheme.

        Parameters
        ----------
        x : (Ni,) ndarray
            N-valued variable vector
        k : int
            ion species (-1 for potential)
        lam: float
            BC coefficient, corresponds to Stern layer thickness
            if applied to potential variable in PNP problem. Here, this steric
            layer is assumed to constitute a region of uniform charge density
            and thus linear potential drop across the interface.
        x0 : float
            right hand side value of BC, corresponds to potential beyond Stern
            layer if applied to poential variable in PNP system.

        Returns
        -------
        float: boundary condition residual
        """
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return Nijk[0] + lam/(2*self.dx)* ( 3.0*np.exp(Nijk[0]) - 4.0*np.exp(Nijk[1]) +np.exp(Nijk[2]) ) - x0

    def rightPotentialRobinBC(self,x,lam,u0=0):
        uij = x[:self.Ni]
        return uij[-1] + lam/(2*self.dx) * ( 3.0*uij[-1] - 4.0*uij[-2] + uij[-3] ) - u0


    def rightRobinBC(self,x,k,lam,x0=0):
        """Construct Robin (u + lam*dudx = u0 ) BC at right boundary."""
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return np.exp(Nijk[-1]) + lam/(2*self.dx) * ( 3.0*np.exp(Nijk[-1]) - 4.0*np.exp(Nijk[-2]) + np.exp(Nijk[-3]) ) - x0

    def numberConservationConstraint(self,x,k,N0):
        """N0: total amount of species, k: ion species"""
        Nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]

        ## TODO: this integration scheme assumes constant concentrations within
        ## an interval. Adapt to controlled volume scheme!

        # rescale to fit interval
        N = np.sum(np.exp(Nijk)) * self.dx * self.N / self.Ni
        constraint_val = N - N0

        self.logger.debug(
            'Number conservation constraint F(x)  = N - N0 = {:.4g} - {:.4g} = {:.4g}'.format(
                N, N0, constraint_val ) )
        return constraint_val

    # standard Poisson equation residual for potential
    def poisson_pde(self,x):
        """Returns Poisson equation resiudal by applying 2nd order FD scheme"""
        uij1 = x[:self.Ni]
        self.logger.debug(
          'potential range [u_min, u_max] = [ {:>.4g}, {:>.4g} ]'.format(
            np.min(uij1),np.max(uij1)))

        Nij1 = x[self.Ni:(self.M+1)*self.Ni]

        Nijk1 = Nij1.reshape( self.M, self.Ni )
        for k in range(self.M):
          self.logger.debug(
            'ion species {:02d} log concentration range [c_min, c_max] = [ {:>.4g}, {:>.4g} ]'.format(
              k,np.min(Nijk1[k,:]),np.max(Nijk1[k,:]) ) )

        # M rows (ion species), N_i cols (grid points)
        zi0nijk1 = self.zi0*np.exp(Nijk1) # z_ik*n_ijk
        for k in range(self.M):
          self.logger.debug(
            'ion species {:02d} charge range [z*c_min, z*c_max] = [ {:>.4g}, {:>.4g} ]'.format(
              k,np.min(zi0nijk1[k,:]),np.max(zi0nijk1[k,:]) ) )

        # charge density sum_k=1^M (z_ik*n_ijk)
        rhoij1 = zi0nijk1.sum(axis=0)
        self.logger.debug(
          'charge density range [rho_min, rho_max] = [ {:>.4g}, {:>.4g} ]'.format(
            np.min(rhoij1),np.max(rhoij1) ) )

        # reduced Poisson equation: d2udx2 = rho
        Fu = - ( np.roll(uij1, -1) - 2*uij1 + np.roll(uij1, 1) ) - 0.5 * rhoij1*self.dx**2

        # linear potential regime due to steric effects incorporated here
        # TODO: incorporate "spatially finite" BC into Robin BC functions
        # replace left and right hand side residuals with linear potential FD

        if not np.isnan(self.lhs_ohp):
            lhs_linear_regime_ndx = (self.X <= self.lhs_ohp)
            lhs_ohp_ndx = np.max( np.nonzero( lhs_linear_regime_ndx ) )

            self.logger.debug(
              'selected {:d} grid points within lhs OHP at grid point index {:d} with x_scaled <= {:>.4g}'.format(
                np.count_nonzero(lhs_linear_regime_ndx), lhs_ohp_ndx, self.lhs_ohp) )

            # dudx = (u[ohp]-u[0])/lambda_S within Stern layer
            Fu[lhs_linear_regime_ndx] = (
                ( np.roll(uij1,-1) - uij1 )[lhs_linear_regime_ndx]
                    * self.lambda_S_scaled - (uij1[lhs_ohp_ndx]-uij1[0])*self.dx )

        if not np.isnan(self.rhs_ohp):
            rhs_linear_regime_ndx = (self.X >= self.rhs_ohp)
            rhs_ohp_ndx = np.min( np.nonzero( rhs_linear_regime_ndx ) )

            self.logger.debug(
              'selected {:d} grid points within lhs OHP at grid point index {:d} with x_scaled >= {:>.4g}'.format(
                 np.count_nonzero(rhs_linear_regime_ndx), rhs_ohp_ndx, self.rhs_ohp) )

            # dudx = (u[ohp]-u[0])/lambda_S within Stern layer
            Fu[rhs_linear_regime_ndx] = (
                ( uij1 - np.roll(uij1,1) )[rhs_linear_regime_ndx]
                    * self.lambda_S_scaled - (uij1[-1]-uij1[rhs_ohp_ndx])*self.dx )

        Fu[0] = self.boundary_conditions[0](x)
        Fu[-1] = self.boundary_conditions[1](x)

        self.logger.debug('Potential BC residual Fu[0]  = {:> 8.4g}'.format(Fu[0]))
        self.logger.debug('Potential BC residual Fu[-1] = {:> 8.4g}'.format(Fu[-1]))

        return Fu

    def nernst_planck_pde(self,x):
        """Returns Nernst-Planck equation resiudal by applying controlled
        volume scheme"""
        uij1 = x[:self.Ni]

        self.logger.debug(
          'potential range [u_min, u_max] = [ {:>.4g}, {:>.4g} ]'.format(
            np.min(uij1),np.max(uij1)))

        Nij1 = x[self.Ni:(self.M+1)*self.Ni]
        Nijk1 = Nij1.reshape( self.M, self.Ni )
        for k in range(self.M):
          self.logger.debug(
            'ion species {:02d} log concentration range [c_min, c_max] = [ {:>.4g}, {:>.4g} ]'.format(
              k,np.min(Nijk1[k,:]),np.max(Nijk1[k,:]) ) )

        Fn = np.zeros([self.M, self.Ni])
        # loop over k = 1..M reduced Nernst-Planck equations:
        # - d2nkdx2 - ddx (zk nk dudx ) = 0
        #nijk1 = np.exp(Nijk1)
        for k in range(self.M):
          # conrolled volume implementation: constant flux across domain
          Fn[k,:] = (
            + B(self.zi0[k,:]*(uij1 - np.roll(uij1,-1))) * np.exp( np.roll(Nijk1[k,:],-1) - Nijk1[k,:] )
            - B(self.zi0[k,:]*(np.roll(uij1,-1) - uij1))
            - B(self.zi0[k,:]*(np.roll(uij1,+1) - uij1))
            + B(self.zi0[k,:]*(uij1 - np.roll(uij1,+1))) * np.exp( np.roll(Nijk1[k,:],+1) - Nijk1[k,:] ) )

          # controlled volume implementation: flux j = 0 in every grid point
          #
          # Fn[k,:] =  (
          #   B(self.zi0[k,:]*(uij1 - np.roll(uij1,-1)))*np.roll(nijk1[k,:],-1)
          #   - B(self.zi0[k,:]*(np.roll(uij1,-1) - uij1))*nijk1[k,:] )
          # linear potential regime due to steric effects incorporated here
          # TODO: incorporate "spatially finite" BC into Robin BC functions
          # replace left and right hand side residuals with linear potential FD

          # left and right hand side outer Helmholtz plane
          # lhs_ohp = self.x0_scaled + self.lambda_S_scaled
          # rhs_ohp = self.x0_scaled + self.L_scaled - self.lambda_S_scaled
          #
          # lhs_linear_regime_ndx = (self.X <= lhs_ohp)
          # rhs_linear_regime_ndx = (self.X >= rhs_ohp)
          #
          # lhs_ohp_ndx = np.max( np.nonzero( lhs_linear_regime_ndx ) )
          # rhs_ohp_ndx = np.min( np.nonzero( rhs_linear_regime_ndx ) )
          #
          # self.logger.debug(
          #   'selected {:d} grid points within lhs OHP at grid point index {:d} with x_scaled <= {:>.4g}'.format(
          #     np.count_nonzero(lhs_linear_regime_ndx), lhs_ohp_ndx, lhs_ohp) )
          # self.logger.debug(
          #   'selected {:d} grid points within lhs OHP at grid point index {:d} with x_scaled >= {:>.4g}'.format(
          #      np.count_nonzero(rhs_linear_regime_ndx), rhs_ohp_ndx, rhs_ohp) )
          #
          # # zero concentration gradient in Stern layer
          # Fn[k,lhs_linear_regime_ndx] = (
          #     ( np.roll(nijk1[k,:],-1)-np.roll(nijk1[k,:],1))[lhs_linear_regime_ndx])
          # Fn[k,rhs_linear_regime_ndx] = (
          #     ( np.roll(nijk1[k,:],-1)-np.roll(nijk1[k,:],1))[rhs_linear_regime_ndx])

          Fn[k,0]  = self.boundary_conditions[2*k+2](x)
          Fn[k,-1] = self.boundary_conditions[2*k+3](x)

          self.logger.debug(
            'ion species {k:02d} BC residual Fn[{k:d},0]  = {:> 8.4g}'.format(
              Fn[k,0],k=k))
          self.logger.debug(
            'ion species {k:02d} BC residual Fn[{k:d},-1]  = {:> 8.4g}'.format(
              Fn[k,-1],k=k))

        return Fn
