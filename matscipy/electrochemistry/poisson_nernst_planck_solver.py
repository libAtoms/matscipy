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
"""
Compute ion concentrations with general Poisson-Nernst-Planck (PNP) equations.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging
import time
import numpy as np
import scipy.constants as sc
import scipy.optimize

logger = logging.getLogger(__name__)

# Druecke nicht-linearen Teil der Transportgleichung (genauer, des Flusses) ueber
# Bernoulli-Funktionen
#
# $$ B(x) = \frac{x}{\exp(x)-1} $$
#
# aus. Damit wir in der Naehe von 0 nicht "in die Bredouille geraten", verwenden
# wir hier lieber die Taylorentwicklung. In der Literatur (Selbherr, S. Analysis
# and Simulation of Semiconductor Devices, Spriger 1984) wird eine noch
# aufwendigere stueckweise Definition empfohlen, allerdings werden wir im
# Folgenden sehen, dass unser Ansatz fuer dieses stationaere Problem genuegt.


def B(x):
    """Bernoulli function."""
    return np.where(
        np.abs(x) < 1e-9,
        1 - x/2 + x**2/12 - x**4/720,  # Taylor
        x / (np.exp(x) - 1))


# "lazy" Ansatz for approximating Jacobian
def jacobian(f, x0, dx=np.NaN):
    """Naive way to construct N x N Jacobin Fij from N-valued function
    f of N-valued vector x0.

    Parameters
    ----------
    f : callable
        N-valued function of N-valued variable vector
    x0 : (N,) ndarray
        N-valued variable vector
    dx : float (default: np.nan)
        Jacobian built with finite difference scheme of spacing ``dx``.
        If ``np.nan``, then use machine precision.

    Returns
    -------
    F : (N,N) ndarray
        NxN-valued 2nd order finite difference scheme approximate of Jacobian
        convention: F_ij = dfidxj, where i are array rows, j are array columns
    """


    N = len(x0)
    # choose step as small as possible
    if np.isnan(dx).any():
        res = np.finfo('float64').resolution
        dx = np.abs(x0) * np.sqrt(res)
        dx[dx < res] = res

    if np.isscalar(dx):
        dx = np.ones(N) * dx

    F = np.zeros((N,N)) # Jacobian Fij

    # convention: dfi_dxj
    # i are rows, j are columns
    for j in range(N):
        dxj = np.zeros(N)
        dxj[j] = dx[j]

        F[:,j] =  (f(x0 + dxj) - f(x0 - dxj)) / (2.0*dxj[j])

    return F

class PoissonNernstPlanckSystem:
    """Describes and solves a 1D Poisson-Nernst-Planck system"""

    # properties "offer" the solution in physical units:
    @property
    def grid(self):
        return self.X*self.l_unit

    @property
    def potential(self):
        return self.uij*self.u_unit

    @property
    def concentration(self):
        return np.where(self.nij > np.finfo('float64').resolution,
            self.nij*self.c_unit, 0.0)

    @property
    def charge_density(self):
        return np.sum(self.F * self.concentration.T * self.z,axis=1)

    @property
    def x1_scaled(self):
        return self.x0_scaled + self.L_scaled

    #TODO:  replace "didactic" Newton solver from IMTEK Simulation course with
    #       some standard package
    def newton(self,f,xij,**kwargs):
        """Newton solver expects system f and initial value xij

        Parameters
        ----------
        f : callable
            N-valued function of N-valued vector
        xij : (N,) ndarray
            N-valued initial value vector

        Returns
        -------
        xij : (N,) ndarray
            N-valued solution vector
        """
        self.xij = []
        self.converged = True
        # assume convergence, set to 'false' if maxit exceeded later

        self.logger.debug('Newton solver, grid points N = {:d}'.format(self.N))
        self.logger.debug('Newton solver, tolerance e = {:> 8.4g}'.format(self.e))
        self.logger.debug('Newton solver, maximum number of iterations M = {:d}'.format(self.maxit))

        i = 0
        delta_rel = 2*self.e

        self.logger.info("Convergence criterion: norm(dx) < {:4.2e}".format(self.e))

        self.convergenceStepAbsolute = np.zeros(self.maxit)
        self.convergenceStepRelative = np.zeros(self.maxit)
        self.convergenceResidualAbsolute = np.zeros(self.maxit)

        dxij = np.zeros(self.N)
        while delta_rel > self.e and i < self.maxit:
            self.logger.debug('*** Newton solver iteration {:d} ***'.format(i))

            # avoid cluttering log
            self.logger.disabled = True
            J = jacobian(f, xij)
            self.logger.disabled = False

            rank = np.linalg.matrix_rank(J)
            self.logger.debug('    Jacobian ({}) rank {:d}'.format(J.shape, rank))

            if rank < self.N:
                self.logger.warn("Singular jacobian of rank"
                      + "{:d} < {:d} at step {:d}".format(
                      rank, self.N, i ))
                break

            F = f(xij)
            invJ = np.linalg.inv(J)

            dxij = np.dot( invJ, F )

            delta = np.linalg.norm(dxij)
            delta_rel = delta / np.linalg.norm(xij)

            xij -= dxij
            self.xij.append(xij)

            normF = np.linalg.norm(F)

            self.logger.debug('    convergence norm(dxij), absolute {:> 8.4g}'.format(delta))
            self.logger.debug('    convergence norm(dxij), realtive {:> 8.4g}'.format(delta_rel))
            self.logger.debug('          residual norm(F), absolute {:> 8.4g}'.format(normF))

            self.convergenceStepAbsolute[i] = delta
            self.convergenceStepRelative[i] = delta_rel
            self.convergenceResidualAbsolute[i] = normF
            self.logger.info("Step {:4d}: norm(dx)/norm(x) = {:4.2e}, norm(dx) = {:4.2e}, norm(F) = {:4.2e}".format(
                i, delta_rel, delta, normF) )

            i += 1

        if i == self.maxit:
            self.logger.warn("Maximum number of iterations reached")
            self.converged = False

        self.logger.info("Ended after {:d} steps.".format(i))

        self.convergenceStepAbsolute = self.convergenceStepAbsolute[:i]
        self.convergenceStepRelative = self.convergenceStepRelative[:i]
        self.convergenceResidualAbsolute = self.convergenceResidualAbsolute[:i]
        return xij

    def solver_callback(self, xij, *_):
        """Callback function that can be used by optimizers of scipy.optimize.
        The second argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. See
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """
        if self.callback_count == 0:
            logger.info(
                "{:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                    "#callback","residual norm","abs dx norm", "rel dx norm",
                    "timing, step", "timing, tot.") )
            self.xij = [ self.xi0 ]

            self.converged = True # TODO remove (?)
            self.convergenceStepAbsolute = []
            self.convergenceStepRelative = []
            self.convergenceResidualAbsolute = []


            dxij = np.zeros(self.N)

        self.xij.append(xij)
        dxij = xij - self.xij[self.callback_count]

        delta = np.linalg.norm(dxij)
        delta_rel = delta / np.linalg.norm(xij)

        fj = self.G(xij)
        norm_fj = np.linalg.norm(fj)

        self.convergenceStepAbsolute.append(delta)
        self.convergenceStepRelative.append(delta_rel)
        self.convergenceResidualAbsolute.append(norm_fj)

        t1 = time.perf_counter()
        dt = t1 - self.tj
        dT = t1 - self.t0
        self.tj = t1

        logger.info(
            "{:12d} {:12.5e} {:12.5e} {:12.5e} {:12.5e} {:12.5e}".format(
                self.callback_count, norm_fj, delta , delta_rel, dt, dT))

        self.callback_count += 1
        return


    def discretize(self):
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

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
          'dx', self.dx, lwidth=self.label_width))

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
        if self.ni0 is None:
            self.ni0 = np.kron( self.c_scaled, np.ones((self.Ni,1)) ).T
        if self.zi0 is None:
            self.zi0 = np.kron( self.z, np.ones((self.Ni,1)) ).T # does not change

        # self.initial_values()

    def initial_values(self):
        """
        Solves decoupled linear system to get inital potential distribution.
        """

        zini0 = self.zi0*self.ni0 # z*n
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
        """Evokes solver

        Returns
        -------
        uij : (Ni,) ndarray
            potential at Ni grid points
        nij : (M,Nij) ndarray
            concentrations of M species at Ni grid points
        lamj: (L,) ndarray
            value of L Lagrange multipliers (not implemented, empty)
        """

        # if not yet done, set up initial values
        if self.ui0 is None:
            self.initial_values()

        if len(self.g) > 0:
            self.xi0 = np.concatenate([self.ui0, self.ni0.flatten(), np.zeros(len(self.g))])
        else:
            self.xi0 = np.concatenate([self.ui0, self.ni0.flatten()])

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
        self.nij  = self.xij1[self.Ni:(self.M+1)*self.Ni].reshape(self.M, self.Ni) # concentrations
        self.lamj = self.xij1[(self.M+1)*self.Ni:] # Lagrange multipliers

        return self.uij, self.nij, self.lamj

    # standard sets of boundary conditions:
    def useStandardInterfaceBC(self):
        """Interface at left hand side and open bulk at right hand side"""
        self.boundary_conditions = []

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled
        self.u1 = 0
        self.logger.info('Left hand side Dirichlet boundary condition:                               u0 = {:> 8.4g}'.format(self.u0))
        self.logger.info('Right hand side Dirichlet boundary condition:                              u1 = {:> 8.4g}'.format(self.u1))

        self.boundary_conditions.extend([
            lambda x: self.leftPotentialDirichletBC(x,self.u0),
            lambda x: self.rightPotentialDirichletBC(x,self.u1) ])
        # self.rightPotentialBC = lambda x: self.rightPotentialDirichletBC(x,self.u1)

        #self.rightConcentrationBC = []
        for k in range(self.M):
          self.logger.info('Ion species {:02d} left hand side concentration Flux boundary condition:       j0 = {:> 8.4g}'.format(k,0))
          self.logger.info('Ion species {:02d} right hand side concentration Dirichlet boundary condition: c1 = {:> 8.4g}'.format(k,self.c_scaled[k]))
          self.boundary_conditions.extend( [
            lambda x, k=k: self.leftControlledVolumeSchemeFluxBC(x,k),
            lambda x, k=k: self.rightDirichletBC(x,k,self.c_scaled[k]) ] )
          #self.rightConcentrationBC.append(
          #  lambda x, k=k: self.rightDirichletBC(x,k,self.c_scaled[k]) )
        # counter-intuitive behavior of lambda in loop:
        # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
        # workaround: default parameter k=k

    def useStandardCellBC(self):
        """Interfaces at left hand side and right hand side"""
        self.boundary_conditions = []

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.0
        self.u1 = - self.delta_u_scaled / 2.
        self.logger.info('{:>{lwidth}s} u0 = {:< 8.4g}'.format(
          'Left hand side Dirichlet boundary condition', self.u0, lwidth=self.label_width))
        self.logger.info('{:>{lwidth}s} u1 = {:< 8.4g}'.format(
          'Right hand side Dirichlet boundary condition', self.u1, lwidth=self.label_width))
        self.boundary_conditions.extend([
          lambda x: self.leftPotentialDirichletBC(x,self.u0),
          lambda x: self.rightPotentialDirichletBC(x,self.u1) ])

        N0 = self.L_scaled*self.c_scaled # total amount of species in cell
        for k in range(self.M):
          self.logger.info('{:>{lwidth}s} j0 = {:<8.4g}'.format(
            'Ion species {:02d} left hand side concentration Flux boundary condition'.format(k),
            0.0, lwidth=self.label_width))
          self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
            'Ion species {:02d} number conservation constraint'.format(k),
            N0[k], lwidth=self.label_width))

          self.boundary_conditions.extend(  [
            lambda x, k=k: self.leftControlledVolumeSchemeFluxBC(x,k),
            lambda x, k=k, N0=N0[k]: self.numberConservationConstraint(x,k,N0) ] )

    def useSternLayerCellBC(self, implicit=False):
        """Interfaces at left hand side and right hand side,
        Stern layer either by prescribing linear potential regime between cell
        boundary and outer Helmholtz plane (OHP), or by applying Robin BC;
        zero flux BC on all ion species.

        Parameters
        ----------
        implicit : bool, optional
            If true, then true Robin BC are applied. Attention:
            if desired, domain must be cropped by manually by twice the Stern
            layer thickness lambda_S. Otherwise, enforces
            constant potential gradient across Stern layer region of thickness
            lambda_S. (default:False)
        """
        self.boundary_conditions = []

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.0
        self.u1 = - self.delta_u_scaled / 2.0

        if implicit: # implicitly treat Stern layer via Robin BC
            self.logger.info('Implicitly treating Stern layer via Robin BC')

            self.logger.info('{:>{lwidth}s} u0 + lambda_S*dudx = {:< 8.4g}'.format(
              'Left hand side Robin boundary condition', self.u0, lwidth=self.label_width))
            self.logger.info('{:>{lwidth}s} u1 + lambda_S*dudx = {:< 8.4g}'.format(
              'Right hand side Robin boundary condition', self.u1, lwidth=self.label_width))
            self.boundary_conditions.extend([
              lambda x: self.leftPotentialRobinBC(x,self.lambda_S_scaled,self.u0),
              lambda x: self.rightPotentialRobinBC(x,self.lambda_S_scaled,self.u1) ])

        else: # explicitly treat Stern layer via linear regime
            self.logger.info('Explicitly treating Stern layer as uniformly charged regions')

            # set left and right hand side outer Helmholtz plane
            self.lhs_ohp = self.x0_scaled + self.lambda_S_scaled
            self.rhs_ohp = self.x0_scaled + self.L_scaled - self.lambda_S_scaled

            self.logger.info('{:>{lwidth}s} u0 = {:< 8.4g}'.format(
              'Left hand side Dirichlet boundary condition', self.u0, lwidth=self.label_width))
            self.logger.info('{:>{lwidth}s} u1 = {:< 8.4g}'.format(
              'Right hand side Dirichlet boundary condition', self.u1, lwidth=self.label_width))
            self.boundary_conditions.extend([
              lambda x: self.leftPotentialDirichletBC(x,self.u0),
              lambda x: self.rightPotentialDirichletBC(x,self.u1) ])


        N0 = self.L_scaled*self.c_scaled # total amount of species in cell
        for k in range(self.M):
          self.logger.info('{:>{lwidth}s} j0 = {:<8.4g}'.format(
            'Ion species {:02d} left hand side concentration Flux boundary condition'.format(k),
            0.0, lwidth=self.label_width))
          self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
            'Ion species {:02d} number conservation constraint'.format(k),
            N0[k], lwidth=self.label_width))

          self.boundary_conditions.extend(  [
            lambda x, k=k: self.leftControlledVolumeSchemeFluxBC(x,k),
            lambda x, k=k, N0=N0[k]: self.numberConservationConstraint(x,k,N0) ] )

    # TODO: meaningful test for Dirichlet BC
    def useStandardDirichletBC(self):
        """Dirichlet BC for all variables at all boundaries"""
        self.boundary_conditions = []

        self.u0 = self.delta_u_scaled
        self.u1 = 0

        self.logger.info('Left hand side potential Dirichlet boundary condition:                     u0 = {:> 8.4g}'.format(self.u0))
        self.logger.info('Right hand side potential Dirichlet boundary condition:                    u1 = {:> 8.4g}'.format(self.u1))

        # set up boundary conditions
        self.boundary_conditions.extend( [
          lambda x: self.leftPotentialDirichletBC(x,self.u0),
          lambda x: self.rightPotentialDirichletBC(x,self.u1) ] )

        for k in range(self.M):
          self.logger.info('Ion species {:02d} left hand side concentration Dirichlet boundary condition:  c0 = {:> 8.4g}'.format(k,self.c_scaled[k]))
          self.logger.info('Ion species {:02d} right hand side concentration Dirichlet boundary condition: c1 = {:> 8.4g}'.format(k,self.c_scaled[k]))
          self.boundary_conditions.extend( [
            lambda x, k=k: self.leftDirichletBC(x,k,self.c_scaled[k]),
            lambda x, k=k: self.rightDirichletBC(x,k,self.c_scaled[k]) ] )

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
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        # 2nd order right hand side finite difference scheme:
        # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2)
        # - dndx - z n dudx = j0
        dndx = -3.0*nijk[0] + 4.0*nijk[1] - nijk[2]
        dudx = -3.0*uij[0]  + 4.0*uij[1]  - uij[2]
        bcval = - dndx - self.zi0[k,0]*nijk[0]*dudx - 2.0*self.dx*j0

        self.logger.debug(
            'Flux BC F[0]  = - dndx - z n dudx - 2*dx*j0 = {:> 8.4g}'.format(bcval))
        self.logger.debug(
            '   = - ({:.2f}) - ({:.0f})*{:.2f}*({:.2f}) - 2*{:.2f}*({:.2f})'.format(
                dndx, self.zi0[k,0], nijk[0], dudx, self.dx, j0))
        return bcval

    def rightFiniteDifferenceSchemeFluxBC(self,x,k,j0=0):
        """
        See ```leftFiniteDifferenceSchemeFluxBC```
        """
        uij = x[:self.Ni]
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        # 2nd order left hand side finite difference scheme:
        # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2)
        # - dndx - z n dudx = j0
        dndx = 3.0*nijk[-1] - 4.0*nijk[-2] + nijk[-3]
        dudx = 3.0*uij[-1]  - 4.0*uij[-2]  + uij[-3]
        bcval = - dndx - self.zi0[k,-1]*nijk[-1]*dudx - 2.0*self.dx*j0

        self.logger.debug(
            'FD flux BC F[-1]  = - dndx - z n dudx - 2*dx*j0 = {:> 8.4g}'.format(bcval))
        self.logger.debug(
            '  = - {:.2f} - {:.0f}*{:.2f}*{:.2f} - 2*{:.2f}*{:.2f}'.format(
                dndx, self.zi0[k,-1], nijk[-1], dudx, self.dx, j0))
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
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]

        # flux by controlled volume scheme:

        bcval = ( + B(self.z[k]*(uij[0]-uij[1]))*nijk[1]
                  - B(self.z[k]*(uij[1]-uij[0]))*nijk[0] - self.dx*j0 )

        self.logger.debug(
            'CV flux BC F[0]  = n1*B(z(u0-u1)) - n0*B(z(u1-u0)) - j0*dx = {:> 8.4g}'.format(bcval))
        return bcval

    def rightControlledVolumeSchemeFluxBC(self,x,k,j0=0):
        """
        Compute right hand side flux boundary condition residual in accord with
        controlled volume scheme. See ``leftControlledVolumeSchemeFluxBC``
        """
        uij = x[:self.Ni]
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]

        # flux by controlled volume scheme:

        bcval = ( + B(self.z[k]*(uij[-2]-uij[-1]))*nijk[-1]
                  - B(self.z[k]*(uij[-1]-uij[-2]))*nijk[-2] - self.dx*j0 )

        self.logger.debug(
            'CV flux BC F[-1]  = n[-1]*B(z(u[-2]-u[-1])) - n[-2]*B(z(u[-1]-u[-2])) - j0*dx = {:> 8.4g}'.format(bcval))
        return bcval

    def leftPotentialDirichletBC(self,x,u0=0):
        return self.leftDirichletBC(x,-1,u0)

    def leftDirichletBC(self,x,k,x0=0):
        """Construct Dirichlet BC at left boundary"""
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return nijk[0] - x0

    def rightPotentialDirichletBC(self,x,x0=0):
        return self.rightDirichletBC(x,-1,x0)

    def rightDirichletBC(self,x,k,x0=0):
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return nijk[-1] - x0

    def leftPotentialRobinBC(self,x,lam,u0=0):
        return self.leftRobinBC(x,-1,lam,u0)

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
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return nijk[0] + lam/(2*self.dx)* ( 3.0*nijk[0] - 4.0*nijk[1] + nijk[2] ) - x0

    def rightPotentialRobinBC(self,x,lam,u0=0):
        return self.rightRobinBC(x,-1,lam,u0)

    def rightRobinBC(self,x,k,lam,x0=0):
        """Construct Robin (u + lam*dudx = u0 ) BC at right boundary."""
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
        return nijk[-1] + lam/(2*self.dx) * ( 3.0*nijk[-1] - 4.0*nijk[-2] + nijk[-3] ) - x0

    def numberConservationConstraint(self,x,k,N0):
        """N0: total amount of species, k: ion species"""
        nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]

        ## TODO: this integration scheme assumes constant concentrations within
        ## an interval. Adapt to controlled volume scheme!

        # rescale to fit interval
        N = np.sum(nijk*self.dx) * self.N / self.Ni
        constraint_val = N - N0

        self.logger.debug(
            'Number conservation constraint F(x)  = N - N0 = {:.4g} - {:.4g} = {:.4g}'.format(
                N, N0, constraint_val ) )
        return constraint_val

    # TODO: remove or standardize
    # def leftNeumannBC(self,x,j0):
    #   """Construct finite difference Neumann BC (flux BC) at left boundary"""
    #   # right hand side first derivative of second order error
    #   # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2) = j0
    #   bcval = -3.0*x[0] + 4.0*x[1] - x[2] - 2.0*self.dx*j0
    #   self.logger.debug(
    #     'Neumann BC F[0]  = -3*x[0]  + 4*x[1]  - x[2]  = {:> 8.4g}'.format(bcval))
    #   return bcval
    #
    # def rightNeumannBC(self,x,j0):
    #   """Construct finite difference Neumann BC (flux BC) at right boundray"""
    #   # left hand side first derivative of second order error
    #   # dfndx = 1 / (2*dx) * (+3 fn - 4 fn-1 + fn-2 ) + O(dx^2) = 0
    #   bcval = 3.0*x[-1] - 4.0*x[-2] + x[-3] - 2.0*self.dx*j0
    #   self.logger.debug(
    #     'Neumann BC F[-1] = -3*x[-1] + 4*x[-2] - nijk[-3] = {:> 8.4g}'.format(bcval))
    #   return bcval

    # standard Poisson equation residual for potential
    def poisson_pde(self,x):
        """Returns Poisson equation resiudal by applying 2nd order FD scheme"""
        uij1 = x[:self.Ni]
        self.logger.debug(
          'potential range [u_min, u_max] = [ {:>.4g}, {:>.4g} ]'.format(
            np.min(uij1),np.max(uij1)))

        nij1 = x[self.Ni:(self.M+1)*self.Ni]

        nijk1 = nij1.reshape( self.M, self.Ni )
        for k in range(self.M):
          self.logger.debug(
            'ion species {:02d} concentration range [c_min, c_max] = [ {:>.4g}, {:>.4g} ]'.format(
              k,np.min(nijk1[k,:]),np.max(nijk1[k,:])))

        # M rows (ion species), N_i cols (grid points)
        zi0nijk1 = self.zi0*nijk1 # z_ik*n_ijk
        for k in range(self.M):
          self.logger.debug(
            'ion species {:02d} charge range [z*c_min, z*c_max] = [ {:>.4g}, {:>.4g} ]'.format(
              k,np.min(zi0nijk1[k,:]), np.max(zi0nijk1[k,:])))

        # charge density sum_k=1^M (z_ik*n_ijk)
        rhoij1 = zi0nijk1.sum(axis=0)
        self.logger.debug(
          'charge density range [rho_min, rho_max] = [ {:>.4g}, {:>.4g} ]'.format(
            np.min(rhoij1),np.max(rhoij1)))

        # reduced Poisson equation: d2udx2 = rho
        Fu = -(np.roll(uij1, -1)-2*uij1+np.roll(uij1, 1))-0.5*rhoij1*self.dx**2

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

        nij1 = x[self.Ni:(self.M+1)*self.Ni]
        nijk1 = nij1.reshape( self.M, self.Ni )
        for k in range(self.M):
          self.logger.debug(
            'ion species {:02d} concentration range [c_min, c_max] = [ {:>.4g}, {:>.4g} ]'.format(
              k,np.min(nijk1[k,:]),np.max(nijk1[k,:]) ) )

        Fn = np.zeros([self.M, self.Ni])
        # loop over k = 1..M reduced Nernst-Planck equations:
        # - d2nkdx2 - ddx (zk nk dudx ) = 0
        for k in range(self.M):
          # conrolled volume implementation: constant flux across domain
          Fn[k,:] = (
            + B(self.zi0[k,:]*(uij1 - np.roll(uij1,-1))) * np.roll(nijk1[k,:],-1)
            - B(self.zi0[k,:]*(np.roll(uij1,-1) - uij1)) * nijk1[k,:]
            - B(self.zi0[k,:]*(np.roll(uij1,+1) - uij1)) * nijk1[k,:]
            + B(self.zi0[k,:]*(uij1 - np.roll(uij1,+1))) * np.roll(nijk1[k,:],+1) )

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

    # non-linear system, "controlled volume" method
    # Selbherr, S. Analysis and Simulation of Semiconductor Devices, Spriger 1984
    def G(self, x):
        """Non-linear system

        Discretization of Poisson-Nernst-Planck system with M ion species.
        Implements "controlled volume" method as found in

          Selbherr, Analysis and Simulation of Semiconductor Devices, Spriger 1984

        Parameters
        ----------
        x : ((M+1)*Ni,) ndarray
            system variables. 1D array of (M+1)*Ni values, wher M is number of
            ion sepcies, Ni number of spatial discretization points. First Ni
            entries are expected to contain potential, following M*Ni points
            contain ion concentrations.
        Returns
        --------
        residual: ((M+1)*Ni,) ndarray
        """
        # reduced Poisson equation: d2udx2 = rho
        Fu = self.potential_pde(x)
        Fn = self.concentration_pde(x)

        # Apply constraints if set (not implemented properly, do not use):
        if len(self.g) > 0:
            Flam = np.array([g(x) for g in self.g])
            F = np.concatenate([Fu,Fn.flatten(),Flam])
        else:
            F = np.concatenate([Fu,Fn.flatten()])

        return F

    @property
    def I(self): # ionic strength
        """Compute the system's ionic strength from charges and concentrations.

        Returns
        -------
        I : float
            ionic strength ( 1/2 * sum(z_i^2*c_i) )
            [concentration unit, i.e. mol m^-3]
        """
        return 0.5*np.sum( np.square(self.z) * self.c )

    @property
    def lambda_D(self):
        """Compute the system's Debye length.

        Returns
        -------
        lambda_D : float
            Debye length, sqrt( epsR*eps*R*T/(2*F^2*I) ) [length unit, i.e. m]
        """
        return np.sqrt(
            self.relative_permittivity*self.vacuum_permittivity*self.R*self.T/(
                2.0*self.F**2*self.I ) )

    # default 0.1 mM (i.e. mol/m^3) NaCl aqueous solution
    def init(self,
        c = np.array([0.1,0.1]),
        z = np.array([1,-1]),
        L = 100e-9, # 100 nm
        lambda_S=0, # Stern layer (compact layer) thickness
        x0 = 0, # zero position
        T = 298.15,
        delta_u = 0.05, # potential difference [V]
        relative_permittivity = 79,
        vacuum_permittivity   = sc.epsilon_0,
        R = sc.value('molar gas constant'),
        F = sc.value('Faraday constant'),
        N = 200, # number of grid segments, number of grid points Ni = N + 1
        e = 1e-10, # absolute tolerance, TODO: switch to standaradized measure
        maxit = 20, # maximum number of Newton iterations
        solver = None,
        options = None,
        potential0 = None,
        concentration0 = None ):
        """Initializes a 1D Poisson-Nernst-Planck system description.

        Expects quantities in SI units per default.

        Parameters
        ----------
        c : (M,) ndarray, optional
            bulk concentrations of each ionic species [mol/m^3]
            (default: [ 0.1, 0.1 ])
        z : (M,) ndarray, optional
            charge of each ionic species [1] (default: [ +1, -1 ])
        x0 : float, optional
            left hand side reference position (default: 0)
        L : float, optional
            1D domain size [m] (default: 100e-9)
        lambda_S: float, optional
            Stern layer thickness in case of Robin BC [m] (default: 0)
        T : float, optional
            temperature of the solution [K] (default: 298.15)
        delta_u : float, optional
            potential drop across 1D cell [V] (default: 0.05)
        relative_permittivity: float, optional
            relative permittivity of the ionic solution [1] (default: 79)
        vacuum_permittivity: float, optional
            vacuum permittivity [F m^-1] (default: 8.854187817620389e-12 )
        R : float, optional
            molar gas constant [J mol^-1 K^-1] (default: 8.3144598)
        F : float, optional
            Faraday constant [C mol^-1] (default: 96485.33289)
        N : int, optional
            number of discretization grid segments (default: 200)
        e : float, optional
            absolute tolerance for Newton solver convergence (default: 1e-10)
        maxit : int, optional
            maximum number of Newton iterations (default: 20)
        solver: func( funx(x), x0), optional
            solver to use (default: None, will use own simple Newton solver)
        potential0: (N+1,) ndarray, optional (default: None)
            potential initial values
        concentration0: (M,N+1) ndarray, optional (default: None)
            concentration initial values
        """

        self.logger = logging.getLogger(__name__)

        assert len(c) == len(z), "Provide concentration AND charge for ALL ion species!"

        # TODO: integrate with constructor initialization parameters above
        # default solver settings
        self.converged = False # solver's convergence flag
        self.N      = N     # discretization segments
        self.e      = e     # Newton solver default tolerance
        self.maxit  = maxit # Newton solver maximum iterations

        # default output settings
        # self.output = False   # let Newton solver output convergence plots...
        # self.outfreq = 1      # ...at every nth iteration
        self.label_width = 40 # charcater width of quantity labels in log

        # standard governing equations
        self.potential_pde      = self.poisson_pde
        self.concentration_pde  = self.nernst_planck_pde

        # empty BC
        self.boundary_conditions = []
        # empty constraints
        self.g = [] # list of constrain functions, not fully implemented / tested

        # system parameters
        self.M = len(c) # number of ion species

        self.c  = c # concentrations
        self.z  = z # number charges
        self.T  = T # temperature
        self.L  = L # 1d domain size
        self.lambda_S = lambda_S # Stern layer thickness
        self.x0 = x0 # reference position
        self.delta_u = delta_u # potential difference


        self.relative_permittivity = relative_permittivity
        self.vacuum_permittivity   = vacuum_permittivity
        # R = N_A * k_B
        # (universal gas constant = Avogadro constant * Boltzmann constant)
        self.R                     = R
        self.F                     = F

        self.f                     = F / (R*T)  # for convenience

        # print all quantities to log
        for i, (c, z) in enumerate(zip(self.c, self.z)):
            self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
                "ion species {:02d} concentration c".format(i), c, lwidth=self.label_width))
            self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
                "ion species {:02d} number charge z".format(i), z, lwidth=self.label_width))

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'temperature T', self.T, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'domain size L', self.L, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'compact layer thickness lambda_S', self.lambda_S, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reference position x0', self.x0, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'potential difference delta_u', self.delta_u, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'relative permittivity eps_R', self.relative_permittivity, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'vacuum permittivity eps_0', self.vacuum_permittivity, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'universal gas constant R', self.R, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'Faraday constant F', self.F, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'f = F / (RT)', self.f, lwidth=self.label_width))

        # scaled units for dimensionless formulation

        # length unit chosen as Debye length lambda
        self.l_unit = self.lambda_D

        # concentration unit is ionic strength
        self.c_unit = self.I

        # no time unit for now, only steady state
        # self.t_unit = self.l_unit**2 / self.Dn # fixes Dn_scaled = 1

        self.u_unit = self.R * self.T / self.F  # thermal voltage

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'spatial unit [l]', self.l_unit, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'concentration unit [c]', self.c_unit, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'potential unit [u]', self.u_unit, lwidth=self.label_width))

        # domain
        self.L_scaled = self.L / self.l_unit

        # compact layer
        self.lambda_S_scaled = self.lambda_S / self.l_unit

        # reference position
        self.x0_scaled = self.x0 / self.l_unit

        # bulk conectrations
        self.c_scaled = self.c / self.c_unit

        # potential difference
        self.delta_u_scaled = self.delta_u / self.u_unit

        # print scaled quantities to log
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced domain size L*', self.L_scaled, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced compact layer thickness lambda_S*', self.lambda_S_scaled, lwidth=self.label_width))
        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced reference position x0*', self.x0_scaled, lwidth=self.label_width))

        for i, c_scaled in enumerate(self.c_scaled):
            self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
                "ion species {:02d} reduced concentration c*".format(i),
                c_scaled, lwidth=self.label_width))

        self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
            'reduced potential delta_u*', self.delta_u_scaled, lwidth=self.label_width))

        # per default, no outer Helmholtz plane
        self.lhs_ohp = np.nan
        self.rhs_ohp = np.nan

        # self.xi0 = None
        # initialize initial value arrays
        if potential0 is not None:
            self.ui0 = potential0 / self.u_unit
        else:
            self.ui0 = None

        if concentration0 is not None:
            self.ni0 = concentration0 / self.c_unit
        else:
            self.ni0 = None

        self.zi0 = None

    def __init__(self, *args, **kwargs):
        """Constructor, see init doc string for arguments.

        Additional Parameters
        ---------------------
        solver: str or func (default: None)
            solver to use. If str, then selected from scipy optimizers.
        options: dict, optional (default: None)
            options object for scipy solver
        """
        self.init(*args, **kwargs)

        if 'solver' in kwargs:
            self.solver = kwargs['solver']
        else:
            self.solver = self.newton

        if 'options' in kwargs:
            self.options = kwargs['options']
        else:
            self.options = None

        self.discretize()
