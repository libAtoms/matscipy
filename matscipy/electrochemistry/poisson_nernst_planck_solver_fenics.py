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
Poisson-Nernst-Planck (PNP) equations via FEniCS.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging, os, sys, time
import numpy as np
import scipy.interpolate
import fenics as fn

from matscipy.electrochemistry.poisson_nernst_planck_solver import PoissonNernstPlanckSystem, B

logger = logging.getLogger(__name__)

class PoissonNernstPlanckSystemFEniCS(PoissonNernstPlanckSystem):
    """Describes and solves a 1D Poisson-Nernst-Planck system,
    using log concentrations internally"""

    @property
    def X(self):
        return self.mesh.coordinates().flatten() # only 1D

    def solve(self):
        """Evokes FEniCS FEM solver

        Returns
        -------
        uij : (Ni,) ndarray
            potential at Ni grid points
        nij : (M,Nij) ndarray
            concentrations of M species at Ni grid points
        lamj: (L,) ndarray
            value of L Lagrange multipliers
        """

        # TODO: implement Neumann and Robin BC:
        rho = 0
        for i in range(self.M):
            rho += self.z[i]*self.p[i]

        source = - 0.5 * rho * self.v * fn.dx

        laplace = fn.dot(fn.grad(self.u), fn.grad(self.v))*fn.dx

        poisson = laplace + source

        nernst_planck = 0
        for i in range(self.M):
            nernst_planck += fn.dot(
                    - fn.grad(self.p[i]) - self.z[i]*self.p[i]*fn.grad(self.u),
                    fn.grad(self.q[i])
                )*fn.dx

        # constraints set up elsewhere
        F = poisson + nernst_planck + self.constraints
        # for i in range self.M:

        fn.solve(F==0,self.w,self.boundary_conditions)

        # store results:

        wij = np.array([ self.w(x) for x in self.X ]).T

        self.uij  = wij[0,:]            # potential
        self.nij  = wij[1:(self.M+1),:] # concentrations
        self.lamj = wij[(self.M+1):]    # Lagrange multipliers

        return self.uij, self.nij, self.lamj

    def boundary_L(self, x, on_boundary):
        return on_boundary and fn.near(x[0], self.x0_scaled, self.bctol)

    def boundary_R(self, x, on_boundary):
        return on_boundary and fn.near(x[0], self.x1_scaled, self.bctol)

    def boundary_inner(self, x, on_boundary, x_ref):
        return fn.near(x[0], x_ref, self.bctol)

    def applyLeftPotentialDirichletBC(self,u0):
        self.boundary_conditions.extend( [
            fn.DirichletBC(self.W.sub(0),u0,
                lambda x, on_boundary: self.boundary_L(x, on_boundary)) ] )

    def applyRightPotentialDirichletBC(self,u0):
        self.boundary_conditions.extend( [
            fn.DirichletBC(self.W.sub(0),u0,
                lambda x, on_boundary: self.boundary_R(x, on_boundary)) ] )

    def applyLeftConcentrationDirichletBC(self,k,c0):
        """FEniCS Dirichlet BC c0 for k'th ion species at left boundary"""
        self.boundary_conditions.extend( [
          fn.DirichletBC(self.W.sub(k+1),c0,
              lambda x, on_boundary: self.boundary_L(x, on_boundary)) ] )

    def applyRightConcentrationDirichletBC(self,k,c0):
        """FEniCS Dirichlet BC c0 for k'th ion species at right boundary"""
        self.boundary_conditions.extend( [
          fn.DirichletBC(self.W.sub(k+1),c0,
              lambda x, on_boundary: self.boundary_R(x, on_boundary)) ] )

    def applyReferenceConcentrationConstraint(self,k,c0,x0):
        """FEniCS Dirichlet BC c0 for k'th ion species at right boundary"""
        self.boundary_conditions.extend( [
          fn.DirichletBC(self.W.sub(k+1),c0,
              lambda x, on_boundary, x0=x0: self.boundary_inner(x, on_boundary, x0)) ] )


    def applyPotentialDirichletBC(self,u0,u1):
        self.applyLeftPotentialDirichletBC(u0)
        self.applyRightPotentialDirichletBC(u1)

    def applyNumberConservationConstraint(self,k,c0):
        """
        Enforce number conservation constraint via Lagrange multiplier.
        See https://fenicsproject.org/docs/dolfin/1.6.0/python/demo/documented/neumann-poisson/python/documentation.html
        """
        self.constraints += self.lam[k]*self.q[k]*fn.dx \
            + (self.p[k]-c0)*self.mu[k]*fn.dx
             # + (self.p[k] - c0)*self.mu[k]*fn.dx

    def useStandardInterfaceBC(self):
        """Interface at left hand side and open bulk at right hand side"""
        self.boundary_conditions = []

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled
        self.u1 = 0

        self.logger.info('Left hand side Dirichlet boundary condition:                               u0 = {:> 8.4g}'.format(self.u0))
        self.logger.info('Right hand side Dirichlet boundary condition:                              u1 = {:> 8.4g}'.format(self.u1))

        self.applyPotentialDirichletBC(self.u0,self.u1)

        for k in range(self.M):
            self.logger.info('Ion species {:02d} right hand side concentration Dirichlet boundary condition: c1 = {:> 8.4g}'.format(k,self.c_scaled[k]))
            self.applyRightConcentrationDirichletBC(k,self.c_scaled[k])

    def useStandardCellBC(self):
        """
        Interfaces at left hand side and right hand side, species-wise
        number conservation within interval."""
        self.boundary_conditions = []

        # Introduce a Lagrange multiplier per species anderson
        # rebuild discretization scheme (function spaces)
        self.K = self.M
        self.discretize()

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.
        self.u1 = - self.delta_u_scaled / 2.
        self.logger.info('{:>{lwidth}s} u0 = {:< 8.4g}'.format(
          'Left hand side Dirichlet boundary condition', self.u0, lwidth=self.label_width))
        self.logger.info('{:>{lwidth}s} u1 = {:< 8.4g}'.format(
          'Right hand side Dirichlet boundary condition', self.u1, lwidth=self.label_width))

        self.applyPotentialDirichletBC(self.u0,self.u1)

        # Number conservation constraints
        self.constraints = 0
        N0 = self.L_scaled*self.c_scaled # total amount of species in cell
        for k in range(self.M):
            self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
                'Ion species {:02d} number conservation constraint'.format(k),
                N0[k], lwidth=self.label_width))
            self.applyNumberConservationConstraint(k,self.c_scaled[k])

    def useReferenceConcentrationBasedCellBC(self):
        """
        Interfaces at left hand side and right hand side, species-wise
        concentration fixed at cell center."""
        self.boundary_conditions = []

        # Introduce a Lagrange multiplier per species anderson
        # rebuild discretization scheme (function spaces)
        self.K = self.M
        self.discretize()

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.
        self.u1 = - self.delta_u_scaled / 2.
        self.logger.info('{:>{lwidth}s} u0 = {:< 8.4g}'.format(
          'Left hand side Dirichlet boundary condition', self.u0, lwidth=self.label_width))
        self.logger.info('{:>{lwidth}s} u1 = {:< 8.4g}'.format(
          'Right hand side Dirichlet boundary condition', self.u1, lwidth=self.label_width))

        self.applyPotentialDirichletBC(self.u0,self.u1)

        # Number conservation constraints
        x_ref = (self.x0_scaled + self.x1_scaled)/2.
        for k in range(self.M):
            # self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
            #    'Ion species {:02d} number conservation constraint'.format(k),
            #    N0[k], lwidth=self.label_width))
            # self.applyNumberConservationConstraint(k,self.c_scaled[k])
            self.logger.info(
                'Ion species {:02d} reference concentration condition: c1 = {:> 8.4g} at x0 = {:> 8.4}'.format(
                    k,self.c_scaled[k],x_ref))

            self.applyReferenceConcentrationConstraint(k,self.c_scaled[k],x_ref)

    def discretize(self):
        """Builds function space, call again after introducing constraints"""
        # FEniCS interface
        self.mesh = fn.IntervalMesh(self.N,self.x0_scaled,self.x1_scaled)

        # http://www.femtable.org/
        # Argyris*                          ARG
        # Arnold-Winther*                   AW
        # Brezzi-Douglas-Fortin-Marini*     BDFM
        # Brezzi-Douglas-Marini             BDM
        # Bubble                            B
        # Crouzeix-Raviart                  CR
        # Discontinuous Lagrange            DG
        # Discontinuous Raviart-Thomas      DRT
        # Hermite*                          HER
        # Lagrange                          CG
        # Mardal-Tai-Winther*               MTW
        # Morley*                           MOR
        # Nedelec 1st kind H(curl)          N1curl
        # Nedelec 2nd kind H(curl)          N2curl
        # Quadrature                        Q
        # Raviart-Thomas                    RT
        # Real                              R

        # construct test function space
        P = fn.FiniteElement('Lagrange',fn.interval,3)
        R  = fn.FiniteElement('Real', fn.interval,0)
        elements = [P]*(1+self.M) + [R]*self.K

        H = fn.MixedElement(elements)
        self.W = fn.FunctionSpace(self.mesh,H)

        # solution functions
        self.w = fn.Function(self.W)

        # set initial values if available
        P = fn.FunctionSpace(self.mesh, 'P', 1)
        dof2vtx = fn.vertex_to_dof_map(P)
        if self.ui0 is not None:
            x = np.linspace(self.x0_scaled, self.x1_scaled, self.ui0.shape[0])
            ui0 = scipy.interpolate.interp1d(x,self.ui0)
            # use linear interpolation on mesh
            self.u0_func = fn.Function(P)
            self.u0_func.vector()[:] = ui0(self.X)[dof2vtx]
            fn.assign( self.w.sub(0),
                fn.interpolate( self.u0_func, self.W.sub(0).collapse() ) )

        if self.ni0 is not None:
            x = np.linspace(self.x0_scaled, self.x1_scaled, self.ni0.shape[1])
            ni0 = scipy.interpolate.interp1d(x,self.ni0)
            self.p0_func = [fn.Function(P)]*self.ni0.shape[0]
            for k in range(self.ni0.shape[0]):
                self.p0_func[k].vector()[:] = ni0(self.X)[k,:][dof2vtx]
                fn.assign( self.w.sub(1+k),
                    fn.interpolate( self.p0_func[k], self.W.sub(k+1).collapse() ) )

        # u represents voltage , p concentrations
        uplam = fn.split(self.w)
        self.u, self.p, self.lam = (uplam[0],[*uplam[1:(self.M+1)]],[*uplam[(self.M+1):]])

        # v, q and mu represent respetive test functions
        vqmu = fn.TestFunctions(self.W)
        self.v, self.q, self.mu = (vqmu[0],[*vqmu[1:(self.M+1)]],[*vqmu[(self.M+1):]])



    def __init__(self, *args, **kwargs):
        self.init(*args,**kwargs)

        # tolerance for identifying boundaries
        self.bctol = 1e-14
        self.K     = 0 # number of Lagrange multipliers (constraints)
        self.constraints = 0 # constraint kernels
        self.discretize()