#
# Copyright 2020 Johannes Hoermann (U. Freiburg)
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
""" Compute ion concentrations consistent with general
Poisson-Nernst-Planck (PNP) equations via FEniCS.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import numpy as np
import scipy.interpolate
import fenics as fn

from matscipy.electrochemistry.poisson_nernst_planck_solver import PoissonNernstPlanckSystem


class Boundary(fn.SubDomain):
    """Mark a point to be within the domain boundary for fenics."""
    # Boundary causes crash kernel if __init__ doe not call super().__init__()
    def __init__(self, x0=0, tol=1e-14):
        super().__init__()
        self.tol = tol
        self.x0 = x0

    def inside(self, x, on_boundary):
        """Mark a point to be within the domain boundary for fenics."""
        return on_boundary and fn.near(x[0], self.x0, self.tol)


class PoissonNernstPlanckSystemFEniCS(PoissonNernstPlanckSystem):
    """Describes and solves a 1D Poisson-Nernst-Planck system,
    using log concentrations internally"""

    @property
    def X(self):
        return self.mesh.coordinates().flatten()  # only 1D

    def solve(self):
        """Evoke FEniCS FEM solver.

        Returns
        -------
        uij : (Ni,) ndarray
            potential at Ni grid points
        nij : (M,Nij) ndarray
            concentrations of M species at Ni grid points
        lamj: (L,) ndarray
            value of L Lagrange multipliers
        """

        # weak form and FEM scheme:

        # in the weak form, u and v are the trial and test functions associated
        # with the Poisson part, p and q the trial and test functions associated
        # with the Nernst-Planck part. lam and mu are trial and test fuctions
        # associated to constraints introduced via Lagrange multipliers.
        # w is the whole set of trial functions [u,p,lam]
        # W is the space all w live in.
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

        fn.solve(F == 0, self.w, self.boundary_conditions)

        # store results:

        wij = np.array([self.w(x) for x in self.X]).T

        self.uij = wij[0, :]  # potential
        self.nij = wij[1:(self.M+1), :]  # concentrations
        self.lamj = wij[(self.M+1):]  # Lagrange multipliers

        return self.uij, self.nij, self.lamj

    def boundary_L(self, x, on_boundary):
        """Mark left boundary. Returns True if x on left boundary."""
        return on_boundary and fn.near(x[0], self.x0_scaled, self.bctol)

    def boundary_R(self, x, on_boundary):
        """Mark right boundary. Returns True if x on right boundary."""
        return on_boundary and fn.near(x[0], self.x1_scaled, self.bctol)

    def boundary_C(self, x, on_boundary):
        """Mark domain center."""
        return fn.near(x[0], (self.x0_scaled + self.x1_scaled)/2., self.bctol)

    def applyLeftPotentialDirichletBC(self, u0):
        """FEniCS Dirichlet BC u0 for potential at left boundary."""
        self.boundary_conditions.extend([
            fn.DirichletBC(self.W.sub(0), u0,
                           lambda x, on_boundary: self.boundary_L(x, on_boundary))])

    def applyRightPotentialDirichletBC(self, u0):
        """FEniCS Dirichlet BC u0 for potential at right boundary."""
        self.boundary_conditions.extend([
            fn.DirichletBC(self.W.sub(0), u0,
                           lambda x, on_boundary: self.boundary_R(x, on_boundary))])

    def applyLeftConcentrationDirichletBC(self, k, c0):
        """FEniCS Dirichlet BC c0 for k'th ion species at left boundary."""
        self.boundary_conditions.extend([
            fn.DirichletBC(self.W.sub(k+1), c0,
                           lambda x, on_boundary: self.boundary_L(x, on_boundary))])

    def applyRightConcentrationDirichletBC(self, k, c0):
        """FEniCS Dirichlet BC c0 for k'th ion species at right boundary."""
        self.boundary_conditions.extend([
            fn.DirichletBC(self.W.sub(k+1), c0,
                           lambda x, on_boundary: self.boundary_R(x, on_boundary))])

    def applyCentralReferenceConcentrationConstraint(self, k, c0):
        """FEniCS Dirichlet BC c0 for k'th ion species at right boundary."""
        self.boundary_conditions.extend([
            fn.DirichletBC(self.W.sub(k+1), c0,
                           lambda x, on_boundary: self.boundary_C(x, on_boundary))])

    # TODO: Robin BC!
    def applyLeftPotentialRobinBC(self, u0, lam0):
        self.logger.warning("Not implemented!")

    def applyRightPotentialRobinBC(self, u0, lam0):
        self.logger.warning("Not implemented!")

    def applyNumberConservationConstraint(self, k, c0):
        """
        Enforce number conservation constraint via Lagrange multiplier.
        See https://fenicsproject.org/docs/dolfin/1.6.0/python/demo/documented/neumann-poisson/python/documentation.html
        """
        self.constraints += self.lam[k]*self.q[k]*fn.dx \
            + (self.p[k]-c0)*self.mu[k]*fn.dx

    def applyPotentialDirichletBC(self, u0, u1):
        """Potential Dirichlet BC u0 and u1 on left and right boundary."""
        self.applyLeftPotentialDirichletBC(u0)
        self.applyRightPotentialDirichletBC(u1)

    def applyPotentialRobinBC(self, u0, u1, lam0, lam1):
        """Potential Robin BC on left and right boundary."""
        self.applyLeftPotentialRobinBC(u0, lam0)
        self.applyRightPotentialRobinBC(u1, lam1)

    def useStandardInterfaceBC(self):
        """Interface at left hand side and open bulk at right hand side"""
        self.boundary_conditions = []

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled
        self.u1 = 0

        self.logger.info('Left hand side Dirichlet boundary condition:  u0 = {:> 8.4g}'.format(self.u0))
        self.logger.info('Right hand side Dirichlet boundary condition: u1 = {:> 8.4g}'.format(self.u1))

        self.applyPotentialDirichletBC(self.u0, self.u1)

        for k in range(self.M):
            self.logger.info(('Ion species {:02d} right hand side concentration '
                              'Dirichlet boundary condition: c1 = {:> 8.4g}').format(k, self.c_scaled[k]))
            self.applyRightConcentrationDirichletBC(k, self.c_scaled[k])

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

        self.applyPotentialDirichletBC(self.u0, self.u1)

        # Number conservation constraints
        self.constraints = 0
        N0 = self.L_scaled*self.c_scaled  # total amount of species in cell
        for k in range(self.M):
            self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
                'Ion species {:02d} number conservation constraint'.format(k),
                N0[k], lwidth=self.label_width))
            self.applyNumberConservationConstraint(k, self.c_scaled[k])

    def useCentralReferenceConcentrationBasedCellBC(self):
        """
        Interfaces at left hand side and right hand side, species-wise
        concentration fixed at cell center."""
        self.boundary_conditions = []

        # Introduce a Lagrange multiplier per species anderson
        # rebuild discretization scheme (function spaces)
        # self.K = self.M
        # self.discretize()

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.
        self.u1 = - self.delta_u_scaled / 2.
        self.logger.info('{:>{lwidth}s} u0 = {:< 8.4g}'.format(
          'Left hand side Dirichlet boundary condition', self.u0, lwidth=self.label_width))
        self.logger.info('{:>{lwidth}s} u1 = {:< 8.4g}'.format(
          'Right hand side Dirichlet boundary condition', self.u1, lwidth=self.label_width))

        self.applyPotentialDirichletBC(self.u0, self.u1)

        for k in range(self.M):
            self.logger.info(
                'Ion species {:02d} reference concentration condition: c1 = {:> 8.4g} at cell center'.format(
                    k, self.c_scaled[k]))
            self.applyCentralReferenceConcentrationConstraint(k, self.c_scaled[k])

    def useSternLayerCellBC(self):
        """
        Interfaces at left hand side and right hand side, species-wise
        number conservation within interval."""
        self.boundary_conditions = []

        # Introduce a Lagrange multiplier per species anderson
        # rebuild discretization scheme (function spaces)
        self.constraints = 0
        self.K = self.M
        self.discretize()

        # Potential Dirichlet BC
        self.u0 = self.delta_u_scaled / 2.
        self.u1 = - self.delta_u_scaled / 2.

        boundary_markers = fn.MeshFunction(
            'size_t', self.mesh, self.mesh.topology().dim()-1)

        bx = [
            Boundary(x0=self.x0_scaled, tol=self.bctol),
            Boundary(x0=self.x1_scaled, tol=self.bctol)]

        # Boundary.mark crashes the kernel if Boundary is internal class
        for i, b in enumerate(bx):
            b.mark(boundary_markers, i)

        boundary_conditions = {
            0: {'Robin': (1./self.lambda_S_scaled, self.u0)},
            1: {'Robin': (1./self.lambda_S_scaled, self.u1)},
        }

        ds = fn.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        integrals_R = []
        for i in boundary_conditions:
            if 'Robin' in boundary_conditions[i]:
                r, s = boundary_conditions[i]['Robin']
                integrals_R.append(r*(self.u-s)*self.v*ds(i))

        self.constraints += sum(integrals_R)

        # Number conservation constraints

        N0 = self.L_scaled*self.c_scaled  # total amount of species in cell
        for k in range(self.M):
            self.logger.info('{:>{lwidth}s} N0 = {:<8.4g}'.format(
                'Ion species {:02d} number conservation constraint'.format(k),
                N0[k], lwidth=self.label_width))
            self.applyNumberConservationConstraint(k, self.c_scaled[k])

    def discretize(self):
        """Builds function space, call again after introducing constraints"""
        # FEniCS interface
        self.mesh = fn.IntervalMesh(self.N, self.x0_scaled, self.x1_scaled)

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

        # construct test and trial function space from elements
        # spanned by Lagrange polynomials for the pyhsical variables of
        # potential and concentration and global elements with a single degree
        # of freedom ('Real') for constraints.
        # For an example of this approach, refer to
        #     https://fenicsproject.org/docs/dolfin/latest/python/demos/neumann-poisson/demo_neumann-poisson.py.html
        # For another example on how to construct and split function spaces
        # for solving coupled equations, refer to
        #     https://fenicsproject.org/docs/dolfin/latest/python/demos/mixed-poisson/demo_mixed-poisson.py.html

        P = fn.FiniteElement('Lagrange', fn.interval, 3)
        R = fn.FiniteElement('Real', fn.interval, 0)
        elements = [P]*(1+self.M) + [R]*self.K

        H = fn.MixedElement(elements)
        self.W = fn.FunctionSpace(self.mesh, H)

        # solution functions
        self.w = fn.Function(self.W)

        # set initial values if available
        P = fn.FunctionSpace(self.mesh, 'P', 1)
        dof2vtx = fn.vertex_to_dof_map(P)
        if self.ui0 is not None:
            x = np.linspace(self.x0_scaled, self.x1_scaled, self.ui0.shape[0])
            ui0 = scipy.interpolate.interp1d(x, self.ui0)
            # use linear interpolation on mesh
            self.u0_func = fn.Function(P)
            self.u0_func.vector()[:] = ui0(self.X)[dof2vtx]
            fn.assign(self.w.sub(0),
                      fn.interpolate(self.u0_func, self.W.sub(0).collapse()))

        if self.ni0 is not None:
            x = np.linspace(self.x0_scaled, self.x1_scaled, self.ni0.shape[1])
            ni0 = scipy.interpolate.interp1d(x, self.ni0)
            self.p0_func = [fn.Function(P)]*self.ni0.shape[0]
            for k in range(self.ni0.shape[0]):
                self.p0_func[k].vector()[:] = ni0(self.X)[k, :][dof2vtx]
                fn.assign(self.w.sub(1+k),
                          fn.interpolate(self.p0_func[k], self.W.sub(k+1).collapse()))

        # u represents voltage , p concentrations
        uplam = fn.split(self.w)
        self.u, self.p, self.lam = (
            uplam[0], [*uplam[1:(self.M+1)]], [*uplam[(self.M+1):]])

        # v, q and mu represent respective test functions
        vqmu = fn.TestFunctions(self.W)
        self.v, self.q, self.mu = (
            vqmu[0], [*vqmu[1:(self.M+1)]], [*vqmu[(self.M+1):]])

    def __init__(self, *args, **kwargs):
        self.init(*args, **kwargs)

        # TODO: don't hard code btcol
        self.bctol = 1e-14  # tolerance for identifying domain boundaries
        self.K = 0  # number of Lagrange multipliers (constraints)
        self.constraints = 0  # holds constraint kernels
        self.discretize()
