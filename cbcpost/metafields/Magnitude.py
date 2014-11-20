# Copyright (C) 2010-2014 Simula Research Laboratory
#
# This file is part of CBCPOST.
#
# CBCPOST is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CBCPOST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CBCPOST. If not, see <http://www.gnu.org/licenses/>.
"""
Compute the (piecewise) magnitude of a Field.
"""
from cbcpost.fieldbases.MetaField import MetaField
from dolfin import (project, sqrt, Function, inner, KrylovSolver, assemble, TrialFunction,
                    TestFunction)
import numpy as np
from cbcpost.utils import cbc_warning


from dolfin import TestFunction, TrialFunction, Vector, assemble, dx, solve 

class Magnitude(MetaField):
    """ Compute the magnitude of a Function-evaluated Field.

    Supports function spaces where all subspaces are equal.
    """

    def before_first_compute(self, get):
        u = get(self.valuename)

        if isinstance(u, Function):
            if u.rank() == 0:
                self.f = Function(u.function_space())
            elif u.rank() >= 1:
                # Assume all subpaces are equal
                V = u.function_space().extract_sub_space([0]).collapse()
                mesh = V.mesh()
                el = V.ufl_element()
                self.f = Function(V)
                
                # Find out if we can operate directly on vectors, or if we have to use a projection
                # We can operate on vectors if all sub-dofmaps are ordered the same way
                # For simplicity, this is only tested for CG- or DG0-spaces
                # (this might always be true for these spaces, but better to be safe than sorry )
                self.use_project = True
                if el.family() == "Lagrange" or (el.family() == "Discontinuous Lagrange" and el.degree()==0):
                    dm = u.function_space().dofmap()
                    dm0 = V.dofmap()
                    self.use_project = False
                    for i in xrange(u.function_space().num_sub_spaces()):
                        Vi = u.function_space().extract_sub_space([i]).collapse()
                        dmi = Vi.dofmap()
                        diff = dmi.tabulate_all_coordinates(mesh)-dm0.tabulate_all_coordinates(mesh)
                        if max(abs(diff)) > 1e-12:
                            self.use_project = True
                            break

                # IF we have to use a projection, build projection matrix only once
                if self.use_project:
                    self.v = TestFunction(V)
                    M = assemble(inner(self.v,TrialFunction(V))*dx)
                    self.projection = KrylovSolver("cg", "default")
                    self.projection.set_operator(M)
                
        else:
            # Don't know how to handle object
            cbc_warning("Don't know how to calculate magnitude of object of type %s." %type(u))

    def compute(self, get):
        u = get(self.valuename)

        if isinstance(u, Function):
            if u.rank() == 0:
                self.f.vector().zero()
                self.f.vector().axpy(1.0, u.vector())
                self.f.vector().abs()
                return self.f
            elif u.rank() >= 1:
                if self.use_project:
                    b = assemble(sqrt(inner(u,u))*self.v*dx(None))
                    self.projection.solve(self.f.vector(), b)
                else:
                    self.f.vector().zero()
                    for i in xrange(u.function_space().num_sub_spaces()):
                        vec = u.split(True)[i].vector()
                        self.f.vector().axpy(1.0, vec*vec)
                    r = self.f.vector().local_range()
                    self.f.vector()[r[0]:r[1]] = np.sqrt(self.f.vector()[r[0]:r[1]])

                return self.f
        else:
            # Don't know how to handle object
            cbc_warning("Don't know how to calculate magnitude of object of type %s. Returning object." %type(u))
            return u
