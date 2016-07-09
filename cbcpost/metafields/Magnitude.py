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
                    TestFunction, MPI, mpi_comm_world, FunctionAssigner, compile_extension_module,
                    dolfin_version)
import numpy as np
from cbcpost.utils import cbc_warning
from distutils.version import LooseVersion

# Import for type-checking
from collections import Iterable
from numbers import Number

from dolfin import TestFunction, TrialFunction, Vector, assemble, dx, solve

_sqrt_in_place_code = """
#include "petscvec.h"
#include <dolfin/la/PETScVector.h>
namespace dolfin
{
    void sqrt_in_place(std::shared_ptr<GenericVector> vec)
    {
        VecSqrtAbs(vec->down_cast<PETScVector>().vec());
    }

}
"""
try:
    sqrt_in_place = compile_extension_module(_sqrt_in_place_code).sqrt_in_place
except:
    sqrt_in_place = None


class Magnitude(MetaField):
    """ Compute the magnitude of a Function-evaluated Field.

    Supports function spaces where all subspaces are equal.
    """

    def before_first_compute(self, get):
        u = get(self.valuename)

        if isinstance(u, Function):

            if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
                rank = len(u.ufl_shape)
            else:
                rank = u.rank()

            if rank == 0:
                self.f = Function(u.function_space())
            elif rank >= 1:
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
                        try:
                            # For 1.6.0+ and newer
                            diff = Vi.tabulate_dof_coordinates()-V.tabulate_dof_coordinates()
                        except:
                            # For 1.6.0 and older
                            diff = dmi.tabulate_all_coordinates(mesh)-dm0.tabulate_all_coordinates(mesh)
                        if len(diff) > 0:
                            max_diff = max(abs(diff))
                        else:
                            max_diff = 0.0
                        max_diff = MPI.max(mpi_comm_world(), max_diff)
                        if max_diff > 1e-12:
                            self.use_project = True
                            break
                        self.assigner = FunctionAssigner([V]*u.function_space().num_sub_spaces(), u.function_space())
                        self.subfuncs = [Function(V) for i in range(u.function_space().num_sub_spaces())]

                # IF we have to use a projection, build projection matrix only once
                if self.use_project:
                    self.v = TestFunction(V)
                    M = assemble(inner(self.v,TrialFunction(V))*dx)
                    self.projection = KrylovSolver("cg", "default")
                    self.projection.set_operator(M)
        elif isinstance(u, Iterable) and all(isinstance(_u, Number) for _u in u):
            pass
        elif isinstance(u, Number):
            pass
        else:
            # Don't know how to handle object
            cbc_warning("Don't know how to calculate magnitude of object of type %s." %type(u))

    def compute(self, get):
        u = get(self.valuename)

        if isinstance(u, Function):
            if not hasattr(self, "use_project"):
                self.before_first_compute(get)

            if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
                rank = len(u.ufl_shape)
            else:
                rank = u.rank()

            if rank == 0:
                self.f.vector().zero()
                self.f.vector().axpy(1.0, u.vector())
                self.f.vector().abs()
                return self.f
            elif rank >= 1:
                if self.use_project:
                    b = assemble(sqrt(inner(u,u))*self.v*dx(None))
                    self.projection.solve(self.f.vector(), b)
                else:
                    self.assigner.assign(self.subfuncs, u)
                    self.f.vector().zero()
                    for i in xrange(u.function_space().num_sub_spaces()):
                        vec = self.subfuncs[i].vector()
                        vec.apply('')
                        self.f.vector().axpy(1.0, vec*vec)

                    try:
                        sqrt_in_place(self.f.vector())
                    except:
                        r = self.f.vector().local_range()
                        self.f.vector()[r[0]:r[1]] = np.sqrt(self.f.vector()[r[0]:r[1]])
                    self.f.vector().apply('')

                return self.f
        elif isinstance(u, Iterable) and all(isinstance(_u, Number) for _u in u):
            return np.sqrt(sum(_u**2 for _u in u))
        elif isinstance(u, Number):
            return abs(u)
        else:
            # Don't know how to handle object
            cbc_warning("Don't know how to calculate magnitude of object of type %s. Returning object." %type(u))
            return u
