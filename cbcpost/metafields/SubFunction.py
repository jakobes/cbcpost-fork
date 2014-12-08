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
"""Functionality to construct a subfunction of a Field."""
from cbcpost.fieldbases.MetaField import MetaField
from cbcpost.utils.utils import import_fenicstools

from dolfin import (Function, VectorFunctionSpace, FunctionSpace, MPI, mpi_comm_world,
                    FunctionAssigner, interpolate)

def _interpolate(u, u0):
    try:
        # Use dolfins LagrangeInterpolator if defined
        from dolfin import LagrangeInterpolator
        LagrangeInterpolator().interpolate(u,u0)        
    except:
        # Otherwise, use fenicstools
        ft = import_fenicstools()
        u.assign(ft.interpolate_nonmatching_mesh(u0, u.function_space()))

    return u

    


class SubFunction(MetaField):
    """SubFunction is used to interpolate a Field on a non-matching mesh.
    
    .. note::

        v1.4.0: This field requires fenicstools.
    
    """

    def __init__(self, field, mesh, *args, **kwargs):
        MetaField.__init__(self, field, *args, **kwargs)
        self.mesh = mesh

    def before_first_compute(self, get):
        u = get(self.valuename)

        V = u.function_space()
        element = V.ufl_element()
        family = element.family()
        degree = element.degree()

        if u.rank() == 1:
            FS = VectorFunctionSpace(self.mesh, family, degree)
            FS_scalar = FS.sub(0).collapse()
            self.assigner = FunctionAssigner(FS, [FS_scalar]*FS.num_sub_spaces())
            self.us = []
            for i in range(FS.num_sub_spaces()):
                self.us.append(Function(FS_scalar))
        elif u.rank() == 0:
            FS = FunctionSpace(self.mesh, family, degree)
        else:
            raise Exception("Does not support TensorFunctionSpace yet")

        self.u = Function(FS, name=self.name)

    def compute(self, get):
        u = get(self.valuename)

        if u.rank() == 1:
            u = u.split()
            U = []
            for i, _u in enumerate(u):
                U.append(_interpolate(self.us[i], _u))
                #U.append(self._ft.interpolate_nonmatching_mesh(_u, self.us.function_space()))
            MPI.barrier(mpi_comm_world())

            self.assigner.assign(self.u, U)

        elif u.rank() == 0:
            #U = self._ft.interpolate_nonmatching_mesh(u, self.u.function_space())
            #U = _interpolate(self.u, u)
            _interpolate(self.u, u)
            MPI.barrier(mpi_comm_world())

            # FIXME: This gives a PETSc-error (VecCopy). Unnecessary interpolation used instead.
            #self.u.assign(U)
            #self.u.assign(interpolate(U, self.u.function_space()))
        return self.u




if __name__ == '__main__':
    #from dolfin import *
    from dolfin import (Expression, UnitSquareMesh, errornorm)
    #expr_scalar = Expression("1+x[0]*x[1]")
    #expr_vector = Expression(("1+x[0]*x[1]", "x[1]-2"))
    expr_scalar = Expression("1+x[0]")
    expr_vector = Expression(("1+x[0]", "x[1]-2"))

    mesh = UnitSquareMesh(12,12)

    submesh = UnitSquareMesh(6,6)
    submesh.coordinates()[:] /= 2.0
    submesh.coordinates()[:] += 0.2

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    Q_sub = FunctionSpace(submesh, "CG", 1)
    V_sub = VectorFunctionSpace(submesh, "CG", 1)

    u = interpolate(expr_scalar, Q)
    v = interpolate(expr_vector, V)

    u_sub = interpolate(expr_scalar, Q_sub)
    v_sub = interpolate(expr_vector, V_sub)

    from fenicstools import interpolate_nonmatching_mesh
    u_sub2 = interpolate_nonmatching_mesh(u, Q_sub)
    v_sub2 = interpolate_nonmatching_mesh(v, V_sub)

    print errornorm(u_sub, u_sub2)
    print errornorm(v_sub, v_sub2)










