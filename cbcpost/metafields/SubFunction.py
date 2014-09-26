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

from cbcpost.fieldbases.Field import Field
from dolfin import Function, VectorFunctionSpace, FunctionSpace, project, as_vector, MPI, FunctionAssigner

from dolfin import plot, interpolate, interactive, norm, errornorm

def import_fenicstools():
    import fenicstools
    return fenicstools

class SubFunction(Field):
    "SubFunction is used to interpolate a Field on a non-matching mesh"
    def __init__(self, field, mesh, params=None, label=None):
        Field.__init__(self, params, label)
        
        import imp
        try:
            imp.find_module("mpi4py")
        except:
            raise ImportError("Can't find module mpi4py. This is required for SubFunction.")

        self._ft = import_fenicstools()

        self.mesh = mesh

        # Store only name, don't need the field
        if isinstance(field, Field):
            field = field.name
        self.valuename = field

    @property
    def name(self):
        n = "SubFunction_%s" % self.valuename
        if self.label: n += "_"+self.label
        return n

    def before_first_compute(self, get):
        u = get(self.valuename)
        
        V = u.function_space()
        element = V.ufl_element()        
        family = element.family()
        degree = element.degree()
        
        if u.rank() == 1:
            FS = VectorFunctionSpace(self.mesh, family, degree)
            FS_scalar = FS.sub(0).collapse()
            self.us = Function(FS_scalar)
            self.assigner = FunctionAssigner(FS, [FS_scalar]*FS.num_sub_spaces())
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
            for _u in u:
                U.append(self._ft.interpolate_nonmatching_mesh(_u, self.us.function_space()))
            MPI.barrier()
            
            self.assigner.assign(self.u, U)

        elif u.rank() == 0:
            U = self._ft.interpolate_nonmatching_mesh(u, self.u.function_space())
            MPI.barrier()
            
            # FIXME: This gives a PETSc-error (VecCopy). Unnecessary interpolation used instead.
            #self.u.assign(U)
            self.u.assign(interpolate(U, self.u.function_space()))
        return self.u
    



if __name__ == '__main__':
    from dolfin import *
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
    
    
    
    
    
    
    
    
    

