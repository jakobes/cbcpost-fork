#!/usr/bin/env py.test

from dolfin import UnitSquareMesh
from cbcpost import SpacePool, get_grad_space

def test_spacepool_base_functionality(mesh):
    p = SpacePool(mesh)
    d = mesh.geometry().dim()
    spaces = []
    shapes = [(), (3,), (2,4)]
    degrees = [0,1,2]
    for shape in shapes:
        for degree in degrees:
            V = p.get_custom_space("DG", degree, shape)
            assert V.ufl_element().degree() == degree
            assert V.ufl_element().value_shape() == shape

            rank = len(shape)
            shape2 = (d,)*rank
            U = p.get_space(degree, rank)
            assert U.ufl_element().degree() == degree
            assert U.ufl_element().value_shape() == shape2

            spaces.append((V,U))

    k = 0
    for shape in shapes:
        for degree in degrees:
            V0, U0 = spaces[k]; k += 1

            V = p.get_custom_space("DG", degree, shape)
            U = p.get_space(degree, len(shape))

            assert id(V0) == id(V)
            assert id(U0) == id(U)


def test_grad_spaces(mesh):
    p = SpacePool(mesh)
    d = mesh.geometry().dim()
    
    #Q = get_custom_space("CG", 1, (1,))
    Q = p.get_space(1,0)
    DQ = p.get_grad_space(Q)
    assert DQ.ufl_element().family() == "Discontinuous Lagrange"
    assert DQ.ufl_element().degree() == Q.ufl_element().degree()-1
    assert DQ.ufl_element().value_shape() == (d,)
    
    V = p.get_space(2,1)
    DV = p.get_grad_space(V)
    assert DV.ufl_element().family() == "Discontinuous Lagrange"
    assert DV.ufl_element().degree() == V.ufl_element().degree()-1
    assert DV.ufl_element().value_shape() == (d,d)
    
    W = p.get_space(2,2)
    DW = p.get_grad_space(W)
    assert DW.ufl_element().family() == "Discontinuous Lagrange"
    assert DW.ufl_element().degree() == V.ufl_element().degree()-1
    assert DW.ufl_element().value_shape() == (d,d,d)
    