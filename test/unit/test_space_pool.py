#!/usr/bin/env py.test

from dolfin import UnitSquareMesh
from cbcpost import SpacePool
#from cbcflow.utils.core import SpacePool, NSSpacePool, NSSpacePoolMixed, NSSpacePoolSplit, NSSpacePoolSegregated

def test_spacepool_base_functionality():
    mesh = UnitSquareMesh(4,4)
    p = SpacePool(mesh)
    d = 2
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
