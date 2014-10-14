#!/usr/bin/env py.test

from cbcpost import PostProcessor, SpacePool
#from conftest import MockScalarField
from conftest import MockFunctionField, MockVectorFunctionField, MockTupleField, MockScalarField
from dolfin import FunctionSpace, VectorFunctionSpace

def test_pyplot():
    pp = PostProcessor()
    
    pp.add_field(MockScalarField(dict(plot=True)))
    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.6, 2)
    pp.update_all({}, 1.6, 3)
    
def test_dolfinplot(mesh):
    # TODO: This fails in dolfin 1.3 sometimes
    if mesh.geometry().dim() == 2:
        return

    pp = PostProcessor()
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)
    
    pp.add_field(MockFunctionField(Q, dict(plot=True)))
    pp.add_field(MockVectorFunctionField(V, dict(plot=True)))
    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.6, 2)
    pp.update_all({}, 1.6, 3)
