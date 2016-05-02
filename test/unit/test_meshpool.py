#!/usr/bin/env py.test

import pytest

from dolfin import UnitSquareMesh, UnitCubeMesh
from cbcpost import MeshPool
from numpy import argmin, array
from numpy.linalg import norm

def test_meshpool_base_functionality(dim):
    
    if dim == 2:
        mesh1 = UnitSquareMesh(6,6)
        mesh2 = UnitSquareMesh(6,6)
        mesh3 = UnitSquareMesh(6,6)
        mesh4 = UnitSquareMesh(7,7)
    elif dim == 3:
        mesh1 = UnitCubeMesh(6,6,6)
        mesh2 = UnitCubeMesh(6,6,6)
        mesh3 = UnitCubeMesh(6,6,6)
        mesh4 = UnitCubeMesh(7,7,7)

    mp = argmin(norm(mesh3.coordinates()-array([0.5]*dim), axis=1))
    mesh3.coordinates()[mp,:] += 0.00001
    
    mesh1 = MeshPool(mesh1)
    mesh2 = MeshPool(mesh2)
    mesh3 = MeshPool(mesh3)
    mesh4 = MeshPool(mesh4)
    
    assert mesh1.id() == mesh2.id(), "1!=2"
    assert mesh1.id() != mesh3.id(), "1==3"
    assert mesh1.id() != mesh4.id(), "1==4"
    assert mesh3.id() != mesh4.id(), "3==4"
    
    # FIXME: While weak referencing meshes don't work, we can not run the following tests
    """
    assert len(MeshPool._existing) == 3
    
    del mesh1
    del mesh2
    del mesh3
    del mesh4
    
    assert len(MeshPool._existing) == 0
    """