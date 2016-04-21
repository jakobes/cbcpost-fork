#!/usr/bin/env py.test

from dolfin import UnitSquareMesh
from cbcpost import MeshPool
from numpy import argmin, array
from numpy.linalg import norm

def test_meshpool_base_functionality():
    mesh1 = UnitSquareMesh(6,6)
    mesh2 = UnitSquareMesh(6,6)
    mesh3 = UnitSquareMesh(6,6)
    mp = argmin(norm(mesh3.coordinates()-array([0.5,0.5]), axis=1))
    mesh3.coordinates()[mp,:] += 0.0001
    
    mesh1 = MeshPool(mesh1)
    mesh2 = MeshPool(mesh2)
    mesh3 = MeshPool(mesh3)
    
    assert mesh1.id() == mesh2.id()
    assert mesh1.id() != mesh3.id()
