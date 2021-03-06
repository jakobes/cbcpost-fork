#!/usr/bin/env py.test

import pytest

from conftest import MockFunctionField, MockVectorFunctionField, MockTupleField, MockScalarField

from cbcpost import PostProcessor, SpacePool, ParamDict
from cbcpost.saver import Saver
from cbcpost import Field

from dolfin import *
import os
import shelve
import shutil
import pickle


def test_default_save(mesh, casedir):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    mff = MockFunctionField(Q, dict(save=True))
    mvff = MockVectorFunctionField(V, dict(save=True))
    mtf = MockTupleField(dict(save=True))
    msf = MockScalarField(dict(save=True))

    pp = PostProcessor(dict(casedir=casedir))
    pp.add_fields([mff, mvff, mtf, msf])

    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.2, 2)
    pp.finalize_all()

    for mf in [mff, mvff]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".hdf5"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".h5"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".xdmf"))

        assert len(os.listdir(pp.get_savedir(mf.name))) == 4

        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'hdf5' in md["0"]
        assert 'hdf5' in md['saveformats']
        assert 'xdmf' in md["0"]
        assert 'xdmf' in md['saveformats']
        assert set(md['saveformats']) == set(['hdf5', 'xdmf'])
        md.close()


    for mf in [mtf, msf]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".txt"))

        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'txt' in md["0"]
        assert 'txt' in md['saveformats']
        assert 'shelve' in md["0"]
        assert 'shelve' in md['saveformats']
        assert set(md['saveformats']) == set(['txt', 'shelve'])
        md.close()



def test_hdf5_save(mesh, casedir):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    mff = MockFunctionField(Q, dict(save=True, save_as="hdf5"))
    mvff = MockVectorFunctionField(V, dict(save=True, save_as="hdf5"))

    pp = PostProcessor(dict(casedir=casedir))
    pp.add_fields([mff, mvff])

    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.2, 2)
    pp.finalize_all()

    for mf, FS in [(mff, Q), (mvff, V)]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".hdf5"))

        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'hdf5' in md["0"]
        assert 'hdf5' in md["1"]
        assert 'hdf5' in md["2"]
        assert 'hdf5' in md['saveformats']

        assert md['saveformats'] == ['hdf5']
        md.close()

        assert len(os.listdir(pp.get_savedir(mf.name))) == 2

        # Read back
        hdf5file = HDF5File(mpi_comm_world(), os.path.join(pp.get_savedir(mf.name), mf.name+".hdf5"), 'r')
        f = Function(FS)
        for i in ["0", "1", "2"]:
            hdf5file.read(f, mf.name+i)

        assert norm(f) == norm(pp.get(mf.name))


def test_xml_save(mesh, casedir):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    mff = MockFunctionField(Q, dict(save=True, save_as="xml"))
    mvff = MockVectorFunctionField(V, dict(save=True, save_as="xml"))

    pp = PostProcessor(dict(casedir=casedir))
    pp.add_fields([mff, mvff])

    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.2, 2)
    pp.finalize_all()

    for mf, FS in [(mff, Q), (mvff, V)]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "mesh.hdf5"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"0.xml"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"1.xml"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"2.xml"))

        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'xml' in md["0"]
        assert 'xml' in md["1"]
        assert 'xml' in md["2"]
        assert 'xml' in md['saveformats']

        assert md['saveformats'] == ['xml']
        md.close()

        assert len(os.listdir(pp.get_savedir(mf.name))) == 1+1+3

        # Read back
        for i in ["0", "1", "2"]:
            f = Function(FS, os.path.join(pp.get_savedir(mf.name), mf.name+i+".xml"))

        assert norm(f) == norm(pp.get(mf.name))


def test_xmlgz_save(mesh, casedir):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    mff = MockFunctionField(Q, dict(save=True, save_as="xml.gz"))
    mvff = MockVectorFunctionField(V, dict(save=True, save_as="xml.gz"))

    pp = PostProcessor(dict(casedir=casedir))
    pp.add_fields([mff, mvff])

    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.2, 2)
    pp.finalize_all()

    for mf, FS in [(mff, Q), (mvff, V)]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "mesh.hdf5"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"0.xml.gz"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"1.xml.gz"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"2.xml.gz"))

        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'xml.gz' in md["0"]
        assert 'xml.gz' in md["1"]
        assert 'xml.gz' in md["2"]
        assert 'xml.gz' in md['saveformats']

        assert md['saveformats'] == ['xml.gz']
        md.close()

        assert len(os.listdir(pp.get_savedir(mf.name))) == 1+1+3

        # Read back
        for i in ["0", "1", "2"]:
            f = Function(FS, os.path.join(pp.get_savedir(mf.name), mf.name+i+".xml.gz"))

        assert norm(f) == norm(pp.get(mf.name))


def test_shelve_save(mesh, casedir):
    mtf = MockTupleField(dict(save=True, save_as="shelve"))
    msf = MockScalarField(dict(save=True, save_as="shelve"))

    pp = PostProcessor(dict(casedir=casedir))
    pp.add_fields([mtf, msf])

    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.2, 2)
    pp.finalize_all()

    for mf in [mtf, msf]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".db"))

        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'shelve' in md["0"]
        assert md['saveformats'] == ['shelve']
        md.close()

        # Read back
        data = shelve.open(os.path.join(pp.get_savedir(mf.name), mf.name+".db"), 'r')
        for i in ["0", "1", "2"]:
            d = data[i]
        data.close()

        assert d == pp.get(mf.name)

def test_pvd_save(mesh, casedir):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    mff = MockFunctionField(Q, dict(save=True, save_as="pvd"))
    mvff = MockVectorFunctionField(V, dict(save=True, save_as="pvd"))

    pp = PostProcessor(dict(casedir=casedir))
    pp.add_fields([mff, mvff])

    pp.update_all({}, 0.0, 0)
    pp.update_all({}, 0.1, 1)
    pp.update_all({}, 0.2, 2)
    pp.finalize_all()

    for mf, FS in [(mff, Q), (mvff, V)]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".pvd"))
        if MPI.size(mpi_comm_world()) == 1:
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.vtu" %0))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.vtu" %1))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.vtu" %2))
        else:
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.pvtu" %0))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.pvtu" %1))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.pvtu" %2))

            for i in range(MPI.size(mpi_comm_world())):
                assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"_p%d_%0.6d.vtu" %(i,0)))
                assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"_p%d_%0.6d.vtu" %(i,1)))
                assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"_p%d_%0.6d.vtu" %(i,2)))


        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'pvd' in md["0"]
        assert 'pvd' in md["1"]
        assert 'pvd' in md["2"]
        assert md['saveformats'] == ['pvd']
        md.close()

        assert len(os.listdir(pp.get_savedir(mf.name))) == 1+1+3+int(MPI.size(mpi_comm_world())!=1)*MPI.size(mpi_comm_world())*3

def test_get_casedir(casedir):
    pp = PostProcessor(dict(casedir=casedir))

    assert os.path.isdir(pp.get_casedir())
    assert os.path.samefile(pp.get_casedir(), casedir)

    pp.update_all({}, 0.0, 0)

    assert len(os.listdir(pp.get_casedir())) == 1
    pp.clean_casedir()
    assert len(os.listdir(pp.get_casedir())) == 0

def test_playlog(casedir):
    pp = PostProcessor(dict(casedir=casedir))

    # Test playlog
    assert not os.path.isfile(os.path.join(casedir, 'play.db'))
    MPI.barrier(mpi_comm_world())

    pp.update_all({}, 0.0, 0)
    pp.finalize_all()

    playlog = pp.get_playlog('r')
    assert playlog == {"0": {"t": 0.0}}
    playlog.close()

    pp.update_all({}, 0.1, 1)
    pp.finalize_all()
    playlog = pp.get_playlog('r')
    assert playlog == {"0": {"t": 0.0}, "1": {"t": 0.1}}
    playlog.close()

def test_store_mesh(casedir):
    pp = PostProcessor(dict(casedir=casedir))

    from dolfin import (UnitSquareMesh, CellFunction, FacetFunction, AutoSubDomain,
                        Mesh, HDF5File, assemble, Expression, ds, dx)

    # Store mesh
    mesh = UnitSquareMesh(6,6)
    celldomains = CellFunction("size_t", mesh)
    celldomains.set_all(0)
    AutoSubDomain(lambda x: x[0]<0.5).mark(celldomains, 1)

    facetdomains = FacetFunction("size_t", mesh)
    AutoSubDomain(lambda x, on_boundary: x[0]<0.5 and on_boundary).mark(facetdomains, 1)

    pp.store_mesh(mesh, celldomains, facetdomains)


    # Read mesh back
    mesh2 = Mesh()
    f = HDF5File(mpi_comm_world(), os.path.join(pp.get_casedir(), "mesh.hdf5"), 'r')
    f.read(mesh2, "Mesh", False)

    celldomains2 = CellFunction("size_t", mesh2)
    f.read(celldomains2, "CellDomains")
    facetdomains2 = FacetFunction("size_t", mesh2)
    f.read(facetdomains2, "FacetDomains")

    e = Expression("1+x[1]", degree=1)

    dx1 = dx(1, domain=mesh, subdomain_data=celldomains)
    dx2 = dx(1, domain=mesh2, subdomain_data=celldomains2)
    C1 = assemble(e*dx1)
    C2 = assemble(e*dx2)
    assert abs(C1-C2) < 1e-10

    ds1 = ds(1, domain=mesh, subdomain_data=facetdomains)
    ds2 = ds(1, domain=mesh2, subdomain_data=facetdomains2)
    F1 = assemble(e*ds1)
    F2 = assemble(e*ds2)
    assert abs(F1-F2) < 1e-10

def test_store_params(casedir):
    pp = PostProcessor(dict(casedir=casedir))
    params = ParamDict(Field=Field.default_params(),
                       PostProcessor=PostProcessor.default_params())

    pp.store_params(params)

    # Read back params
    params2 = None
    with open(os.path.join(pp.get_casedir(), "params.pickle"), 'r') as f:
        params2 = pickle.load(f)
    assert params2 == params

    str_params2 = open(os.path.join(pp.get_casedir(), "params.txt"), 'r').read()
    assert str_params2 == str(params)
