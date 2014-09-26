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

@pytest.fixture(scope="function")
def casedir():
    casedir = "test_saver"
    MPI.barrier(mpi_comm_world())
    try:
        shutil.rmtree(casedir)
    except:
        pass
    MPI.barrier(mpi_comm_world())
    return casedir


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
    
    for mf in [mtf, msf]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".db"))
        
        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'shelve' in md["0"]
        assert md['saveformats'] == ['shelve']
        
        # Read back
        data = shelve.open(os.path.join(pp.get_savedir(mf.name), mf.name+".db"))
        for i in ["0", "1", "2"]:
            d = data[i]
        
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
    
    for mf, FS in [(mff, Q), (mvff, V)]:
        assert os.path.isdir(pp.get_savedir(mf.name))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), "metadata.db"))
        assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+".pvd"))
        if MPI.num_processes() == 1:
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.vtu" %0))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.vtu" %1))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.vtu" %2))
        else:
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.pvtu" %0))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.pvtu" %1))
            assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"%0.6d.pvtu" %2))
            
            for i in range(MPI.num_processes()):
                assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"_p%d_%0.6d.vtu" %(i,0)))
                assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"_p%d_%0.6d.vtu" %(i,1)))
                assert os.path.isfile(os.path.join(pp.get_savedir(mf.name), mf.name+"_p%d_%0.6d.vtu" %(i,2)))
                
        
        md = shelve.open(os.path.join(pp.get_savedir(mf.name), "metadata.db"), 'r')
        assert 'pvd' in md["0"]
        assert 'pvd' in md["1"]
        assert 'pvd' in md["2"]       
        assert md['saveformats'] == ['pvd']
        
        assert len(os.listdir(pp.get_savedir(mf.name))) == 1+1+3+int(MPI.num_processes()!=1)*MPI.num_processes()*3
    
def test_get_casedir(casedir):
    pp = PostProcessor(dict(casedir=casedir))
    
    assert os.path.isdir(pp.get_casedir())
    assert os.path.samefile(pp.get_casedir(), casedir)
    
    pp.update_all({}, 0.0, 0)
    
    assert len(os.listdir(pp.get_casedir())) == 1
    pp._saver._clean_casedir()
    assert len(os.listdir(pp.get_casedir())) == 0
    
def test_playlog(casedir):
    pp = PostProcessor(dict(casedir=casedir))

    # Test playlog
    playlog = pp.get_playlog()
    assert playlog == {}
    pp.update_all({}, 0.0, 0)
    playlog = pp.get_playlog()
    assert playlog == {"0": {"t": 0.0}}
    
    pp.update_all({}, 0.1, 1)
    playlog = pp.get_playlog()
    assert playlog == {"0": {"t": 0.0}, "1": {"t": 0.1}}

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
    f.read(mesh2, "Mesh")
    
    celldomains2 = CellFunction("size_t", mesh2)
    f.read(celldomains2, "CellDomains")
    facetdomains2 = FacetFunction("size_t", mesh2)
    f.read(facetdomains2, "FacetDomains")   
    
    e = Expression("1+x[1]")
    
    C1 = assemble(e*dx(1), mesh=mesh, cell_domains=celldomains)
    C2 = assemble(e*dx(1), mesh=mesh2, cell_domains=celldomains2)
    assert abs(C1-C2) < 1e-10
    
    F1 = assemble(e*ds(1), mesh=mesh, exterior_facet_domains=facetdomains)
    F2 = assemble(e*ds(1), mesh=mesh2, exterior_facet_domains=facetdomains2)
    assert abs(F1-F2) < 1e-10
    
def test_store_params(casedir):
    pp = PostProcessor()
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

