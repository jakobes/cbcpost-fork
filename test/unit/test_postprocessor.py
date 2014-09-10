from cbcpost import PostProcessor, ParamDict
from cbcpost import SolutionField, Field, MetaField, MetaField2, Norm
import os, shutil, pickle
"""
cbcpost.PostProcessor.add_field       cbcpost.PostProcessor.get_playlog
cbcpost.PostProcessor.add_fields      cbcpost.PostProcessor.get_savedir
cbcpost.PostProcessor.default_params  cbcpost.PostProcessor.mro
cbcpost.PostProcessor.description     cbcpost.PostProcessor.shortname
cbcpost.PostProcessor.finalize_all    cbcpost.PostProcessor.store_mesh
cbcpost.PostProcessor.get             cbcpost.PostProcessor.store_params
cbcpost.PostProcessor.get_casedir     cbcpost.PostProcessor.update_all
"""

class MockField(Field):
    def __init__(self, params=None):
        Field.__init__(self, params=params)
        self.touched = 0
        self.finalized = False
    def after_last_compute(self, get):
        self.finalized = True
    
class MockVelocity(MockField):
    def __init__(self, params=None):
        MockField.__init__(self, params)
    
    def compute(self, get):
        self.touched += 1
        return "u"

class MockPressure(MockField):
    def __init__(self, params=None):
        MockField.__init__(self, params)
    
    def compute(self, get):
        self.touched += 1
        return "p"
    
class MockVelocityGradient(MockField):
    def __init__(self, params=None):
        MockField.__init__(self, params)
        
    def compute(self, get):
        self.touched += 1
        u = get("MockVelocity")
        return "grad(%s)" %u
    
class MockStrain(MockField):
    def __init__(self, params=None):
        MockField.__init__(self, params)
    
    def compute(self, get):
        self.touched += 1
        Du = get("MockVelocityGradient")
        return "epsilon(%s)" % Du
    
class MockStress(MockField):
    def __init__(self, params=None):
        MockField.__init__(self, params)

    def compute(self, get):
        self.touched += 1
        epsilon = get("MockStrain")
        p = get("MockPressure")
        return "sigma(%s, %s)" % (epsilon, p)

def test_add_field():
    pp = PostProcessor()
    pp.add_field(SolutionField("foo"))
    assert "foo" in pp._fields.keys()
    pp.add_field(SolutionField("bar"))
    assert set(["foo", "bar"]) == set(pp._fields.keys())
    
    pp.add_fields([
        MetaField("foo"),
        MetaField2("foo", "bar"),
    ])
    
    assert set(["foo", "bar", "MetaField_foo", "MetaField2_foo_bar"]) == set(pp._fields.keys())

def test_finalize_all():
    # Move to conftest, make fixture for casedir
    casedir = "mockresults"
    if os.path.isdir(casedir):
        shutil.rmtree(casedir)
        
    pp = PostProcessor(dict(casedir=casedir))
    
    velocity = MockVelocity(dict(finalize=True))
    pressure = MockPressure()
    pp.add_fields([velocity, pressure])

    pp.get("MockVelocity")
    pp.get("MockPressure")
    
    # Nothing finalized yet
    assert pp._finalized == {}
    assert velocity.finalized == False
    
    # finalize_all should finalize velocity only
    pp.finalize_all()
    assert pp._finalized == {"MockVelocity": "u"}
    assert velocity.finalized == True
    
    # Still able to get it
    assert pp.get("MockVelocity") == "u"
    
def test_get():
    pp = PostProcessor()
    velocity = MockVelocity()
    pp.add_field(velocity)
    
    # Check that compute is triggered
    assert velocity.touched == 0
    assert pp.get("MockVelocity") == "u"
    assert velocity.touched == 1
    
    # Check that get doesn't trigger second compute count
    pp.get("MockVelocity")
    assert velocity.touched == 1
    
def test_compute_calls():
    pressure = MockPressure()
    velocity = MockVelocity()
    Du = MockVelocityGradient()
    epsilon = MockStrain()
    sigma = MockStress()

    # Add fields to postprocessor
    pp = PostProcessor()
    pp.add_fields([pressure, velocity, Du, epsilon, sigma])
    
    # Nothing has been computed yet
    assert velocity.touched == 0
    assert Du.touched == 0
    assert epsilon.touched == 0
    assert pressure.touched == 0
    assert sigma.touched == 0
    
    # Get strain twice
    for i in range(2):
        strain = pp.get("MockStrain")
        # Check value
        assert strain == "epsilon(grad(u))"
        # Check the right things are computed but only the first time
        assert velocity.touched == 1 # Only increased first iteration!
        assert Du.touched == 1 # ...
        assert epsilon.touched == 1 # ...
        assert pressure.touched == 0 # Not computed!
        assert sigma.touched == 0 # ...
        
    # Get stress twice
    for i in range(2):
        stress = pp.get("MockStress")
        # Check value
        assert stress == "sigma(epsilon(grad(u)), p)"
        # Check the right things are computed but only the first time
        assert velocity.touched == 1 # Not recomputed!
        assert Du.touched == 1 # ...
        assert epsilon.touched == 1 # ...
        assert pressure.touched == 1 # Only increased first iteration!
        assert sigma.touched == 1 # ...
    
def test_get_casedir():
    # Move to conftest, make fixture for casedir
    casedir = "mockresults"
    if os.path.isdir(casedir):
        shutil.rmtree(casedir)
    pp = PostProcessor(dict(casedir=casedir))
    
    assert os.path.isdir(pp.get_casedir())
    assert os.path.samefile(pp.get_casedir(), casedir)
    
    pp.update_all({}, 0.0, 0)
    
    assert len(os.listdir(pp.get_casedir())) == 1
    pp._saver._clean_casedir()
    assert len(os.listdir(pp.get_casedir())) == 0
    
def test_playlog():
    casedir = "mockresults"
    if os.path.isdir(casedir):
        shutil.rmtree(casedir)
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

def test_store_mesh():
    # Move to conftest, make fixture for casedir
    casedir = "mockresults"
    if os.path.isdir(casedir):
        shutil.rmtree(casedir)
        
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
    f = HDF5File(os.path.join(pp.get_casedir(), "mesh.hdf5"), 'r')
    f.read(mesh2, "Mesh")
    
    celldomains2 = CellFunction("size_t", mesh2)
    f.read(celldomains2, "CellDomains")
    facetdomains2 = FacetFunction("size_t", mesh2)
    f.read(facetdomains2, "FacetDomains")   
    
    e = Expression("1+x[1]")
    
    C1 = assemble(e*dx(1), mesh=mesh, cell_domains=celldomains)
    C2 = assemble(e*dx(1), mesh=mesh2, cell_domains=celldomains2)
    assert C1 == C2
    
    F1 = assemble(e*ds(1), mesh=mesh, exterior_facet_domains=facetdomains)
    F2 = assemble(e*ds(1), mesh=mesh2, exterior_facet_domains=facetdomains2)
    assert F1 == F2
    
def test_store_params():
    # Move to conftest, make fixture for casedir
    casedir = "mockresults"
    if os.path.isdir(casedir):
        shutil.rmtree(casedir)

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

def test_update_all():    
    pressure = SolutionField("MockPressure") #MockPressure()
    velocity = SolutionField("MockVelocity") #MockVelocity()
    Du = MockVelocityGradient()
    epsilon = MockStrain(dict(start_timestep=3))
    sigma = MockStress(dict(start_time=0.5, end_time=0.8))
    
    # Add fields to postprocessor
    pp = PostProcessor()
    pp.add_fields([pressure, velocity, Du, epsilon, sigma])
    
    N = 11
    T = [(i, float(i)/(N-1)) for i in xrange(N)]
    
    for timestep, t in T:
        pp.update_all({"MockPressure": lambda: "p"+str(timestep), "MockVelocity": lambda: "u"+str(timestep)}, t, timestep)
        
        assert Du.touched == timestep+1
        
        assert pp._cache[0]["MockPressure"] == "p%d" %timestep
        assert pp._cache[0]["MockVelocity"] == "u%d" %timestep
        assert pp._cache[0]["MockVelocityGradient"] == "grad(u%d)" %timestep
        
        if timestep >= 3:
            assert pp._cache[0]["MockStrain"] == "epsilon(grad(u%d))" %timestep
        else:
            assert "MockStrain" not in pp._cache[0]
            
        if 0.5 <= t <= 0.8:
            assert pp._cache[0]["MockStress"] == "sigma(epsilon(grad(u%d)), p%d)" %(timestep, timestep)
        else:
            assert "MockStress" not in pp._cache[0]
            
    


