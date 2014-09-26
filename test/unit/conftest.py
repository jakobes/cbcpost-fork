import pytest
from dolfin import MPI, MPI_Comm, Function, Expression, set_log_level, UnitCubeMesh, UnitSquareMesh
set_log_level(40)
import shutil
from cbcpost import Field

@pytest.fixture(scope="module", params=[2,3])
def mesh(request):
    if request.param == 2:
        return UnitSquareMesh(8,8)
    else:
        return UnitCubeMesh(6,6,6)

@pytest.fixture(scope="function")
def casedir():
    casedir = "test_saver"
    MPI.barrier(MPI_Comm())
    try:
        shutil.rmtree(casedir)
    except:
        pass

    return casedir


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
    
    
class MockFunctionField(Field):
    def __init__(self, Q, params=None, **args):
        Field.__init__(self, params, **args)
        self.f = Function(Q)
    
    def before_first_compute(self, get):
        t = get('t')
        self.expr = Expression("1+x[0]*x[1]*t", t=t)
        
    def compute(self, get):
        t = get('t')
        self.expr.t = t
        self.f.interpolate(self.expr)
        return self.f

class MockVectorFunctionField(Field):
    def __init__(self, V, params=None):
        Field.__init__(self, params)
        self.f = Function(V)
    
    def before_first_compute(self, get):
        t = get('t')

        
        D = self.f.function_space().mesh().geometry().dim()
        if D == 2:
            self.expr = Expression(("1+x[0]*t", "3+x[1]*t"), t=t)
        elif D == 3:
            self.expr = Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t)
        
    def compute(self, get):
        t = get('t')
        self.expr.t = t
        self.f.interpolate(self.expr)
        return self.f
     
class MockTupleField(Field):
    def compute(self, get):
        t = get('t')
        return (t, 3*t, 1+5*t)
    
class MockScalarField(Field):
    def compute(self, get):
        t = get('t')
        return 3*t**0.5