
import pytest
import shutil, os, gc

from dolfin import MPI, mpi_comm_world, Function, Expression, set_log_level, UnitCubeMesh, UnitSquareMesh
set_log_level(40)

from cbcpost import Field

@pytest.fixture(scope="module", params=[2,3])
def mesh(request):
    if request.param == 2:
        return UnitSquareMesh(8,8)
    else:
        return UnitCubeMesh(6,6,6)

@pytest.fixture(scope="function")
def casedir(request):
    # Some code here copied from dolfin_utils dev:

    # Get directory name of test_foo.py file
    testfile = request.module.__file__
    testfiledir = os.path.dirname(os.path.abspath(testfile))

    # Construct name test_foo_tempdir from name test_foo.py
    testfilename = os.path.basename(testfile)
    outputname = testfilename.replace(".py", "_casedir")

    # Get function name test_something from test_foo.py
    function = request.function.__name__

    # Join all of these to make a unique path for this test function
    basepath = os.path.join(testfiledir, outputname)
    casedir = os.path.join(basepath, function)

    # Unlike the dolfin_utils tempdir fixture, here we make sure the directory is _deleted_:
    gc.collect() # Workaround for possible dolfin deadlock
    MPI.barrier(mpi_comm_world())
    try:
        # Only on root node in parallel
        if MPI.size(mpi_comm_world()) == 1 or MPI.rank(mpi_comm_world()) == 0:
            shutil.rmtree(casedir)
    except:
        pass
    MPI.barrier(mpi_comm_world())

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
