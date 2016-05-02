#!/usr/bin/env py.test
"""
Tests of the different metafields (through the PostProcessor).
"""
from conftest import MockFunctionField, MockVectorFunctionField, MockTupleField, MockScalarField
from collections import defaultdict

from cbcpost import *
from cbcpost.utils import cbc_warning, create_submesh, import_fenicstools

import pytest

from dolfin import *
from math import sqrt
from numpy.random import random
from numpy import linspace

# TODO: Setting global parameters here affects all tests and makes test execution depend on test ordering.
# Avoid negative norms caused by instable tensor representation:
import dolfin
dolfin.parameters["form_compiler"]["representation"] = "quadrature"
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1
dolfin.parameters["allow_extrapolation"] = True


# TODO: Move these to shared code:
def has_mpi4py():
    try:
        import mpi4py
        return True
    except:
        return False

def has_h5py():
    try:
        import h5py
        return True
    except:
        return False

def has_fenicstools():
    try:
        import_fenicstools()
        return True
    except:
        return False

#require_mpi4py = pytest.mark.skipif(not has_mpi4py(), reason="Requires mpi4py which is not installed.")
require_fenicstools = pytest.mark.skipif(not has_fenicstools(), reason="Requires fenicstools which is not installed.")
require_fenicstools14 = pytest.mark.skipif(dolfin_version() == '1.4.0' and not has_fenicstools(), reason="Requires fenicstools in dolfin 1.4.0 which is not installed.")
#require_h5py = pytest.mark.skipif(not has_h5py(), reason="Requires h5py which is not installed.")
skip_in_parallel = pytest.mark.skipif(MPI.size(mpi_comm_world()) != 1, reason="Currently not supported in parallel")

class MockProblem(Parameterized):
    def __init__(self, D, params=None):
        Parameterized.__init__(self, params)
        self.D = D
        if D == 2:
            self.mesh = UnitSquareMesh(6,6)
        elif D == 3:
            self.mesh = UnitCubeMesh(3,3,3)

    @classmethod
    def default_params(cls):
        params = ParamDict(
            T0 = 0.0,
            T = 2.0,
            start_timestep=0,
            dt = 0.1,
            mu = 0.1,
            rho = 0.9,
        )
        return params

@pytest.fixture(scope="module", params=[2,3])
def problem(request):
    problem = MockProblem(request.param, dict(T=2.0))
    return problem

@pytest.fixture(scope="function", autouse=True)
def set_problem_params(request, problem):
    problem.params.dt = request.getfuncargvalue('dt')

@pytest.fixture(scope="function")
def pp(problem):
    return PostProcessor(params=dict(initial_dt=problem.params.dt))

def compute_regular_timesteps(problem):
    """Compute fixed timesteps for problem.

    Returns (dt, t0, timesteps), where timesteps does not include t0.
    """
    from numpy import arange
    # Get the time range for the problem
    T0 = problem.params.T0
    T = problem.params.T

    # The timestep must be given in the problem
    dt = problem.params.dt

    # Compute regular timesteps, not including t0
    timesteps = arange(T0, T+dt, dt)
    assert abs( (timesteps[1]-timesteps[0]) - dt) < 1e-8, "Timestep size does not match specified dt."
    assert timesteps[-1] >= T-dt/1e6, "Timestep range not including end time."

    return dt, timesteps, problem.params.start_timestep

def test_TimeDerivative(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    params = dict(finalize=True, start_time=start_time, end_time=end_time)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    pp.add_fields([
            TimeDerivative("t", params),
            TimeDerivative("timestep", params),
            TimeDerivative("MockFunctionField", params),
            TimeDerivative("MockVectorFunctionField", params),
            TimeDerivative("MockTupleField", params),
            ])

    # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
    pp.finalize_all()

    # Get and check values from the final timestep
    assert abs( (pp.get("TimeDerivative_t", compute=False)) - (1.0) ) < 1e-8
    assert abs( (pp.get("TimeDerivative_timestep", compute=False)) - (1.0/dt) ) < 1e-8
    assert errornorm(pp.get("TimeDerivative_MockFunctionField"), interpolate(Expression("x[0]*x[1]"), Q)) < 1e-8
    D = problem.D
    if D == 2:
        assert errornorm(pp.get("TimeDerivative_MockVectorFunctionField"), interpolate(Expression(("x[0]", "x[1]")), V)) < 1e-8
    elif D == 3:
        assert errornorm(pp.get("TimeDerivative_MockVectorFunctionField"), interpolate(Expression(("x[0]", "x[1]", "x[2]")), V)) < 1e-8

    assert max( [ abs(x1-x0) for x1,x0 in zip(pp.get("TimeDerivative_MockTupleField"), (1,3,5)) ] ) < 1e-8

def test_TimeIntegral(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    params = dict(finalize=True, start_time=start_time, end_time=end_time)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    pp.add_fields([
            TimeIntegral("t", params),
            TimeIntegral("MockFunctionField", params),
            TimeIntegral("MockVectorFunctionField", params),
            TimeIntegral("MockTupleField", params),
            ])

    # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)

    pp.finalize_all()

    assert abs( pp.get("TimeIntegral_t") - (0.5*(end_time**2-start_time**2)) ) < 1e-8
    assert errornorm(
        pp.get("TimeIntegral_MockFunctionField"),
        interpolate(Expression("t1-t0+0.5*x[0]*x[1]*(t1*t1-t0*t0)", t1=end_time, t0=start_time), Q)
        ) < 1e-8

    D = problem.D
    if D == 2:
        assert errornorm(
           pp.get("TimeIntegral_MockVectorFunctionField"),
           interpolate(Expression(("t1-t0+0.5*x[0]*(t1*t1-t0*t0)", "3*(t1-t0)+0.5*x[1]*(t1*t1-t0*t0)"), t1=end_time, t0=start_time), V)
        ) < 1e-8
    elif D == 3:
        assert errornorm(
           pp.get("TimeIntegral_MockVectorFunctionField"),
           interpolate(Expression(("t1-t0+0.5*x[0]*(t1*t1-t0*t0)", "3*(t1-t0)+0.5*x[1]*(t1*t1-t0*t0)", "10*(t1-t0)+0.5*x[2]*(t1*t1-t0*t0)"), t1=end_time, t0=start_time), V)
        ) < 1e-8

    I = 0.5*(end_time**2-start_time**2)
    assert max( [ abs(x1-x0) for x1,x0 in zip(pp.get("TimeIntegral_MockTupleField"), (I, 3*I, end_time-start_time+5*I) ) ] ) < 1e-8

def test_TimeAverage(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    params = dict(start_time=start_time, end_time=end_time, finalize=True)
    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    pp.add_fields([
            TimeAverage("t", params),
            TimeAverage("MockFunctionField", params),
            TimeAverage("MockVectorFunctionField", params),
            TimeAverage("MockTupleField", params),
            ])

    # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)

    pp.finalize_all()

    assert abs( pp.get("TimeAverage_t") - (0.5*(end_time**2-start_time**2))/(end_time-start_time) ) < 1e-8
    assert errornorm(
        pp.get("TimeAverage_MockFunctionField"),
        interpolate(Expression("1+0.5*x[0]*x[1]*(t1+t0)", t1=end_time, t0=start_time), Q)
        ) < 1e-8

    D = problem.D
    if D == 2:
        assert errornorm(
           pp.get("TimeAverage_MockVectorFunctionField"),
           interpolate(Expression(("1+0.5*x[0]*(t1+t0)", "3+0.5*x[1]*(t1+t0)"), t1=end_time, t0=start_time), V)
        ) < 1e-8
    elif D == 3:
        assert errornorm(
           pp.get("TimeAverage_MockVectorFunctionField"),
           interpolate(Expression(("1+0.5*x[0]*(t1+t0)", "3+0.5*x[1]*(t1+t0)", "10+0.5*x[2]*(t1+t0)"), t1=end_time, t0=start_time), V)
        ) < 1e-8

    I = (0.5*end_time*end_time-0.5*start_time*start_time)/(end_time-start_time)
    assert max( [ abs(x1-x0) for x1,x0 in zip(pp.get("TimeAverage_MockTupleField"), (I, 3*I, 1+5*I) ) ] ) < 1e-8

def test_TimeIntegral_of_TimeDerivative(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    pp.add_fields([
        TimeDerivative("t"),
        TimeDerivative("MockFunctionField"),
        TimeDerivative("MockVectorFunctionField"),
        TimeDerivative("MockTupleField"),
        ])

    params = dict(start_time=start_time, end_time=end_time, finalize=False)
    pp.add_fields([
        TimeIntegral("TimeDerivative_t", params),
        TimeIntegral("TimeDerivative_MockFunctionField", params),
        TimeIntegral("TimeDerivative_MockVectorFunctionField", params),
        TimeIntegral("TimeDerivative_MockTupleField", params),
    ])

    # Because of backward finite differencing in TimeDerivative, a numerical error will be introduced if start_time<dt
    err_factor = max(0.0, dt-start_time)

    err_t = err_factor*0.5*(1-start_time/dt)
    err_MockFunctionField = err_factor*norm(interpolate(Expression("0.5*x[0]*x[1]*(1-t0/dt)", dt=dt, t0=start_time), Q))

    D = problem.mesh.geometry().dim()
    if D == 2:
        err_MockVectorFunctionField = err_factor*norm(interpolate(Expression(("0.5*x[0]*(1-t0/dt)", "0.5*x[1]*(1-t0/dt)"), dt=dt, t0=start_time), V))
    elif D == 3:
        err_MockVectorFunctionField = err_factor*norm(interpolate(Expression(("0.5*x[0]*(1-t0/dt)", "0.5*x[1]*(1-t0/dt)", "0.5*x[2]*(1-t0/dt)"), dt=dt, t0=start_time), V))
    else:
        raise Exception("D must be 2 or 3")

    err_MockTupleField = tuple([err_factor*(1-start_time/dt)*x/2.0 for x in [1.0, 3.0, 5.0]])

    if start_time > dt:
        assert err_factor == 0.0

     # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
        if start_time < t < end_time:
            assert abs( pp.get("TimeIntegral_TimeDerivative_t") - ((t-start_time)-err_t) ) < 1e-8

    pp.finalize_all()

    assert err_t-1e-8 < abs( pp.get("TimeIntegral_TimeDerivative_t") - (end_time-start_time)) < err_t+1e-8
    assert err_MockFunctionField-1e-8 < \
            errornorm(pp.get("TimeIntegral_TimeDerivative_MockFunctionField"),
                    interpolate(Expression("x[0]*x[1]*(t1-t0)", t0=start_time, t1=end_time), Q)
                   ) < \
          err_MockFunctionField+1e-8

    if D == 2:
        assert err_MockVectorFunctionField-1e-8 < \
            errornorm(pp.get("TimeIntegral_TimeDerivative_MockVectorFunctionField"),
                    interpolate(Expression(("x[0]*(t1-t0)", "x[1]*(t1-t0)"), t0=start_time, t1=end_time), V)
                   ) < \
          err_MockVectorFunctionField+1e-8
    elif D == 3:
        assert err_MockVectorFunctionField-1e-8 < \
            errornorm(pp.get("TimeIntegral_TimeDerivative_MockVectorFunctionField"),
                    interpolate(Expression(("x[0]*(t1-t0)", "x[1]*(t1-t0)", "x[2]*(t1-t0)"), t0=start_time, t1=end_time), V)
                   ) < \
          err_MockVectorFunctionField+1e-8
    else:
        raise Exception("D must be 2 or 3")

    I = end_time-start_time
    assert abs( abs(pp.get("TimeIntegral_TimeDerivative_MockTupleField")[0] - I) - err_MockTupleField[0])  < 1e-8
    assert abs( abs(pp.get("TimeIntegral_TimeDerivative_MockTupleField")[1] - 3*I) - err_MockTupleField[1])  < 1e-8
    assert abs( abs(pp.get("TimeIntegral_TimeDerivative_MockTupleField")[2] - 5*I) - err_MockTupleField[2])  < 1e-8

def test_Maximum(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])


    pp.add_fields([
        Maximum("t"),
        Maximum("MockFunctionField"),
        Maximum("MockVectorFunctionField"),
        Maximum("MockTupleField"),
        ])

    xmax = MPI.max(mpi_comm_world(), max(problem.mesh.coordinates()[:,0]))
    ymax = MPI.max(mpi_comm_world(), max(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmax = MPI.max(mpi_comm_world(), max(problem.mesh.coordinates()[:,2]))

     # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
        if start_time < t < end_time:
            assert abs(pp.get("Maximum_t") - t) < 1e-8
            assert abs(pp.get("Maximum_MockFunctionField") - (1+xmax*ymax*t)) < 1e-8
            if problem.D == 2:
                assert abs(pp.get("Maximum_MockVectorFunctionField") - (3+ymax*t)) < 1e-8
            elif problem.D == 3:
                assert abs(pp.get("Maximum_MockVectorFunctionField") - (10+zmax*t)) < 1e-8
            assert abs(pp.get("Maximum_MockTupleField") - (1+5*t)) < 1e-8

    pp.finalize_all()

    assert abs(pp.get("Maximum_t") - t) < 1e-8
    assert abs(pp.get("Maximum_MockFunctionField") - (1+xmax*ymax*t)) < 1e-8
    if problem.D == 2:
        assert abs(pp.get("Maximum_MockVectorFunctionField") - (3+ymax*t)) < 1e-8
    elif problem.D == 3:
        assert abs(pp.get("Maximum_MockVectorFunctionField") - (10+zmax*t)) < 1e-8
    assert abs(pp.get("Maximum_MockTupleField") - (1+5*t)) < 1e-8

def test_Minimum(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    pp.add_fields([
        Minimum("t"),
        Minimum("MockFunctionField"),
        Minimum("MockVectorFunctionField"),
        Minimum("MockTupleField"),
        ])

    xmin = MPI.min(mpi_comm_world(), min(problem.mesh.coordinates()[:,0]))
    ymin = MPI.min(mpi_comm_world(), min(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmin = MPI.min(mpi_comm_world(), min(problem.mesh.coordinates()[:,2]))

     # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
        if start_time < t < end_time:
            assert abs(pp.get("Minimum_t") - t) < 1e-8
            assert abs(pp.get("Minimum_MockFunctionField") - (1+xmin*ymin*t)) < 1e-8
            assert abs(pp.get("Minimum_MockVectorFunctionField") - (1+xmin*t)) < 1e-8
            assert abs(pp.get("Minimum_MockTupleField") - t) < 1e-8

    pp.finalize_all()

    assert abs(pp.get("Minimum_t") - t) < 1e-8
    assert abs(pp.get("Minimum_MockFunctionField") - (1+xmin*ymin*t)) < 1e-8
    assert abs(pp.get("Minimum_MockVectorFunctionField") - (1+xmin*t)) < 1e-8
    assert abs(pp.get("Minimum_MockTupleField") - t) < 1e-8


def test_Norm(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    pp.add_fields([
        Norm("t"),
        Norm("t", dict(norm_type='l2')),
        Norm("t", dict(norm_type='linf')),
        Norm("MockFunctionField"),
        Norm("MockFunctionField", dict(norm_type='L2')),
        Norm("MockFunctionField", dict(norm_type='H10')),
        Norm("MockVectorFunctionField"),
        Norm("MockVectorFunctionField", dict(norm_type='L2')),
        Norm("MockVectorFunctionField", dict(norm_type='H10')),
        Norm("MockTupleField"),
        Norm("MockTupleField", dict(norm_type='l4')),
        Norm("MockTupleField", dict(norm_type='linf')),
        ])

    D = problem.mesh.geometry().dim()

    # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)

        if start_time < t < end_time:
            assert abs(pp.get("Norm_t") - t) < 1e-14
            assert abs(pp.get("Norm_l2_t") - t) < 1e-14
            assert abs(pp.get("Norm_linf_t") - t) < 1e-14
            assert abs(pp.get("Norm_MockFunctionField") - norm(interpolate(Expression("1+x[0]*x[1]*t", t=t), Q))) < 1e-14
            if D == 2:
                assert abs(pp.get("Norm_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V))) < 1e-14
                assert abs(pp.get("Norm_L2_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V), 'L2')) < 1e-14
                assert abs(pp.get("Norm_H10_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V), 'H10')) < 1e-14
            elif D == 3:
                assert abs(pp.get("Norm_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V))) < 1e-14
                assert abs(pp.get("Norm_L2_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V), 'L2')) < 1e-14
                assert abs(pp.get("Norm_H10_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V), 'H10')) < 1e-14
            assert abs(pp.get("Norm_MockTupleField") - sqrt(t**2+(3*t)**2+(1+5*t)**2)) < 1e-14
            assert abs(pp.get("Norm_l4_MockTupleField") - (t**4+(3*t)**4+(1+5*t)**4)**(0.25)) < 1e-14
            assert abs(pp.get("Norm_linf_MockTupleField") - (1+5*t)) < 1e-14

    pp.finalize_all()

    assert abs(pp.get("Norm_t") - t) < 1e-14
    assert abs(pp.get("Norm_l2_t") - t) < 1e-14
    assert abs(pp.get("Norm_linf_t") - t) < 1e-14
    assert abs(pp.get("Norm_MockFunctionField") - norm(interpolate(Expression("1+x[0]*x[1]*t", t=t), Q))) < 1e-14
    if D == 2:
        assert abs(pp.get("Norm_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V))) < 1e-14
        assert abs(pp.get("Norm_L2_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V), 'L2')) < 1e-14
        assert abs(pp.get("Norm_H10_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V), 'H10')) < 1e-14
    elif D == 3:
        assert abs(pp.get("Norm_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V))) < 1e-14
        assert abs(pp.get("Norm_L2_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V), 'L2')) < 1e-14
        assert abs(pp.get("Norm_H10_MockVectorFunctionField") - norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V), 'H10')) < 1e-14
    assert abs(pp.get("Norm_MockTupleField") - sqrt(t**2+(3*t)**2+(1+5*t)**2)) < 1e-14
    assert abs(pp.get("Norm_l4_MockTupleField") - (t**4+(3*t)**4+(1+5*t)**4)**(0.25)) < 1e-14
    assert abs(pp.get("Norm_linf_MockTupleField") - (1+5*t)) < 1e-14

@require_fenicstools
def test_PointEval(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(2,0)
    V = spacepool.get_space(2,1)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    D = problem.mesh.geometry().dim()

    xmin = MPI.min(mpi_comm_world(), min(problem.mesh.coordinates()[:,0]))
    ymin = MPI.min(mpi_comm_world(), min(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmin = MPI.min(mpi_comm_world(), min(problem.mesh.coordinates()[:,2]))

    xmax = MPI.max(mpi_comm_world(), max(problem.mesh.coordinates()[:,0]))
    ymax = MPI.max(mpi_comm_world(), max(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmax = MPI.max(mpi_comm_world(), max(problem.mesh.coordinates()[:,2]))

    points = []
    #for i in range(5):
    bbtree = problem.mesh.bounding_box_tree()
    while len(points) < 5:
        p = [xmin+(xmax-xmin)*random(), ymin+(ymax-ymin)*random()]
        if D == 3:
            p += [zmin+(zmax-xmin)*random()]

        points.append(p)

    pp.add_fields([
        PointEval("MockFunctionField", points),
        PointEval("MockVectorFunctionField", points),
        ])

    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)

        if start_time < t < end_time:
            pevalfunction = pp.get("PointEval_MockFunctionField")
            pevalvectorfunction = pp.get("PointEval_MockVectorFunctionField")
            for i, p in enumerate(points):
                if D == 2:
                    x,y = p
                    assert abs( pevalfunction[i] - (1+x*y*t) ) < 1e-10
                    assert sum( abs( pevalvectorfunction[i][k] - (1+x*t, 3+y*t)[k]) for k in range(D)) < 1e-10
                elif D == 3:
                    x,y,z = p
                    assert abs( pevalfunction[i] - (1+x*y*t) ) < 1e-10
                    assert sum( abs( pevalvectorfunction[i][k] - (1+x*t, 3+y*t, 10+z*t)[k]) for k in range(D)) < 1e-10


    pp.finalize_all()

    pevalfunction = pp.get("PointEval_MockFunctionField")
    pevalvectorfunction = pp.get("PointEval_MockVectorFunctionField")
    for i, p in enumerate(points):
        if D == 2:
            x,y = p
            assert abs( pevalfunction[i] - (1+x*y*t) ) < 1e-10
            assert sum( abs( pevalvectorfunction[i][k] - (1+x*t, 3+y*t)[k]) for k in range(D)) < 1e-10
        elif D == 3:
            x,y,z = p
            assert abs( pevalfunction[i] - (1+x*y*t) ) < 1e-10
            assert sum( abs( pevalvectorfunction[i][k] - (1+x*t, 3+y*t, 10+z*t)[k]) for k in range(D)) < 1e-10

def test_ErrorNorm(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    class MockFunctionField2(MockFunctionField):
        def before_first_compute(self, get):
            t = get('t')
            self.expr = Expression("1+x[0]*x[1]*t+0.4", t=t)

    class MockVectorFunctionField2(MockVectorFunctionField):
        def before_first_compute(self, get):
            t = get('t')

            D = self.f.function_space().mesh().geometry().dim()
            if D == 2:
                self.expr = Expression(("1+x[0]*t+0.2", "3+x[1]*t+0.3"), t=t)
            elif D == 3:
                self.expr = Expression(("1+x[0]*t+0.2", "3+x[1]*t+0.3", "10+x[2]*t+0.4"), t=t)

    class MockTupleField2(MockTupleField):
        def compute(self, get):
            return (t+0.2, 3*t+0.3, 1+5*t+0.4)

    class MockScalarField2(MockScalarField):
        def compute(self, get):
            t = get('t')
            return 3*t**0.5+0.3

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
        MockScalarField(),
    ])

    pp.add_fields([
        MockFunctionField2(Q),
        MockVectorFunctionField2(V),
        MockTupleField2(),
        MockScalarField2(),
    ])

    pp.add_fields([
        ErrorNorm("MockFunctionField", "MockFunctionField2"),
        ErrorNorm("MockVectorFunctionField", "MockVectorFunctionField2"),
        ErrorNorm("MockTupleField", "MockTupleField2"),
        ErrorNorm("MockScalarField", "MockScalarField2"),
    ])

    D = problem.D
    # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
        if start_time < t < end_time:
            assert abs(pp.get("ErrorNorm_MockFunctionField_MockFunctionField2") - \
                       norm(interpolate(Expression("0.4"), Q))) < 1e-8

            assert abs(pp.get("ErrorNorm_MockVectorFunctionField_MockVectorFunctionField2") - \
                       norm(interpolate(Expression(["0.2", "0.3", "0.4"][:D]), V))) < 1e-8

            assert abs(pp.get("ErrorNorm_MockTupleField_MockTupleField2") - \
                       sum([0.2**2, 0.3**2, 0.4**2])**0.5) < 1e-8

            assert abs(pp.get("ErrorNorm_MockScalarField_MockScalarField2")-0.3) < 1e-8

    pp.finalize_all()

    assert abs(pp.get("ErrorNorm_MockFunctionField_MockFunctionField2") - \
                       norm(interpolate(Expression("0.4"), Q))) < 1e-8

    assert abs(pp.get("ErrorNorm_MockVectorFunctionField_MockVectorFunctionField2") - \
               norm(interpolate(Expression(["0.2", "0.3", "0.4"][:D]), V))) < 1e-8

    assert abs(pp.get("ErrorNorm_MockTupleField_MockTupleField2") - \
               sum([0.2**2, 0.3**2, 0.4**2])**0.5) < 1e-8

    assert abs(pp.get("ErrorNorm_MockScalarField_MockScalarField2")-0.3) < 1e-8

def test_Boundary(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    bmesh = BoundaryMesh(problem.mesh, "exterior")
    bspacepool = SpacePool(bmesh)
    Qb = bspacepool.get_space(1,0)
    Vb = bspacepool.get_space(1,1)

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
    ])

    pp.add_fields([
        Boundary("MockFunctionField"),
        Boundary("MockVectorFunctionField"),
        ])
    D = problem.D

     # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
        if start_time < t < end_time:
            assert errornorm(
                    pp.get("Boundary_MockFunctionField"),
                    interpolate(Expression("1+x[0]*x[1]*t", t=t), Qb)
            ) < 1e-8

            if D == 2:
                assert errornorm(
                    pp.get("Boundary_MockVectorFunctionField"),
                    interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), Vb)
                ) < 1e-8
            else:
                assert errornorm(
                    pp.get("Boundary_MockVectorFunctionField"),
                    interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), Vb)
                ) < 1e-8

    pp.finalize_all()
    assert errornorm(
             pp.get("Boundary_MockFunctionField"),
             interpolate(Expression("1+x[0]*x[1]*t", t=t), Qb)
        ) < 1e-8

    if D == 2:
        assert errornorm(
            pp.get("Boundary_MockVectorFunctionField"),
            interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), Vb)
        ) < 1e-8
    else:
        assert errornorm(
            pp.get("Boundary_MockVectorFunctionField"),
            interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), Vb)
        ) < 1e-8

def test_DomainAvg_DomainSD(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    mesh = problem.mesh
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)
    V = spacepool.get_space(2,1)
    D = V.num_sub_spaces()

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])


    measures = []
    facet_domains = FacetFunction("size_t", problem.mesh)
    facet_domains.set_all(0)
    cell_domains = CellFunction("size_t", problem.mesh)
    cell_domains.set_all(0)

    subdomains = AutoSubDomain(lambda x: x[0]<0.5)
    subdomains.mark(facet_domains, 1)
    subdomains.mark(cell_domains, 1)

    measures = [dict(),
                dict(measure=ds),
                dict(cell_domains=cell_domains, indicator=0),
                dict(cell_domains=cell_domains, indicator=1),
                dict(facet_domains=facet_domains, indicator=0),
                dict(facet_domains=facet_domains, indicator=1),]

    for m in measures:
        pp.add_field(DomainAvg("MockFunctionField", **m))
        pp.add_field(DomainAvg("MockVectorFunctionField", **m))
        pp.add_field(DomainSD("MockFunctionField", **m))
        pp.add_field(DomainSD("MockVectorFunctionField", **m))


    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)

        v = assemble(Constant(1)*dx(domain=mesh))
        v_dx0 = assemble(Constant(1)*dx(0, domain=mesh, subdomain_data=cell_domains))
        v_dx1 = assemble(Constant(1)*dx(1, domain=mesh, subdomain_data=cell_domains))
        v_ds = assemble(Constant(1)*ds(domain=mesh))
        v_ds0 = assemble(Constant(1)*ds(0, domain=mesh, subdomain_data=facet_domains))
        v_ds1 = assemble(Constant(1)*ds(1, domain=mesh, subdomain_data=facet_domains))

        u = pp.get("MockFunctionField")
        uv = pp.get("MockVectorFunctionField")

        avg = assemble(u*dx)/v
        avg_dx0 = assemble(u*dx(0, subdomain_data=cell_domains))/v_dx0
        avg_dx1 = assemble(u*dx(1, subdomain_data=cell_domains))/v_dx1

        avg_ds = assemble(u*ds)/v_ds
        avg_ds0 = assemble(u*ds(0, subdomain_data=facet_domains))/v_ds0
        avg_ds1 = assemble(u*ds(1, subdomain_data=facet_domains))/v_ds1
        
        assert abs(pp.get("DomainAvg_MockFunctionField") - avg) < 1e-8
        assert abs(pp.get("DomainAvg_MockFunctionField-dx0") - avg_dx0) < 1e-8
        assert abs(pp.get("DomainAvg_MockFunctionField-dx1") - avg_dx1) < 1e-8
        assert abs(pp.get("DomainAvg_MockFunctionField-ds") - avg_ds) < 1e-8
        assert abs(pp.get("DomainAvg_MockFunctionField-ds0") - avg_ds0) < 1e-8
        assert abs(pp.get("DomainAvg_MockFunctionField-ds1") - avg_ds1) < 1e-8
        
        std = sqrt(assemble((u-Constant(avg))**2*dx)/v)
        std_dx0 = sqrt(assemble((u-Constant(avg_dx0))**2*dx(0, subdomain_data=cell_domains))/v_dx0)
        std_dx1 = sqrt(assemble((u-Constant(avg_dx1))**2*dx(1, subdomain_data=cell_domains))/v_dx1)
        
        std_ds = sqrt(assemble((u-Constant(avg_ds))**2*ds)/v_ds)
        std_ds0 = sqrt(assemble((u-Constant(avg_ds0))**2*ds(0, subdomain_data=facet_domains))/v_ds0)
        std_ds1 = sqrt(assemble((u-Constant(avg_ds1))**2*ds(1, subdomain_data=facet_domains))/v_ds1)
        
        assert abs(pp.get("DomainSD_MockFunctionField") - std) < 1e-8
        assert abs(pp.get("DomainSD_MockFunctionField-dx0") - std_dx0) < 1e-8
        assert abs(pp.get("DomainSD_MockFunctionField-dx1") - std_dx1) < 1e-8
        assert abs(pp.get("DomainSD_MockFunctionField-ds") - std_ds) < 1e-8
        assert abs(pp.get("DomainSD_MockFunctionField-ds0") - std_ds0) < 1e-8
        assert abs(pp.get("DomainSD_MockFunctionField-ds1") - std_ds1) < 1e-8

        avg = [assemble(uv[i]*dx)/v for i in xrange(D)]
        avg_dx0 = [assemble(uv[i]*dx(0, subdomain_data=cell_domains))/v_dx0 for i in xrange(D)]
        avg_dx1 = [assemble(uv[i]*dx(1, subdomain_data=cell_domains))/v_dx1 for i in xrange(D)]

        avg_ds = [assemble(uv[i]*ds)/v_ds for i in xrange(D)]
        avg_ds0 = [assemble(uv[i]*ds(0, subdomain_data=facet_domains))/v_ds0 for i in xrange(D)]
        avg_ds1 = [assemble(uv[i]*ds(1, subdomain_data=facet_domains))/v_ds1 for i in xrange(D)]

        assert max(abs(x-y) for x,y in zip(avg, pp.get("DomainAvg_MockVectorFunctionField"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(avg_dx0, pp.get("DomainAvg_MockVectorFunctionField-dx0"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(avg_dx1, pp.get("DomainAvg_MockVectorFunctionField-dx1"))) < 1e-8

        assert max(abs(x-y) for x,y in zip(avg_ds, pp.get("DomainAvg_MockVectorFunctionField-ds"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(avg_ds0, pp.get("DomainAvg_MockVectorFunctionField-ds0"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(avg_ds1, pp.get("DomainAvg_MockVectorFunctionField-ds1"))) < 1e-8
        
        std = [sqrt(assemble((uv[i]-Constant(avg[i]))**2*dx)/v) for i in xrange(D)]
        std_dx0 = [sqrt(assemble((uv[i]-Constant(avg_dx0[i]))**2*dx(0, subdomain_data=cell_domains))/v_dx0) for i in xrange(D)]
        std_dx1 = [sqrt(assemble((uv[i]-Constant(avg_dx1[i]))**2*dx(1, subdomain_data=cell_domains))/v_dx1) for i in xrange(D)]
        
        std_ds = [sqrt(assemble((uv[i]-Constant(avg_ds[i]))**2*ds)/v_ds) for i in xrange(D)]
        std_ds0 = [sqrt(assemble((uv[i]-Constant(avg_ds0[i]))**2*ds(0, subdomain_data=facet_domains))/v_ds0) for i in xrange(D)]
        std_ds1 = [sqrt(assemble((uv[i]-Constant(avg_ds1[i]))**2*ds(1, subdomain_data=facet_domains))/v_ds1) for i in xrange(D)]
        
        assert max(abs(x-y) for x,y in zip(std, pp.get("DomainSD_MockVectorFunctionField"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(std_dx0, pp.get("DomainSD_MockVectorFunctionField-dx0"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(std_dx1, pp.get("DomainSD_MockVectorFunctionField-dx1"))) < 1e-8

        assert max(abs(x-y) for x,y in zip(std_ds, pp.get("DomainSD_MockVectorFunctionField-ds"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(std_ds0, pp.get("DomainSD_MockVectorFunctionField-ds0"))) < 1e-8
        assert max(abs(x-y) for x,y in zip(std_ds1, pp.get("DomainSD_MockVectorFunctionField-ds1"))) < 1e-8
        

def test_Restrict(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    mesh = problem.mesh
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)
    V = spacepool.get_space(2,1)
    D = V.num_sub_spaces()

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
    ])

    measures = []
    cell_domains = CellFunction("size_t", problem.mesh)
    cell_domains.set_all(0)
    subdomains = AutoSubDomain(lambda x: x[0]<0.5)
    subdomains.mark(cell_domains, 1)
    MPI.barrier(mpi_comm_world())
    submesh = create_submesh(mesh, cell_domains, 1)

    pp.add_fields([
        Restrict("MockFunctionField", submesh),
        Restrict("MockVectorFunctionField", submesh),
    ])


    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)

        assert abs(assemble(pp.get("MockFunctionField")*dx(1, subdomain_data=cell_domains)) - \
                   assemble(pp.get("Restrict_MockFunctionField")*dx)) < 1e-8

        uv = pp.get("MockVectorFunctionField")
        uvr = pp.get("Restrict_MockVectorFunctionField")
        assert abs(assemble(inner(uv,uv)*dx(1, subdomain_data=cell_domains)) - assemble(inner(uvr, uvr)*dx)) < 1e-8

@require_fenicstools14
def test_SubFunction(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    mesh = problem.mesh
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)
    V = spacepool.get_space(2,1)

    mff = MockFunctionField(Q)
    mvff = MockVectorFunctionField(V)

    pp.add_fields([mff, mvff])

    D = mesh.geometry().dim()

    if D == 3:
        submesh = UnitCubeMesh(6,6,6)
    elif D == 2:
        submesh = UnitSquareMesh(8,8)
    submesh.coordinates()[:] /= 2.0
    submesh.coordinates()[:] += 0.2

    Q_sub = FunctionSpace(submesh, "CG", 2)
    V_sub = VectorFunctionSpace(submesh, "CG", 2)

    pp.add_fields([
        SubFunction("MockFunctionField", submesh),
        SubFunction("MockVectorFunctionField", submesh),
    ])


    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
        assert abs(norm(interpolate(mff.expr, Q_sub)) - norm(pp.get("SubFunction_MockFunctionField"))) < 1e-8
        assert abs(norm(interpolate(mvff.expr, V_sub)) - norm(pp.get("SubFunction_MockVectorFunctionField"))) < 1e-8

def test_Magnitude(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    mesh = problem.mesh
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)
    
    D = mesh.geometry().dim()
    
    f = Function(Q)
    fv = Function(Q)
    
    mff = MockFunctionField(Q)
    mvff = MockVectorFunctionField(V)

    pp.add_fields([mff, mvff])

    pp.add_fields([
        Magnitude("MockFunctionField"),
        Magnitude("MockVectorFunctionField"),
    ])
    
    if D == 2:
        vec_expr_mag = Expression("sqrt(pow(1+x[0]*t,2)+pow(3+x[1]*t,2))", t=0.0)
    elif D == 3:
        vec_expr_mag = Expression("sqrt(pow(1+x[0]*t,2)+pow(3+x[1]*t,2)+pow(10+x[2]*t,2))", t=0.0)
    
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)

        expr_mag = mff.expr
        f.interpolate(expr_mag)
        f.vector().abs()
        vec_expr_mag.t = t
        fv.interpolate(vec_expr_mag)

        assert norm(f.vector()-pp.get("Magnitude_MockFunctionField").vector())<1e-12
        assert norm(fv.vector()-pp.get("Magnitude_MockVectorFunctionField").vector())<1e-12


def test_Operators(problem, pp, start_time, end_time, dt):
    # Setup some mock scheme state
    dt, timesteps, start_timestep = compute_regular_timesteps(problem)
    spacepool = SpacePool(problem.mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)
    
    D = problem.mesh.geometry().dim()


    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
        MockTupleField(),
    ])

    fields = [
        Norm("MockFunctionField"),
        Norm("MockFunctionField", dict(norm_type='L2')),
        Norm("MockFunctionField", dict(norm_type='H10')),
        Norm("MockVectorFunctionField"),
        Norm("MockVectorFunctionField", dict(norm_type='L2')),
        Norm("MockVectorFunctionField", dict(norm_type='H10')),
        ]
    pp.add_fields(fields)
    
    exact = dict()
    exact["Norm_MockFunctionField"] = lambda t: norm(interpolate(Expression("1+x[0]*x[1]*t", t=t), Q))
    exact["Norm_L2_MockFunctionField"] = lambda t: norm(interpolate(Expression("1+x[0]*x[1]*t", t=t), Q), 'L2')
    exact["Norm_H10_MockFunctionField"] = lambda t: norm(interpolate(Expression("1+x[0]*x[1]*t", t=t), Q), 'H10')
    if D == 2:
        exact["Norm_MockVectorFunctionField"] = lambda t: norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V))
        exact["Norm_L2_MockVectorFunctionField"] = lambda t: norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V), 'L2')
        exact["Norm_H10_MockVectorFunctionField"] = lambda t: norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t"), t=t), V), 'H10')
    elif D == 3:
        exact["Norm_MockVectorFunctionField"] = lambda t: norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V))
        exact["Norm_L2_MockVectorFunctionField"] = lambda t: norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V), 'L2')
        exact["Norm_H10_MockVectorFunctionField"] = lambda t: norm(interpolate(Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t"), t=t), V), 'H10')

    for f1 in fields:
        print f1.name
        pp.add_field(f1+2.0)
        pp.add_field(2.0+f1)
        pp.add_field(f1-2.0)
        pp.add_field(2.0-f1)
        pp.add_field(f1*2.0)
        pp.add_field(2.0*f1)
        pp.add_field(f1/2.0)
        pp.add_field(2.0/f1)
        for f2 in fields:
            pp.add_field(f1+f2)
            pp.add_field(f1*f2)
            pp.add_field(f1-f2)
            pp.add_field(f1/f2)

    # Update postprocessor for a number of timesteps, this is where the main code under test is
    for timestep, t in enumerate(timesteps, start_timestep):
        # Run postprocessing step
        pp.update_all({}, t, timestep)
        """
        # Skip these time consuming tests
        if start_time < t < end_time:
            for f1 in fields:
                E1 = exact[f1.name](t)
                assert abs(pp.get("Add_%s_2.0" %f1.name) - (2.0+E1)) < 1e-14
                assert abs(pp.get("Add_2.0_%s" %f1.name) - (2.0+E1)) < 1e-14
                assert abs(pp.get("Subtract_%s_2.0" %f1.name) - (E1-2.0)) < 1e-14
                assert abs(pp.get("Subtract_2.0_%s" %f1.name) - (2.0-E1)) < 1e-14
                assert abs(pp.get("Multiply_%s_2.0" %f1.name) - (2.0*E1)) < 1e-14
                assert abs(pp.get("Multiply_2.0_%s" %f1.name) - (2.0*E1)) < 1e-14
                assert abs(pp.get("Divide_%s_2.0" %f1.name) - (E1/2.0)) < 1e-14
                if abs(E1) > 1e-14:
                    assert abs(pp.get("Divide_2.0_%s" %f1.name) - (2.0/E1)) < 1e-14
                
                for f2 in fields:
                    E2 = exact[f2.name](t)
                    assert abs(pp.get("Add_%s_%s" %(f1.name, f2.name))-(E1+E2)) < 1e-14
                    assert abs(pp.get("Multiply_%s_%s" %(f1.name, f2.name))-(E1*E2)) < 1e-14
                    assert abs(pp.get("Subtract_%s_%s" %(f1.name, f2.name))-(E1-E2)) < 1e-14
                    if abs(E2) > 1e-14:
                        assert abs(pp.get("Divide_%s_%s" %(f1.name, f2.name))-(E1/E2)) < 1e-14
        """
    pp.finalize_all()
    for f1 in fields:
        E1 = exact[f1.name](t)
        assert abs(pp.get("Add_%s_2.0" %f1.name) - (2.0+E1)) < 1e-14
        assert abs(pp.get("Add_2.0_%s" %f1.name) - (2.0+E1)) < 1e-14
        assert abs(pp.get("Subtract_%s_2.0" %f1.name) - (E1-2.0)) < 1e-14
        assert abs(pp.get("Subtract_2.0_%s" %f1.name) - (2.0-E1)) < 1e-14
        assert abs(pp.get("Multiply_%s_2.0" %f1.name) - (2.0*E1)) < 1e-14
        assert abs(pp.get("Multiply_2.0_%s" %f1.name) - (2.0*E1)) < 1e-14
        assert abs(pp.get("Divide_%s_2.0" %f1.name) - (E1/2.0)) < 1e-14
        if abs(E1) > 1e-14:
            assert abs(pp.get("Divide_2.0_%s" %f1.name) - (2.0/E1)) < 1e-14
        
        for f2 in fields:
            E2 = exact[f2.name](t)
            assert abs(pp.get("Add_%s_%s" %(f1.name, f2.name))-(E1+E2)) < 1e-14
            assert abs(pp.get("Multiply_%s_%s" %(f1.name, f2.name))-(E1*E2)) < 1e-14
            assert abs(pp.get("Subtract_%s_%s" %(f1.name, f2.name))-(E1-E2)) < 1e-14
            if abs(E2) > 1e-14:
                assert abs(pp.get("Divide_%s_%s" %(f1.name, f2.name))-(E1/E2)) < 1e-14
