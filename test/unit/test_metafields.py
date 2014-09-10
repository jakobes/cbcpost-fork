#!/usr/bin/env py.test
"""
Tests of postprocessing framework in cbcflow.
"""

from collections import defaultdict

#from cbcflow import (ParamDict, NSProblem, NSPostProcessor, NSScheme,
#    Field, Velocity, Pressure, VelocityGradient, Strain, Stress, WSS,
#    TimeDerivative, SecondTimeDerivative, TimeIntegral)
#from cbcflow.fields import *
#from cbcflow.post import *
from cbcpost import *
from cbcpost.utils import cbc_warning

#from cbcflow.core.parameterized import Parameterized
#from cbcflow.core.paramdict import ParamDict

#from cbcflow.utils.core import NSSpacePoolSplit
#from cbcflow.utils.schemes import compute_regular_timesteps

import pytest

import dolfin
dolfin.set_log_level(40)
#from dolfin import (UnitSquareMesh, Function, Expression, norm, errornorm, assemble, dx,
#                    interpolate, plot)
from dolfin import *
from math import sqrt
from numpy.random import random
from numpy import linspace

# Avoid negative norms caused by instable tensor representation:
dolfin.parameters["form_compiler"]["representation"] = "quadrature"
dolfin.parameters["form_compiler"]["quadrature_degree"] = 1

class MockProblem(Parameterized):
    def __init__(self, D, params=None):
        Parameterized.__init__(self, params)
        self.D = D
        if D == 2:
            self.mesh = UnitSquareMesh(6,6)
        elif D == 3:
            self.mesh = UnitCubeMesh(3,3,3)
        elif D == 'complex':
            self.mesh = Mesh("../cbcflow-data/dog_mesh_37k.xml.gz")
        #self.initialize_geometry(mesh)

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

@pytest.fixture(scope="module")
def spacepool(problem):
    #return NSSpacePoolSplit(problem.mesh, 1, 1)
    return SpacePool(problem.mesh)

@pytest.fixture(scope="module")
def spacepool2(problem):
    return SpacePool(problem.mesh)

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
    
    #if timesteps[-1] > T+dt/1e6:
    #    cbc_warning("End time for simulation does not match end time set for problem (T-T0 not a multiple of dt).")
    
    return dt, timesteps, problem.params.start_timestep

class MockFunctionField(Field):
    def __init__(self, Q, params=None):
        Field.__init__(self, params)
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

def test_TimeAverage(problem, spacepool, pp, start_time, end_time, dt):
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
    
    xmax = MPI.max(max(problem.mesh.coordinates()[:,0]))
    ymax = MPI.max(max(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmax = MPI.max(max(problem.mesh.coordinates()[:,2]))
    
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
    
    xmin = MPI.min(min(problem.mesh.coordinates()[:,0]))
    ymin = MPI.min(min(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmin = MPI.min(min(problem.mesh.coordinates()[:,2]))
    
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
    
    xmin = MPI.min(min(problem.mesh.coordinates()[:,0]))
    ymin = MPI.min(min(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmin = MPI.min(min(problem.mesh.coordinates()[:,2]))
        
    xmax = MPI.max(max(problem.mesh.coordinates()[:,0]))
    ymax = MPI.max(max(problem.mesh.coordinates()[:,1]))
    if problem.D > 2:
        zmax = MPI.max(max(problem.mesh.coordinates()[:,2]))
        
    points = []
    #for i in range(5):
    bbtree = problem.mesh.bounding_box_tree()
    while len(points) < 5:
        p = [xmin+(xmax-xmin)*random(), ymin+(ymax-ymin)*random()]
        if D == 3:
            p += [zmin+(zmax-xmin)*random()]
        _,d = bbtree.compute_closest_entity(Point(*p))

        if d < 1e-12:
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

