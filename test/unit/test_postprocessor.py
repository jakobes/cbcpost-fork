#!/usr/bin/env py.test

from cbcpost import PostProcessor, ParamDict
from cbcpost import SolutionField, Field, MetaField, MetaField2, Norm
import os, pickle
from conftest import MockField, MockPressure, MockStrain, MockStress, MockVelocity, MockVelocityGradient

"""
cbcpost.PostProcessor.add_field       cbcpost.PostProcessor.get_playlog
cbcpost.PostProcessor.add_fields      cbcpost.PostProcessor.get_savedir
cbcpost.PostProcessor.default_params  cbcpost.PostProcessor.mro
cbcpost.PostProcessor.description     cbcpost.PostProcessor.shortname
cbcpost.PostProcessor.finalize_all    cbcpost.PostProcessor.store_mesh
cbcpost.PostProcessor.get             cbcpost.PostProcessor.store_params
cbcpost.PostProcessor.get_casedir     cbcpost.PostProcessor.update_all
"""


def test_add_field():
    pp = PostProcessor()

    pp.add_field(SolutionField("foo"))
    assert "foo" in pp._fields.keys()

    pp += SolutionField("bar")
    assert set(["foo", "bar"]) == set(pp._fields.keys())

    pp += [SolutionField("a"), SolutionField("b")]
    assert set(["foo", "bar", "a", "b"]) == set(pp._fields.keys())

    pp.add_fields([
        MetaField("foo"),
        MetaField2("foo", "bar"),
    ])

    assert set(["foo", "bar", "a", "b", "MetaField_foo", "MetaField2_foo_bar"]) == set(pp._fields.keys())

def test_finalize_all(casedir):
    pp = PostProcessor(dict(casedir=casedir))

    velocity = MockVelocity(dict(finalize=True))
    pressure = MockPressure()
    pp.add_fields([velocity, pressure])

    pp.get("MockVelocity")
    pp.get("MockPressure")

    # Nothing finalized yet
    assert pp._finalized == {}
    assert velocity.finalized is False

    # finalize_all should finalize velocity only
    pp.finalize_all()
    assert pp._finalized == {"MockVelocity": "u"}
    assert velocity.finalized is True

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




