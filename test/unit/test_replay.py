#!/usr/bin/env py.test

from cbcpost import PostProcessor, SpacePool, ParamDict, Norm, SolutionField, Replay, TimeIntegral
from conftest import MockFunctionField, MockVectorFunctionField, MockTupleField, MockScalarField
import os, shelve, time


def test_basic_replay(mesh, casedir):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)
    V = spacepool.get_space(1,1)

    pp = PostProcessor(dict(casedir=casedir))
    pp.add_fields([
        MockFunctionField(Q, dict(save=True)),
        MockVectorFunctionField(V, dict(save=True))
    ])

    replay_fields = lambda save: [Norm("MockFunctionField", dict(save=save)),
                             Norm("MockVectorFunctionField", dict(save=save)),
                             TimeIntegral("Norm_MockFunctionField", dict(save=save)),]
    rf_names = [f.name for f in replay_fields(False)]

    # Add fields, but don't save (for testing)
    pp.add_fields(replay_fields(False))

    # Solutions to check against
    checks = {}
    pp.update_all({}, 0.0, 0)
    checks[0] = dict([(name, pp.get(name)) for name in rf_names])
    pp.update_all({}, 0.1, 1)
    checks[1] = dict([(name, pp.get(name)) for name in rf_names])
    pp.update_all({}, 0.2, 2)
    pp.finalize_all()
    checks[2] = dict([(name, pp.get(name)) for name in rf_names])

    # Make sure that nothing is saved yet
    for name in rf_names:
        assert not os.path.isfile(os.path.join(pp.get_savedir(name), name+".db"))


    # ----------- Replay -----------------
    pp = PostProcessor(dict(casedir=casedir))

    pp.add_fields([
        MockFunctionField(Q),
        MockVectorFunctionField(V),
    ])

    # This time, save the fields
    pp.add_fields(replay_fields(True))

    replayer = Replay(pp)
    replayer.replay()
    
    # Test that replayed solution is the same as computed in the original "solve"
    for name in rf_names:
        
        time.sleep(1)
        print name
        data = shelve.open(os.path.join(pp.get_savedir(name), name+".db"), 'r')

        for i in range(3):
            assert data.get(str(i), None) == checks[i][name] or abs(data.get(str(i), None) - checks[i][name]) < 1e-8
        data.close()

