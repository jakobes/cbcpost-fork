from numpy.random import random, seed, randint
#from random import random, seed, randint
from sys import maxsize
from dolfin import MPI, mpi_comm_world

def pytest_addoption(parser):
    parser.addoption("--all", action="store_true",
        help="run all combinations")


def pytest_generate_tests(metafunc):
    if 'dim' in metafunc.fixturenames:
        metafunc.parametrize("dim", [2, 3])

    # Set random seed
    new_seed = MPI.sum(mpi_comm_world(), randint(0,1e6))/MPI.size(mpi_comm_world())
    seed(new_seed)
    
    # TODO: Make options to select all or subset of schemes for this factory,
    #       copy from or look at regression conftest,
    if 'scheme_factory' in metafunc.fixturenames:
        metafunc.parametrize("scheme_factory", create_scheme_factories())
        
    if 'D' in metafunc.fixturenames:
        metafunc.parametrize("D", [2,3])
    
    if 'start_time' in metafunc.fixturenames:
        start_times = [0.0]
        if metafunc.config.option.all:
            start_times += list(0.8*random(3))
        metafunc.parametrize("start_time", start_times)
        
    if 'end_time' in metafunc.fixturenames:
        end_times = [2.0]
        if metafunc.config.option.all:
            end_times += list(1.2+0.8*random(3))
        metafunc.parametrize("end_time", end_times)
        
    if 'dt' in metafunc.fixturenames:
        dts = [0.1]
        if metafunc.config.option.all:
            dts += [0.05+0.05*random(), 0.2+0.2*random()]
        metafunc.parametrize("dt", dts)
