#!/usr/bin/env py.test

from dolfin import *

"""
Restarting a problem
-----------------------------------------
If we want to restart any problem, where a solution has been stored by cbcpost, we can simply point to the
case directory: ::

    from cbcpost import *
    restart = Restart(dict(casedir='Results/'))
    restart_data = restart.get_restart_conditions()

If you for instance try to restart the simple case of the heat equation, *restart_data* will be a *dict* of
the format {t0: {"Temperature": U0}}. If you try to restart for example a (Navier-)Stokes-problem, it will take
a format of {t0: {"Velocity": U0, "Pressure": P0}}.

There are several options for fetching the restart conditions.

Specify restart time
#########################################

You can easily specify the restart time to fetch the solution from: ::

    t0 = 2.5
    restart = Restart(dict(casedir='Results/', restart_times=t0))
    restart_data = restart.get_restart_conditions()

If the restart time does not match a solution time, it will do a linear interpolation between the closest
existing solution times.

Fetch multiple restart times
#########################################

For many problems (for example the wave equation), initial conditions are required at several time points
prior to the desired restart time. This can also be handled through: ::

    dt = 0.01
    t1 = 2.5
    t0 = t1-dt
    restart = Restart(dict(casedir='Results/', restart_times=[t0,t1]))
    restart_data = restart.get_restart_conditions()


Rollback case directory for restart
#########################################

If you wish to write the restarted solution to the same case directory, you will need to clean up the case
directory to avoid write errors. This is done by setting the parameter *rollback_casedir*: ::

    t0 = 2.5
    restart = Restart(dict(casedir='Results/', restart_times=t0, rollback_casedir=True))
    restart_data = restart.get_restart_conditions()

Specifying solution names to fetch
#########################################

By default, the Restart-module will search through the case directory for all data stored as a
:class:`SolutionField`. However, you can also specify other fields to fetch as restart data: ::

    solution_names = ["MyField", "MyField2"]
    restart = Restart(dict(casedir='Results/', solution_names=solution_names))
    restart_data = restart.get_restart_conditions()

In this case, all :class:`SolutionField`-names will be ignored, and only restart conditions from fields
named *MyField* and *MyField2* will be returned.


Changing function spaces
#########################################

If you wish to restart the simulation using different function spaces, you can pass the function spaces
to *get_restart_conditions*: ::

    V = FunctionSpace(mesh, "CG", 3)
    restart = Restart(dict(casedir='Results/'))
    restart_data = restart.get_restart_conditions(spaces={"Temperature": V})

.. todo:: Make this work for different meshes as well.
"""

import pytest

from cbcpost import *
from dolfin import *
import os, shelve, glob
from conftest import MockFunctionField

@pytest.fixture(scope="function")
def filled_casedir(mesh, casedir):
    # Setup some mock scheme state
    pp = PostProcessor(dict(casedir=casedir))

    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)
    V = spacepool.get_space(2,1)

    pp.add_fields([
            SolutionField("MockSolutionFunctionField", dict(save=True, save_as=["xml", "xml.gz", "hdf5", "pvd", "xdmf"])),
            SolutionField("MockSolutionVectorFunctionField", dict(save=True, save_as=["xml", "xml.gz", "hdf5", "pvd", "xdmf"])),
            SolutionField("MockSolutionTupleField", dict(save=True, save_as=["txt", "shelve"])),
            SolutionField("MockSolutionScalarField", dict(save=True, save_as=["txt", "shelve"])),
        ])


    pp.add_fields([
        MockFunctionField(Q, params=dict(save=True, save_as="xml"),label="xml"),
        MockFunctionField(Q, params=dict(save=True, save_as="xml.gz"),label="xmlgz"),
        MockFunctionField(Q, params=dict(save=True, save_as="hdf5"),label="hdf5"),
    ])

    D = mesh.geometry().dim()
    expr_scalar = Expression("1+x[0]*x[1]*t", t=0.0)
    expr = Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t")[:D], t=0.0)

    t = 0.0; timestep=0; expr.t = t; expr_scalar.t=t;
    pp.update_all({
        "MockSolutionFunctionField": lambda: interpolate(expr_scalar, Q),
        "MockSolutionVectorFunctionField": lambda: interpolate(expr, V),
        "MockSolutionTupleField": lambda: (t, 3*t, 1+5*t),
        "MockSolutionScalarField": lambda: 3*t,
    }, t, timestep)


    t = 0.25; timestep=1; expr.t = t; expr_scalar.t=t;
    pp.update_all({
        "MockSolutionFunctionField": lambda: interpolate(expr_scalar, Q),
        "MockSolutionVectorFunctionField": lambda: interpolate(expr, V),
        "MockSolutionTupleField": lambda: (t, 3*t, 1+5*t),
        "MockSolutionScalarField": lambda: 3*t,
    }, t, timestep)

    t = 0.5; timestep=2; expr.t = t; expr_scalar.t=t;
    pp.update_all({
        "MockSolutionFunctionField": lambda: interpolate(expr_scalar, Q),
        "MockSolutionVectorFunctionField": lambda: interpolate(expr, V),
        "MockSolutionTupleField": lambda: (t, 3*t, 1+5*t),
        "MockSolutionScalarField": lambda: 3*t,
    }, t, timestep)


    t = 0.75; timestep=3; expr.t = t; expr_scalar.t=t;
    pp.update_all({
        "MockSolutionFunctionField": lambda: interpolate(expr_scalar, Q),
        "MockSolutionVectorFunctionField": lambda: interpolate(expr, V),
        "MockSolutionTupleField": lambda: (t, 3*t, 1+5*t),
        "MockSolutionScalarField": lambda: 3*t,
    }, t, timestep)

    t = 1.0; timestep=4; expr.t = t; expr_scalar.t=t;
    pp.update_all({
        "MockSolutionFunctionField": lambda: interpolate(expr_scalar, Q),
        "MockSolutionVectorFunctionField": lambda: interpolate(expr, V),
        "MockSolutionTupleField": lambda: (t, 3*t, 1+5*t),
        "MockSolutionScalarField": lambda: 3*t,
    }, t, timestep)

    return casedir

@pytest.fixture(scope="module", params=[1.0, 0.75, 0.86])
def t(request):
    return request.param

def test_restart_from_solutionfield(filled_casedir, mesh, t):
    if t == 1.0:
        restart = Restart(dict(casedir=filled_casedir))
    else:
        restart = Restart(dict(casedir=filled_casedir, restart_times=t))
    data = restart.get_restart_conditions()

    assert t in data.keys()
    assert set(data[t].keys()) == set(["MockSolutionFunctionField", "MockSolutionVectorFunctionField",
                                         "MockSolutionTupleField", "MockSolutionScalarField"])

    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)
    V = spacepool.get_space(2,1)

    D = mesh.geometry().dim()
    expr_scalar = Expression("1+x[0]*x[1]*t", t=t)
    expr = Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t")[:D], t=t)

    assert abs(norm(data[t]["MockSolutionFunctionField"]) - norm(interpolate(expr_scalar, Q))) < 1e-8
    assert abs(norm(data[t]["MockSolutionVectorFunctionField"]) - norm(interpolate(expr, V))) < 1e-8
    assert max(abs(x-y) for x,y in zip(data[t]["MockSolutionTupleField"], (t, 3*t, 1+5*t))) < 1e-8
    assert abs(data[t]["MockSolutionScalarField"] - 3*t) < 1e-8


def test_restart_from_xml(filled_casedir, mesh, t):
    if t == 1.0:
        restart = Restart(dict(casedir=filled_casedir, solution_names="MockFunctionField-xml"))
    else:
        restart = Restart(dict(casedir=filled_casedir, restart_times=t, solution_names="MockFunctionField-xml"))

    data = restart.get_restart_conditions()
    assert t in data.keys()

    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)

    D = mesh.geometry().dim()
    expr_scalar = Expression("1+x[0]*x[1]*t", t=t)

    assert abs(norm(data[t]["MockFunctionField-xml"]) - norm(interpolate(expr_scalar, Q))) < 1e-8

def test_restart_from_xmlgz(filled_casedir, mesh, t):
    if t == 1.0:
        restart = Restart(dict(casedir=filled_casedir, solution_names="MockFunctionField-xmlgz"))
    else:
        restart = Restart(dict(casedir=filled_casedir, restart_times=t, solution_names="MockFunctionField-xmlgz"))

    data = restart.get_restart_conditions()
    assert t in data.keys()

    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)

    D = mesh.geometry().dim()
    expr_scalar = Expression("1+x[0]*x[1]*t", t=t)

    assert abs(norm(data[t]["MockFunctionField-xmlgz"]) - norm(interpolate(expr_scalar, Q))) < 1e-8

def test_restart_from_hdf5(filled_casedir, mesh, t):
    if t == 1.0:
        restart = Restart(dict(casedir=filled_casedir, solution_names="MockFunctionField-hdf5"))
    else:
        restart = Restart(dict(casedir=filled_casedir, restart_times=t, solution_names="MockFunctionField-hdf5"))

    data = restart.get_restart_conditions()
    assert t in data.keys()

    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)

    expr_scalar = Expression("1+x[0]*x[1]*t", t=t)

    assert abs(norm(data[t]["MockFunctionField-hdf5"]) - norm(interpolate(expr_scalar, Q))) < 1e-8

@pytest.mark.skipif(MPI.size(mpi_comm_world()) != 1, reason="Currently not supported in parallel")
def test_restart_change_function_space(filled_casedir, mesh):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(1,0)

    t = 1.0
    expr_scalar = Expression("1+x[0]*x[1]*t", t=t)
    restart = Restart(dict(casedir=filled_casedir, solution_names="MockFunctionField-hdf5"))

    data = restart.get_restart_conditions(function_spaces={"MockFunctionField-hdf5": Q})

    assert abs(norm(data[t]["MockFunctionField-hdf5"]) - norm(interpolate(expr_scalar, Q))) < 1e-8

def test_rollback_casedir(filled_casedir, mesh, t):
    spacepool = SpacePool(mesh)
    Q = spacepool.get_space(2,0)
    V = spacepool.get_space(2,1)

    D = mesh.geometry().dim()
    expr_scalar = Expression("1+x[0]*x[1]*t", t=t)
    expr = Expression(("1+x[0]*t", "3+x[1]*t", "10+x[2]*t")[:D], t=t)

    expr_scalar = Expression("1+x[0]*x[1]*t", t=t)

    restart = Restart(dict(casedir=filled_casedir, rollback_casedir=True, restart_times=t))
    assert os.path.isfile(os.path.join(filled_casedir, "play.db"))
    #print os.path.join(filled_casedir, "play.db")
    data = restart.get_restart_conditions()
    #assert os.path.isfile(os.path.join(filled_casedir, "play.db"))
    #print os.path.isfile("test_saver/play.db")
    #return
    playlog = shelve.open(os.path.join(filled_casedir, "play.db"), 'r')

    assert max([v["t"] for v in playlog.values()]) < t
    assert max([v["t"] for v in playlog.values()]) > t - 0.25 - 1e-14

    for d in os.listdir(filled_casedir):
        if not os.path.isdir(os.path.join(filled_casedir, d)):
            continue

        assert os.path.isfile(os.path.join(filled_casedir, d, "metadata.db"))
        md = shelve.open(os.path.join(filled_casedir, d, "metadata.db"), 'r')

        savetimes = {"shelve": [], "txt": [], "xml": [], "xml.gz": [], "hdf5": [], "pvd": [], "xdmf": []}
        assert max([v.get("time", -1) for v in md.values() if isinstance(v, dict)]) < t
        for k in md:
            try:
                int(k)
            except:
                continue

            for sf in set(md[k].keys()).intersection(savetimes.keys()):
                savetimes[sf].append(k)

        for sf, st in savetimes.items():
            if st == []:
                continue
            if sf in ["xml", "xml.gz"]:
                xmlfiles = glob.glob1(os.path.join(filled_casedir, d),"*."+sf)
                assert sorted(xmlfiles) == sorted([d+i+"."+sf for i in st])
            elif sf == "shelve":
                data = shelve.open(os.path.join(filled_casedir, d, d+".db"))
                assert sorted(data.keys()) == sorted(st)
            elif sf == "txt":
                data = open(os.path.join(filled_casedir, d, d+".txt"), 'r').readlines()
                assert len(data) == len(st)
            elif sf == "pvd":
                pass
            elif sf == "xdmf":
                xdmffiles = glob.glob1(os.path.join(filled_casedir, d),"*_RS0.xdmf")
                assert xdmffiles == [d+"_RS0.xdmf"]
                h5files = glob.glob1(os.path.join(filled_casedir, d),"*_RS0.h5")
                assert h5files == [d+"_RS0.h5"]
            elif sf == "hdf5":
                filename = os.path.join(filled_casedir, d, d+".hdf5")
                assert os.path.isfile(filename)
                #datasets = [u''+d+i for i in st]+[u'Mesh']
                datasets = [d+i for i in st]+['Mesh']

                cpp_code = """
                #include <hdf5.h>
                std::size_t size(MPI_Comm comm,
                          const std::string hdf5_filename,
                          bool use_mpiio)
                {
                    hid_t hdf5_file_id = HDF5Interface::open_file(comm, hdf5_filename, "r", use_mpiio);
                    std::size_t num_datasets = HDF5Interface::num_datasets_in_group(hdf5_file_id, "/");
                    HDF5Interface::close_file(hdf5_file_id);
                    return num_datasets;
                    //herr_t status = H5Lcreate_hard(hdf5_file_id, link_from.c_str(), H5L_SAME_LOC,
                    //                    link_to.c_str(), H5P_DEFAULT, H5P_DEFAULT);

                    //dolfin_assert(status != HDF5_FAIL);

                }
                """

                cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/io/HDF5Interface.h"])
                num_datasets = cpp_module.size(mpi_comm_world(), filename, MPI.size(mpi_comm_world()) > 1)
                assert num_datasets == len(st)+2

                f = HDF5File(mpi_comm_world(), filename, 'r')
                for ds in datasets:
                    assert f.has_dataset(ds)
                del f
                return
