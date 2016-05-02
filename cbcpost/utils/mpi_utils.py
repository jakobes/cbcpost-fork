# Copyright (C) 2010-2014 Simula Research Laboratory
#
# This file is part of CBCPOST.
#
# CBCPOST is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CBCPOST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CBCPOST. If not, see <http://www.gnu.org/licenses/>.
"""Utilities for providing a very simple interface to functions in dolfin.MPI
that are not exposed in the python interface of dolfin, in addition to some
communcation used internally.
"""

from dolfin import MPI, mpi_comm_world, compile_extension_module
import numpy as np

def broadcast(array, from_process):
    "Broadcast array to all processes"
    if not hasattr(broadcast, "cpp_module"):
        cpp_code = '''

        namespace dolfin {
            std::vector<double> broadcast(const MPI_Comm mpi_comm, const Array<double>& inarray, int from_process)
            {
                int this_process = dolfin::MPI::rank(mpi_comm);
                std::vector<double> outvector(inarray.size());

                if(this_process == from_process) {
                    for(int i=0; i<inarray.size(); i++)
                    {
                        outvector[i] = inarray[i];
                    }
                }
                dolfin::MPI::barrier(mpi_comm);
                dolfin::MPI::broadcast(mpi_comm, outvector, from_process);

                return outvector;
            }
        }
        '''
        cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])

        broadcast.cpp_module = cpp_module

    cpp_module = broadcast.cpp_module

    if MPI.rank(mpi_comm_world()) == from_process:
        array = np.array(array, dtype=np.float)
        shape = array.shape
        shape = np.array(shape, dtype=np.float_)
    else:
        array = np.array([], dtype=np.float)
        shape = np.array([], dtype=np.float_)

    shape = cpp_module.broadcast(mpi_comm_world(), shape, from_process)
    array = array.flatten()

    out_array = cpp_module.broadcast(mpi_comm_world(), array, from_process)
    if len(shape) != 0:
        out_array = out_array.reshape(*shape)

    return out_array

def distribution(number):
    "Get distribution of number on all processes"
    if not hasattr(distribution, "cpp_module"):
        cpp_code = '''
        namespace dolfin {
            std::vector<unsigned int> distribution(const MPI_Comm mpi_comm, int number)
            {
                // Variables to help in synchronization
                int num_processes = dolfin::MPI::size(mpi_comm);
                int this_process = dolfin::MPI::rank(mpi_comm);

                std::vector<uint> distribution(num_processes);

                for(uint i=0; i<num_processes; i++) {
                    if(i==this_process) {
                        distribution[i] = number;
                    }
                    dolfin::MPI::barrier(mpi_comm);
                    dolfin::MPI::broadcast(mpi_comm, distribution, i);
                }
                return distribution;
          }
        }
        '''
        distribution.cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])

    cpp_module = distribution.cpp_module
    return cpp_module.distribution(mpi_comm_world(), number)


def gather(array, on_process=0, flatten=False):
    "Gather array from all processes on a single process"
    if not hasattr(gather, "cpp_module"):
        cpp_code = '''
        namespace dolfin {
            std::vector<double> gather(const MPI_Comm mpi_comm, const Array<double>& inarray, int on_process)
            {
                int this_process = dolfin::MPI::rank(mpi_comm);

                std::vector<double> outvector(dolfin::MPI::size(mpi_comm)*dolfin::MPI::sum(mpi_comm, inarray.size()));
                std::vector<double> invector(inarray.size());

                for(int i=0; i<inarray.size(); i++)
                {
                    invector[i] = inarray[i];
                }

                dolfin::MPI::gather(mpi_comm, invector, outvector, on_process);
                return outvector;
            }
        }
        '''
        gather.cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])

    cpp_module = gather.cpp_module
    array = np.array(array, dtype=np.float)
    out_array = cpp_module.gather(mpi_comm_world(), array, on_process)

    if flatten:
        return out_array

    dist = distribution(len(array))
    cumsum = [0]+[sum(dist[:i+1]) for i in range(len(dist))]
    out_array = [[out_array[cumsum[i]:cumsum[i+1]]] for i in range(len(cumsum)-1)]

    return out_array


def distribute_meshdata(cells, vertices):
    """Because dolfin does not support a distributed mesh that is empty on some processes,
    we move a single cell from the process with the largest mesh to all processes with
    empty meshes."""
    global_cell_distribution = distribution(len(cells))

    x_per_v = 0
    v_per_cell = 0
    if len(vertices.values()) > 0:
        x_per_v = len(vertices.values()[0][1])
        v_per_cell = len(cells[0])

    x_per_v = int(MPI.max(mpi_comm_world(), x_per_v))
    v_per_cell = int(MPI.max(mpi_comm_world(), v_per_cell))

    # Move a single cell to process with no cells
    while 0 in global_cell_distribution:
        to_process = list(global_cell_distribution).index(0)
        from_process = list(global_cell_distribution).index(max(global_cell_distribution))

        # Extract vertices and remove cells[0] on from_process
        v_out = np.zeros((1+x_per_v)*v_per_cell)
        if MPI.rank(mpi_comm_world()) == from_process:
            # Structure v_out as (ind0, x0, y0, .., ind1, x1, .., )
            for i, v in enumerate(cells[0]):
                v_out[i*(x_per_v+1)] = vertices[v][0]
                v_out[i*(x_per_v+1)+1:(i+1)*(x_per_v+1)] = vertices[v][1]

            # Remove vertices no longer used in remaining cells.
            for i,v in enumerate(cells[0]):
                if not any([v in c for c in cells[1:]]):
                    for j in xrange(len(cells)):
                        cells[j] = [vi-1 if vi > v else vi for vi in cells[j]]

                    for vi in range(v, max(vertices)):
                        vertices[vi] = vertices[vi+1]
                    vertices.pop(max(vertices))

            cells.pop(0)
        MPI.barrier(mpi_comm_world())
        # Broadcast vertices in cell[0] on from_process
        v_in = broadcast(v_out, from_process)
        MPI.barrier(mpi_comm_world())
        # Create cell and vertices on to_process
        if MPI.rank(mpi_comm_world()) == to_process:
            for i in xrange(v_per_cell):
                vertices[i] = (int(v_in[i*(x_per_v+1)]), v_in[i*(x_per_v+1)+1:(i+1)*(x_per_v+1)])

            assert len(cells) == 0
            cells = [range(v_per_cell)]

        MPI.barrier(mpi_comm_world())

        # Update distribution
        global_cell_distribution = distribution(len(cells))

    return cells, vertices