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
from dolfin import MPI, MPI_Comm, compile_extension_module
import numpy as np

def broadcast(array, from_process):
    "Broadcast array to all processes"
    cpp_code = '''
    
    namespace dolfin {
        std::vector<double> broadcast(const Array<double>& inarray, int from_process)
        {
            int this_process = dolfin::MPI::process_number();
    
            std::vector<double> outvector(inarray.size());
    
            if(this_process == from_process) {
                for(int i=0; i<inarray.size(); i++)
                {
                    outvector[i] = inarray[i];
                }
            }
            dolfin::MPI::barrier();
            
            dolfin::MPI::broadcast(outvector, from_process);
            
            return outvector;
        }
    }
    '''
    array = np.array(array, dtype=np.float)
    cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])
    out_array = cpp_module.broadcast(array, from_process)
    return out_array

def distribution(number):
    "Get distribution of number on all processes"
    cpp_code = '''
    namespace dolfin {
        std::vector<unsigned int> distribution(int number)
        {
            // Variables to help in synchronization
            int num_processes = dolfin::MPI::num_processes();
            int this_process = dolfin::MPI::process_number();
            
            std::vector<uint> distribution(num_processes);
        
            for(uint i=0; i<num_processes; i++) {
                if(i==this_process) {
                    distribution[i] = number;
                }
                dolfin::MPI::barrier();
                dolfin::MPI::broadcast(distribution, i);    
            }
            return distribution;
      }
    }
    '''
    
    cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])
    return cpp_module.distribution(number)


def gather(array, on_process=0, flatten=False):
    "Gather array from all processes on a single process"
    cpp_code = '''
    namespace dolfin {
        std::vector<double> gather(const Array<double>& inarray, int on_process)
        {
            int this_process = dolfin::MPI::process_number();
    
            std::vector< std::vector<double> > outvector(dolfin::MPI::num_processes());
            std::vector<double> invector(inarray.size());
            
            for(int i=0; i<inarray.size(); i++)
            {
                invector[i] = inarray[i];
            }

            dolfin::MPI::gather(invector, outvector, on_process);

            std::vector<double> flat_outvector;
            for(int i=0; i<dolfin::MPI::num_processes(); i++)
            {
                for(int j=0; j<outvector[i].size(); j++)
                {
                    flat_outvector.push_back(outvector[i][j]);
                    
                }
            }
            return flat_outvector;
        }
    }
    '''
    
    cpp_module = compile_extension_module(cpp_code, additional_system_headers=["dolfin/common/MPI.h"])
    array = np.array(array, dtype=np.float)
    out_array = cpp_module.gather(array, on_process)

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

    x_per_v = int(MPI.max(x_per_v))
    v_per_cell = int(MPI.max(v_per_cell))
    
    # Move a single cell to process with no cells
    while 0 in global_cell_distribution:
        to_process = list(global_cell_distribution).index(0)
        from_process = list(global_cell_distribution).index(max(global_cell_distribution))
        
        # Extract vertices and remove cells[0] on from_process
        v_out = np.zeros((1+x_per_v)*v_per_cell)
        if MPI.rank(MPI_Comm()) == from_process:
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
        MPI.barrier(MPI_Comm())
        # Broadcast vertices in cell[0] on from_process
        v_in = broadcast(v_out, from_process)
        MPI.barrier(MPI_Comm())
        # Create cell and vertices on to_process
        if MPI.rank(MPI_Comm()) == to_process:
            for i in xrange(v_per_cell):
                vertices[i] = (int(v_in[i*(x_per_v+1)]), v_in[i*(x_per_v+1)+1:(i+1)*(x_per_v+1)])

            assert len(cells) == 0
            cells = [range(v_per_cell)]

        MPI.barrier(MPI_Comm())
        
        # Update distribution
        global_cell_distribution = distribution(len(cells))

    return cells, vertices