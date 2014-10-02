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
"""Smaller utilities used across cbcpost."""

import keyword
import os
from time import time
from dolfin import compile_extension_module, MPI, mpi_comm_world, log, warning

def import_fenicstools():
    "Import fenicstools helper function. TODO: Move."
    import fenicstools
    return fenicstools

def on_master_process():
    """Return True if on process number 0."""
    return MPI.rank(mpi_comm_world()) == 0

def in_serial():
    """Return True if running in serial."""
    return MPI.size(mpi_comm_world()) == 1

def strip_code(code):
    """Strips code of unnecessary spaces, comments etc."""
    s = []

    code = code.split('\n')
    for i, l in enumerate(reversed(code)):
        if l.count(')') > l.count('('):
            code[-2-i] += " "+l.strip(' ')
            code[-1-i] = ''

    for l in code:
        l = l.split('#')[0]
        l = l.replace('\t', '    ')
        l = l.replace('    ', ' ')

        l = l.rstrip(' ')        
        l = l.split(' ')
        
        l_new = ""
        string_flag = False
        for c in l:
            if c.count('"') == 1 or c.count("'") == 1:
                if string_flag:
                    l_new += " "+c
                else:
                    l_new += c
                string_flag = not string_flag
            elif string_flag:
                l_new += " "+c
            elif c == '':
                l_new += ' '
            elif c in keyword.kwlist:
                if l_new[-1] == " ":
                    l_new += c+" "
                else:
                    l_new += " "+c+" "
            else:
                l_new += c

        if l_new.strip(' ') != '':
            s.append(l_new)

    s = '\n'.join(s)
    return s

# --- I/O stuff ---
class _HDF5Link:
    """Helper class for creating links in HDF5-files."""
    cpp_link_module = None
    def __init__(self):
        cpp_link_code = '''
        #include <hdf5.h>
        void link_dataset(MPI_Comm comm,
                          const std::string hdf5_filename,
                          const std::string link_from,
                          const std::string link_to, bool use_mpiio)
        {
            hid_t hdf5_file_id = HDF5Interface::open_file(comm, hdf5_filename, "a", use_mpiio);
            herr_t status = H5Lcreate_hard(hdf5_file_id, link_from.c_str(), H5L_SAME_LOC,
                                link_to.c_str(), H5P_DEFAULT, H5P_DEFAULT);
            dolfin_assert(status != HDF5_FAIL);
            
            HDF5Interface::close_file(hdf5_file_id);
        }
        '''
        
        self.cpp_link_module = compile_extension_module(cpp_link_code, additional_system_headers=["dolfin/io/HDF5Interface.h"])
    
    def link(self, hdf5filename, link_from, link_to):
        "Create link in hdf5file."
        use_mpiio = MPI.size(mpi_comm_world()) > 1
        self.cpp_link_module.link_dataset(mpi_comm_world(), hdf5filename, link_from, link_to, use_mpiio)
hdf5_link = _HDF5Link().link


def safe_mkdir(dir):
    """Create directory without exceptions in parallel."""
    # Create directory
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
        except:
            # Allow race condition when multiple processes
            # work in same directory, ignore exception.
            pass

    # Wait for all processes to finish, hopefully somebody
    # managed to create the directory...
    MPI.barrier(mpi_comm_world())

    # Warn if this failed
    if not os.path.isdir(dir):
        #warning("FAILED TO CREATE DIRECTORY %s" % (dir,))
        Exception("FAILED TO CREATE DIRECTORY %s" % (dir,))


loadable_formats = ["hdf5", "xml", "xml.gz", "shelve"]
from dolfin import HDF5File, Function
import shelve
class Loadable():
    """Create an instance that reads a Field from file as specified by the
    parameters. Requires that the file is written in cbcpost (or in the same format).
    
    :param filename: Filename where function is stored
    :param fieldname: Name of Field
    :param timestep: Timestep to load
    :param time: Time
    :param saveformat: Saveformat of field
    :params function: Function to load Field into
    
    This class is used internally from :class:'.Replay' and :class:'Restart',
    and made to be passed to *PostProcessor.update_all*.
    """
    def __init__(self, filename, fieldname, timestep, time, saveformat, function):
        self.filename = filename
        self.fieldname = fieldname
        self.timestep = timestep
        self.time = time
        self.saveformat = saveformat
        self.function = function
        
        assert self.saveformat in loadable_formats
        
    def __call__(self):
        """Load file"""
        cbc_log(20, "Loading: "+self.filename+", Timestep: "+str(self.timestep))
        if self.saveformat == 'hdf5':
            hdf5file = HDF5File(mpi_comm_world(), self.filename, 'r')
            hdf5file.read(self.function, self.fieldname+str(self.timestep))
            del hdf5file
            data = self.function
        elif self.saveformat in ["xml", "xml.gz"]:
            V = self.function.function_space()
            self.function.assign(Function(V, self.filename))
            data = self.function
        elif self.saveformat == "shelve":
            shelvefile = shelve.open(self.filename)
            data = shelvefile[str(self.timestep)]
        
        cbc_log(20, "Loaded: "+self.filename+", Timestep: "+str(self.timestep))
        return data


from dolfin import Mesh, HDF5File, Function
from cbcpost import SpacePool
def create_function_from_metadata(pp, fieldname, metadata, saveformat):
    "Create a function from metadata"
    assert metadata['type'] == 'Function'

    # Load mesh
    if saveformat == 'hdf5':    
        mesh = Mesh()
        hdf5file = HDF5File(mpi_comm_world(), os.path.join(pp.get_savedir(fieldname), fieldname+'.hdf5'), 'r')
        hdf5file.read(mesh, "Mesh", False)
        del hdf5file
    elif saveformat == 'xml' or saveformat == 'xml.gz':
        mesh = Mesh()
        hdf5file = HDF5File(mpi_comm_world(), os.path.join(pp.get_savedir(fieldname), "mesh.hdf5"), 'r')
        hdf5file.read(mesh, "Mesh", False)
        del hdf5file

    shape = eval(metadata["element_value_shape"])
    degree = eval(metadata["element_degree"])
    family = eval(metadata["element_family"])
    
    # Get space from existing function spaces if mesh is the same
    spaces = SpacePool(mesh)
    space = spaces.get_custom_space(family, degree, shape)

    return Function(space, name=fieldname)



# --- Logging ---

def cbc_warning(msg):
    "Raise warning on master process."
    if on_master_process():
        warning(msg)

def cbc_print(msg):
    "Print on master process."
    if on_master_process():
        print msg

def cbc_log(level, msg):
    "Log on master process."
    if on_master_process():
        log(level, msg)


# --- System inspection ---

from os import getpid
from commands import getoutput
def get_memory_usage():
    """Return memory usage in MB"""
    try:
        from fenicstools import getMemoryUsage
        return getMemoryUsage()
    except:
        cbc_warning("Unable to load fenicstools to check memory usage. Falling back to unsafe memory check.")
        mypid = getpid()
        mymemory = getoutput("ps -o rss %s" % mypid).split()[1]
        return int(mymemory)/1024

# --- Timing ---

class Timer:
    """Class to perform timing.
    
    :param frequency: Frequency which to report timings.
    
    """
    def __init__(self, frequency=0):
        self._frequency = frequency
        self._timer = time()
        self._timings = {}
        self._keys = []
        self._N = 0

    def completed(self, key, summables={}):
        "Called when *key* is completed."
        if self._frequency == 0:
            return
        
        if key not in self._timings:
            self._keys.append(key)
            self._timings[key] = [0,0, {}]
        
        t = time()
        ms = (t - self._timer)*1000
        self._timings[key][0] += ms
        self._timings[key][1] += 1
        
        for k,v in summables.items():
            if k not in self._timings[key][2]:
                self._timings[key][2][k] = 0
            self._timings[key][2][k] += v
        
        if self._frequency == 1:
            s = "%10.0f ms: %s" % (ms, key)
            ss = []
            #if summables != {}:
            #    s += "  ("
            for k, v in summables.items():
                #s += "%s: %s, " %(k,v)
                ss.append("%s=%s" %(k,v))
            if len(ss) > 0:
                ss = "  ("+", ".join(ss)+")"
            else:
                ss = ""
            s += ss

            cbc_print(s)

        self._timer = time()
    
    def _print_summary(self):
        "Print a summary of timing"
        cbc_print("Timings summary: ")
        
        for key in self._keys:
            tot = self._timings[key][0]
            N = self._timings[key][1]
            avg = int(1.0*tot/N)
            
            s = "%10.0f ms (avg: %8.0f ms, N: %5d): %s" %(tot, avg, N, key)
            
            summables = self._timings[key][2]
            ss = []
            #if summables != {}:
            #    s += "("
            for k, tot in summables.items():
                avg = int(1.0*tot/N)
                #s += "%s: %s (avg: %s), " %(k,tot,avg)
                ss.append("%s=%s (avg: %s)" %(k,tot,avg))
            #if summables != {}:
            #    s += ")"
            if len(ss) > 0:
                ss = "  ("+", ".join(ss)+")"
            else:
                ss = ""
            s += ss
            cbc_print(s)
    
    def _reset(self):
        "Reset timings"
        self._timings = {}
        self._keys = []
        self._N = 0
        
    def increment(self):
        "Increment timer"
        self._N += 1
        if self._frequency > 1 and self._N % self._frequency == 0:
            self._print_summary()
            self._reset()

def timeit(t0=None, msg=None):
    "Simple timer"
    if t0 is None:
        return time()
    else:
        t = time() - t0
        cbc_print("%s: %g" % (msg, t))
        return t


