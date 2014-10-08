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
"""
The restart module let's users fetch restart conditions to use as initial conditions
in a restarted simulation.
"""
from cbcpost import Parameterized, ParamDict, PostProcessor, SpacePool
from cbcpost.utils import (cbc_log, Loadable, loadable_formats,
                           create_function_from_metadata, cbc_warning, on_master_process)


import os, shelve, subprocess
from collections import Iterable, defaultdict
from numpy import array, where, inf

from dolfin import Mesh, Function, HDF5File, tic, toc, norm, project, interpolate, compile_extension_module, MPI, parameters, mpi_comm_world
from commands import getstatusoutput

def find_solution_presence(pp, play_log, fields):
    "Search play-log to find where solution items are saved in a loadable format"
    present_solution = defaultdict(list)

    functions = dict()
    metadatas = dict()
    for ts, data in play_log.items():
        for fieldname in data.get("fields", []):
            # Continue if field in a format we can't read back
            if not any([saveformat in loadable_formats for saveformat in data["fields"][fieldname]["save_as"]]):
                continue
            
            # Check if field is present and part of solution we're searching for
            is_present = False
            if fields == "default" and data["fields"][fieldname]["type"] == "SolutionField":
                is_present = True
            elif fieldname in fields:
                is_present = True

            metadata = metadatas.setdefault(fieldname, shelve.open(os.path.join(pp.get_savedir(fieldname), "metadata.db"), 'r'))
            if is_present:
                function = None
                if 'hdf5' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+'.hdf5')

                    if fieldname in functions: function = functions[fieldname]
                    else: function = functions.setdefault(fieldname, create_function_from_metadata(pp, fieldname, metadata, 'hdf5'))

                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], 'hdf5', function))
                elif 'xml' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+str(ts)+'.xml')
                    
                    if fieldname in functions: function = functions[fieldname]
                    else: function = functions.setdefault(fieldname, create_function_from_metadata(pp, fieldname, metadata, 'xml'))
                    
                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], 'xml', function))
                elif 'xml.gz' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+str(ts)+'.xml.gz')
                    
                    if fieldname in functions: function = functions[fieldname]
                    else: function = functions.setdefault(fieldname, create_function_from_metadata(pp, fieldname, metadata, 'xml.gz'))

                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], 'xml.gz', function))
                elif 'shelve' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+'.db')
                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], "shelve", None))

    return present_solution


def find_restart_items(restart_times, present_solution):
    "Find which items should be used for computing restart data"
    if not isinstance(restart_times, Iterable):
        restart_times = [restart_times]

    restart_times = [r if r > 0 else inf for r in restart_times]
    #for f, present_timesteps in present_solution.items():
    loadables = dict()
    for restart_time in sorted(restart_times):
        loadables[restart_time] = dict()
        for fieldname in present_solution:
            sorted_ps = sorted(present_solution[fieldname], key=lambda l: l.time)
            present_times = array([l.time for l in sorted_ps])
            
            # Find lower and upper limit
            limits = []
            lower = where(present_times <= restart_time)[0]
            if len(lower) > 0: limits.append((present_times[lower[-1]], sorted_ps[lower[-1]]))
            upper = where(present_times >= restart_time)[0]
            if len(upper) > 0 and upper[0] != lower[-1]: limits.append((present_times[upper[0]], sorted_ps[upper[0]]))
            
            loadables[restart_time][fieldname] = limits

    for k, v in loadables.items():

        if k == inf:
            loadables.pop(k)
            new_k = v.values()[0][0][0]
            loadables[new_k] = v

    return loadables

class Restart(Parameterized):
    """Class to fetch restart conditions through."""
    #def __init__(self, params=None):
    #    Parameterized.__init__(self, params)
    
    @classmethod
    def default_params(cls):
        """
        Default parameters are:
        
        +----------------------+-----------------------+--------------------------------------------------------------+
        |Key                   | Default value         |  Description                                                 |
        +======================+=======================+==============================================================+
        | casedir              | '.'                   | Case directory - relative path to read solutions from        |
        +----------------------+-----------------------+--------------------------------------------------------------+
        | restart_times        | -1                    | float or list of floats to find restart times from. If -1,   |
        |                      |                       | restart from last available time.                            |
        +----------------------+-----------------------+--------------------------------------------------------------+
        | solution_names       | 'default'             | Solution names to look for. If 'default', will fetch all     |
        |                      |                       | fields stored as SolutionField.                              |
        +----------------------+-----------------------+--------------------------------------------------------------+
        | rollback_casedir     | False                 | Rollback case directory by removing all items stored after   |
        |                      |                       | largest restart time. This allows for saving data from a     |
        |                      |                       | restarted simulation in the same case directory.             |
        +----------------------+-----------------------+--------------------------------------------------------------+
        
        """
        params = ParamDict(
                casedir='.',
                restart_times=-1,
                #restart_timesteps=-1,
                solution_names="default",
                rollback_casedir=False,
                #interpolate=True,
                #dt=None,
            )
        return params
    
    def get_restart_conditions(self, function_spaces="default"):
        """ Return restart conditions as requested.
        
        :param dict function_spaces: A dict of dolfin.FunctionSpace on which to return the restart conditions with solution name as key.

        """
        self._pp = PostProcessor(dict(casedir=self.params.casedir, clean_casedir=False))
        
        playlog = self._pp.get_playlog()
        assert playlog != {}, "Playlog is empty! Unable to find restart data."
        
        loadable_solutions = find_solution_presence(self._pp, playlog, self.params.solution_names)
        loadables = find_restart_items(self.params.restart_times, loadable_solutions)
        
        if function_spaces != "default":
            assert isinstance(function_spaces, dict), "Expecting function_spaces kwarg to be a dict"
            assert set(loadables.values()[0].keys()) == set(function_spaces.keys()), "Expecting a function space for each solution variable"
        
        
        def restart_conditions(spaces, loadables):
            # loadables[restart_time0][solution_name] = [(t0, Lt0)] # will load Lt0
            # loadables[restart_time0][solution_name] = [(t0, Lt0), (t1, Lt1)] # will interpolate to restart_time
            functions = {}
            for t in loadables:
                functions[t] = dict()
                for solution_name in loadables[t]:
                    assert len(loadables[t][solution_name]) in [1,2]
                    
                    if len(loadables[t][solution_name]) == 1:
                        f = loadables[t][solution_name][0][1]()
                    elif len(loadables[t][solution_name]) == 2:
                        # Interpolate
                        t0, Lt0 = loadables[t][solution_name][0]
                        t1, Lt1 = loadables[t][solution_name][1]
                        
                        assert t0 <= t <= t1
                        if Lt0.function != None:
                            
                            # The copy-function raise a PETSc-error in parallel
                            #f = Function(Lt0())
                            f0 = Lt0()
                            f = Function(f0.function_space())
                            f.vector().axpy(1.0, f0.vector())
                            del f0
    
                            df = Lt1().vector()
                            df.axpy(-1.0, f.vector())
                            f.vector().axpy((t-t0)/(t1-t0), df)
                        else:
                            f0 = Lt0()
                            f1 = Lt1()
                            datatype = type(f0)
                            if not issubclass(datatype, Iterable):
                                f0 = [f0]; f1 = [f1]
                            
                            f = []
                            for _f0, _f1 in zip(f0, f1):
                                val = _f0 + (t-t0)/(t1-t0)*(_f1-_f0)
                                f.append(val)
                            
                            if not issubclass(datatype, Iterable):
                                f = f[0]
                            else:
                                f = datatype(f)
                     
                    if solution_name in spaces:
                        space = spaces[solution_name]
                        if space != f.function_space():
                            #from fenicstools import interpolate_nonmatching_mesh
                            #f = interpolate_nonmatching_mesh(f, space)
                            try:
                                f = interpolate(f, space)
                            except:
                                f = project(f, space)
                        
                    functions[t][solution_name] = f

            return functions
        
        if function_spaces == "default":
            function_spaces = {}
            for fieldname in loadables.values()[0]:
                try:
                    function_spaces[fieldname] = loadables.values()[0][fieldname][0][1].function.function_space()
                except AttributeError:
                    # This was not a function field
                    pass
        
        result = restart_conditions(function_spaces, loadables)

        ts = 0
        while playlog[str(ts)]["t"] < max(loadables)-1e-14:
            ts += 1
        self.restart_timestep = ts
        if self.params.rollback_casedir:
            self._correct_postprocessing(playlog, ts)
        
        return result
        
        

    def _correct_postprocessing(self, play_log, restart_timestep):
        "Removes data from casedir found at timestep>restart_timestep."
        play_log_to_remove = {}
        for k,v in play_log.items():
            if int(k) >= restart_timestep:
                play_log_to_remove[k] = play_log.pop(k)
        
        all_fields_to_clean = []
                
        for k,v in play_log_to_remove.items():
            if not "fields" in v:
                continue
            else:
                all_fields_to_clean += v["fields"].keys()
        all_fields_to_clean = list(set(all_fields_to_clean))

        for fieldname in all_fields_to_clean:
            self._clean_field(fieldname, restart_timestep)
    
    def _clean_field(self, fieldname, restart_timestep):
        "Deletes data from field found at timestep>restart_timestep."
        metadata = shelve.open(os.path.join(self._pp.get_savedir(fieldname), 'metadata.db'), 'w')
        metadata_to_remove = {}
        for k in metadata.keys():
            try:
                k = int(k)
            except:
                continue
            if k >= restart_timestep:
                metadata_to_remove[str(k)] = metadata.pop(str(k))

        # Remove files and data for all save formats
        self._clean_hdf5(fieldname, metadata_to_remove)
        self._clean_files(fieldname, metadata_to_remove)
        self._clean_txt(fieldname, metadata_to_remove)
        self._clean_shelve(fieldname, metadata_to_remove)
        self._clean_xdmf(fieldname, metadata_to_remove)
        self._clean_pvd(fieldname, metadata_to_remove)
    
    def _clean_hdf5(self, fieldname, del_metadata):        
        delete_from_hdf5_file = '''
        namespace dolfin {
            #include <hdf5.h>  
            void delete_from_hdf5_file(std::string filename, std::string dataset)
            {
                const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
                // Open file existing file for append
                hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
                
                H5Ldelete(file_id, dataset.c_str(), H5P_DEFAULT);
                
                herr_t status = H5Fclose(file_id);
            }
        }
        '''

        cpp_module = compile_extension_module(delete_from_hdf5_file)
        
        hdf5filename = os.path.join(self._pp.get_savedir(fieldname), fieldname+'.hdf5')
        if not os.path.isfile(hdf5filename):
            return
        for k, v in del_metadata.items():
            if not 'hdf5' in v:
                continue
            else:
                cpp_module.delete_from_hdf5_file(hdf5filename, v['hdf5']['dataset'])
        hdf5tmpfilename = os.path.join(self._pp.get_savedir(fieldname), fieldname+'_tmp.hdf5')
        #import ipdb; ipdb.set_trace()
        if on_master_process():
            status, result = getstatusoutput("h5repack -V")
            if status != 0:
                cbc_warning("Unable to run h5repack. Will not repack hdf5-files before replay, which may cause bloated hdf5-files.")
            else:
                subprocess.call("h5repack %s %s" %(hdf5filename, hdf5tmpfilename), shell=True)
                os.remove(hdf5filename)
                os.rename(hdf5tmpfilename, hdf5filename)
        MPI.barrier(mpi_comm_world())
        
    def _clean_files(self, fieldname, del_metadata):
        for k, v in del_metadata.items():
            for i in v.values():
                try:
                    i["filename"]
                except:
                    continue
                
                fullpath = os.path.join(self._pp.get_savedir(fieldname), i['filename'])
                if on_master_process():
                    os.remove(fullpath)
                MPI.barrier(mpi_comm_world())
            """
            #print k,v
            if not 'filename' in v:
                continue
            else:
                fullpath = os.path.join(self.postprocesor.get_savedir(fieldname), v['filename'])
                os.remove(fullpath)
            """

    def _clean_txt(self, fieldname, del_metadata):
        txtfilename = os.path.join(self._pp.get_savedir(fieldname), fieldname+".txt")
        if not os.path.isfile(txtfilename):
            return
        
        txtfile = open(txtfilename, 'r')
        txtfilelines = txtfile.readlines()
        txtfile.close()
        
        num_lines_to_strp = ['txt' in v for v in del_metadata.values()].count(True)
        
        txtfile = open(txtfilename, 'w')
        [txtfile.write(l) for l in txtfilelines[:-num_lines_to_strp]]
        txtfile.close()
        
    def _clean_shelve(self, fieldname, del_metadata):
        shelvefilename = os.path.join(self._pp.get_savedir(fieldname), fieldname+".db")
        if not os.path.isfile(shelvefilename):
            return
        if on_master_process():           
            shelvefile = shelve.open(shelvefilename, 'c')
            for k,v in del_metadata.items():
                if 'shelve' in v:
                    shelvefile.pop(str(k))
            shelvefile.close()
        MPI.barrier(mpi_comm_world())
    
    def _clean_xdmf(self, fieldname, del_metadata):
        basename = os.path.join(self._pp.get_savedir(fieldname), fieldname)
        if not os.path.isfile(basename+".xdmf"):
            return
        i = 0
        while True:
            h5_filename = basename+"_RS"+str(i)+".h5"
            if not os.path.isfile(h5_filename):
                break
            i = i + 1
        
        xdmf_filename = basename+"_RS"+str(i)+".xdmf"
        
        if on_master_process():
            os.rename(basename+".h5", h5_filename)
            os.rename(basename+".xdmf", xdmf_filename)
            
            f = open(xdmf_filename, 'r').read()
            
            new_f = open(xdmf_filename, 'w')
            new_f.write(f.replace(os.path.split(basename)[1]+".h5", os.path.split(h5_filename)[1]))
            new_f.close()
        MPI.barrier(mpi_comm_world())
    
    def _clean_pvd(self, fieldname, del_metadata):
        if os.path.isfile(os.path.join(self._pp.get_savedir(fieldname), fieldname+'.pvd')):
            cbc_warning("No functionality for cleaning pvd-files for restart. Will overwrite.")

