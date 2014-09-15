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
from cbcpost import Parameterized, ParamDict, PostProcessor, SpacePool

#from cbcpost import PostProcessor
#from cbcpost import SpacePool
from cbcpost.utils import cbc_log, Loadable, fetchable_formats


import os, shelve, subprocess
from collections import Iterable, defaultdict
from numpy import array, where, inf

from dolfin import Mesh, Function, HDF5File, tic, toc, norm, project, interpolate, compile_extension_module
from commands import getstatusoutput

#fetchable_formats = ["hdf5", "xml", "xml.gz", "shelve"]

def _create_function_from_metadata(pp, fieldname, metadata, saveformat):
    assert metadata['type'] == 'Function'
    
    # Load mesh
    if saveformat == 'hdf5':    
        mesh = Mesh()
        hdf5file = HDF5File(os.path.join(pp.get_savedir(fieldname), fieldname+'.hdf5'), 'r')
        hdf5file.read(mesh, "Mesh")
        del hdf5file
    elif saveformat == 'xml' or saveformat == 'xml.gz':
        mesh = Mesh(os.path.join(self.postproc.get_casedir(), fieldname, "mesh."+saveformat))
    
    shape = eval(metadata["element_value_shape"])
    degree = eval(metadata["element_degree"])
    family = eval(metadata["element_family"])
    
    # Get space from existing function spaces if mesh is the same
    spaces = SpacePool(mesh)
    space = spaces.get_custom_space(family, degree, shape)

    return Function(space, name=fieldname)


#def find_common_savetimes(play_log, fields):
def find_solution_presence(pp, play_log, fields):
    #if fields == "default": fields = ["default"]
    #common_keys = []
    present_solution = defaultdict(list)
    #times = defaultdict(float)
    functions = dict()
    metadatas = dict()
    for ts, data in play_log.items():
        #if "fields" not in data:
        #    continue
        for fieldname in data.get("fields", []):
            if not any([saveformat in fetchable_formats for saveformat in data["fields"][fieldname]["save_as"]]):
                continue
            
            
            
            is_present = False
            if fields == "default" and data["fields"][fieldname]["type"] == "SolutionField":
                #loadable = Loadable(filename, f, ts, data[ts]["t"], saveformat, function)
                #present_solution[f].append(ts)
                is_present = True
            elif fieldname in fields:
                is_present = True
                #present_solution[f].append(ts)
                
            #loadable = Loadable(filename, f, ts, data[ts]["t"], saveformat, function)
            
            metadata = metadatas.setdefault(fieldname, shelve.open(os.path.join(pp.get_savedir(fieldname), "metadata.db"), 'r'))

            if is_present:
                tic()
                function = None
                if 'hdf5' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+'.hdf5')

                    if fieldname in functions: function = functions[fieldname]
                    else: function = functions.setdefault(fieldname, _create_function_from_metadata(pp, fieldname, metadata, 'hdf5'))

                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], 'hdf5', function))
                elif 'xml' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+'.xml')
                    
                    if fieldname in functions: function = functions[fieldname]
                    else: function = functions.setdefault(fieldname, _create_function_from_metadata(pp, fieldname, metadata, 'xml'))
                    
                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], 'xml', function))
                elif 'xml.gz' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+'.xml.gz')
                    
                    if fieldname in functions: function = functions[fieldname]
                    else: function = functions.setdefault(fieldname, _create_function_from_metadata(pp, fieldname, metadata, 'xml.gz'))

                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], 'xml.gz', function))
                elif 'shelve' in data["fields"][fieldname]["save_as"]:
                    filename = os.path.join(pp.get_savedir(fieldname), fieldname+'.db')
                    present_solution[fieldname].append(Loadable(filename, fieldname, ts, data["t"], "shelve", None))
                print toc()

        #times[ts] = data["t"]
    #return times, present_solution
    return present_solution


def find_restart_items(restart_times, present_solution):
    if not isinstance(restart_times, Iterable):
        restart_times = [restart_times]

    restart_times = [r if r > 0 else inf for r in restart_times]
    #for f, present_timesteps in present_solution.items():
    loadables = dict()
    for restart_time in sorted(restart_times):
        loadables[restart_time] = dict()
        #for fieldname in present_solution:
        for fieldname in present_solution:
            
            
            sorted_ps = sorted(present_solution[fieldname], key=lambda l: l.time)
            present_times = array([l.time for l in sorted_ps])
            #print times
            #continue
            #present_times = [times[ts] for ts in present_timesteps]
            #present_times = array(sorted([l.time for l in present_solution[fieldname]]))
            
            # Find lower and upper limit

            limits = []
            # loadables[restart_time0][solution_name] = [(t0, Lt0)] # will load Lt0
            # loadables[restart_time0][solution_name] = [(t0, Lt0), (t1, Lt1)] # will interpolate to restart_time
            lower = where(present_times <= restart_time)[0]
            if len(lower) > 0: limits.append((present_times[lower[-1]], sorted_ps[lower[-1]]))
            #print lower            
            
            upper = where(present_times >= restart_time)[0]
            if len(upper) > 0 and upper[0] != lower[-1]: limits.append((present_times[upper[0]], sorted_ps[upper[0]]))
            
            loadables[restart_time][fieldname] = limits
            #print "*"*30
            #print upper
            #print lower
            #print limits   
            #else:
            #    upper = None
            #print upper[0]
            #print dir(upper[0])
    
    for k, v in loadables.items():
        print k, v
        if k == inf:
            loadables.pop(k)
            new_k = v.values()[0][0][0]
            loadables[new_k] = v
    #import ipdb; ipdb.set_trace()
            #loadables[]
    #print loadables
    #print min(upper)
    
    
    #pass
    return loadables
    
#def create_loadables()



class Restart(Parameterized):
    def __init__(self, params=None):
        Parameterized.__init__(self, params)
    
    @classmethod
    def default_params(cls):
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
    
    #def get_restart_conditions(self, function_spaces="default", return_as_function=False, depth=1):
    def get_restart_conditions(self, function_spaces="default"):
        #import ipdb; ipdb.set_trace()
        self._pp = PostProcessor(dict(casedir=self.params.casedir, clean_casedir=False))
        
        playlog = self._pp.get_playlog()
        #timesteps, times = find_common_savetimes(playlog, self.params.solution_names)
        tic()
        loadable_solutions = find_solution_presence(self._pp, playlog, self.params.solution_names)
        print toc()
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
                        print "*"*50
                        print t0, t, t1
                        print Lt0.time, t, Lt1.time
                        
                        f = Function(Lt0())
                        print t0, t, t1, norm(f)
                        df = Lt1().vector()
                        df.axpy(-1.0, f.vector())
                        f.vector().axpy((t-t0)/(t1-t0), df)
                        print t0, t, t1, norm(f)
                        
                    space = spaces[solution_name]
                    if space != f.function_space():
                        try:
                            f = interpolate(f, space)
                        except:
                            f = project(f, space)
                        
                    functions[t][solution_name] = f

            return functions
        
        if function_spaces == "default":
            function_spaces = {}
            for fieldname in loadables.values()[0]:
                function_spaces[fieldname] = loadables.values()[0][fieldname][0][1].function.function_space()
        
        #import ipdb; ipdb.set_trace()
        #tic()
        
        result = restart_conditions(function_spaces, loadables)
        
        
        #print ts, playlog[str(ts)]["t"]
        #print toc()
        
        ts = 0
        while playlog[str(ts)]["t"] < max(loadables):
            ts += 1
        self.restart_timestep = ts
        if self.params.rollback_casedir:
            self._correct_postprocessing(playlog, ts)
        #print restart_conditions(function_spaces, loadables)
        
        
        
        #return restart_conditions(function_spaces, loadables)
        return result
        
        

    def _correct_postprocessing(self, play_log, restart_timestep):
        "Removes data from casedir found at timestep>restart_timestep."
        play_log_to_remove = {}
        #import ipdb; ipdb.set_trace()
        for k,v in play_log.items():
            if int(k) >= restart_timestep:
                play_log_to_remove[k] = play_log.pop(k)

        all_fields_to_clean = []
        print play_log_to_remove
        #import ipdb; ipdb.set_trace()
                
        for k,v in play_log_to_remove.items():
            if not "fields" in v:
                continue
            else:
                all_fields_to_clean += v["fields"].keys()
        print all_fields_to_clean
        #import ipdb; ipdb.set_trace()
        all_fields_to_clean = list(set(all_fields_to_clean))
        print all_fields_to_clean
        #import ipdb; ipdb.set_trace()
        for fieldname in all_fields_to_clean:
            
            self._clean_field(fieldname, restart_timestep)
    
    def _clean_field(self, fieldname, restart_timestep):
        "Deletes data from field found at timestep>restart_timestep."
        metadata = shelve.open(os.path.join(self._pp.get_savedir(fieldname), 'metadata.db'), 'w')
        #import ipdb; ipdb.set_trace()
        metadata_to_remove = {}
        for k in metadata.keys():
            try:
                k = int(k)
            except:
                continue
            if k > restart_timestep:
                metadata_to_remove[str(k)] = metadata.pop(str(k))
        #import ipdb; ipdb.set_trace()
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
        status, result = getstatusoutput("h5repack -V")
        if status != 0:
            cbc_warning("Unable to run h5repack. Will not repack hdf5-files before replay, which may cause bloated hdf5-files.")
        else:
            subprocess.call("h5repack %s %s" %(hdf5filename, hdf5tmpfilename), shell=True)
            os.remove(hdf5filename)
            os.rename(hdf5tmpfilename, hdf5filename)
        
        
    def _clean_files(self, fieldname, del_metadata):
        for k, v in del_metadata.items():
            if not 'filename' in v:
                continue
            else:
                fullpath = os.path.join(self.postprocesor.get_savedir(fieldname), v['filename'])
                os.remove(fullpath)

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
        
        shelvefile = shelve.open(shelvefilename, 'c')
        for k,v in del_metadata.items():
            if 'shelve' in v:
                shelvefile.pop(str(k))
        shelvefile.close()
    
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
        
        os.rename(basename+".h5", h5_filename)
        os.rename(basename+".xdmf", xdmf_filename)
        
        f = open(xdmf_filename, 'r').read()
        
        new_f = open(xdmf_filename, 'w')
        new_f.write(f.replace(os.path.split(basename)[1]+".h5", os.path.split(h5_filename)[1]))
        new_f.close()
    
    def _clean_pvd(self, fieldname, del_metadata):
        if os.path.isfile(os.path.join(self._pp.get_savedir(fieldname), fieldname+'.pvd')):
        #if os.path.isfile(self.casedir+"/"+fieldname+"/"+fieldname+".pvd"):
            cbc_warning("No functionality for cleaning pvd-files for restart. Will overwrite.")
    
    