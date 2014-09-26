from hashlib import sha1
from dolfin import Function, MPI, mpi_comm_world, plot, File, HDF5File, XDMFFile, error

from cbcpost.utils import safe_mkdir, hdf5_link
from cbcpost.fieldbases import Field

import os, shelve, pickle
from shutil import rmtree

from utils import on_master_process

class Saver():
    def __init__(self, timer, casedir):
        self._timer = timer
        # Caches for file storage
        self._datafile_cache = {}
        self._casedir = casedir
        
        self._create_casedir()
        

    def _get_save_formats(self, field, data):
        if data == None:
            return []
        
        if field.params.save_as == Field.default_save_as():
            # Determine proper file formats from data type if not specifically provided
            if isinstance(data, Function):
                save_as = ['xdmf', 'hdf5']
            elif isinstance(data, (float, int, list, tuple, dict)):
                save_as = ['txt', 'shelve']
            else:
                error("Unknown data type %s for field %s, cannot determine file type automatically." % (type(data).__name__, field.name))
        else:
            if isinstance(field.params.save_as, (list, tuple)):
                save_as = list(field.params.save_as)
            else:
                save_as = [field.params.save_as]
        return save_as

    def get_casedir(self):
        return self._casedir

    def _clean_casedir(self):
        "Cleans out all files produced by cbcpost in the current casedir."
        if on_master_process():
            if os.path.isdir(self.get_casedir()):
                playlogfilename = os.path.join(self.get_casedir(), "play.db")
                if os.path.isfile(playlogfilename):
                    playlog = shelve.open(playlogfilename, 'r')

                    all_fields = []
                    for k,v in playlog.items():
                        all_fields += v.get("fields", {}).keys()
    
                    all_fields = list(set(all_fields))
                    playlog.close()
                    
                    for field in all_fields:
                        rmtree(os.path.join(self.get_casedir(), field))
                    
                    for f in ["mesh.hdf5", "play.db", "params.txt", "params.pickle"]:
                        if os.path.isfile(os.path.join(self.get_casedir(), f)):
                            os.remove(os.path.join(self.get_casedir(), f))
        MPI.barrier(mpi_comm_world())

    def _create_casedir(self):
        casedir = self._casedir
        safe_mkdir(casedir)
        return casedir

    def get_savedir(self, field_name):
        "Returns savedir for given fieldname"
        return os.path.join(self._casedir, field_name)

    def _create_savedir(self, field_name):
        self._create_casedir()
        savedir = self.get_savedir(field_name)
        safe_mkdir(savedir)
        return savedir

    """
    def _init_metadata_file(self, field_name, init_data):
        savedir = self._create_savedir(field_name)
        if on_master_process():
            metadata_filename = os.path.join(savedir, 'metadata.db')
            metadata_file = shelve.open(metadata_filename)
            metadata_file["init_data"] = init_data
            metadata_file.close()
    """
    """
    def _finalize_metadata_file(self, field_name, finalize_data):
        if on_master_process():
            savedir = self.get_savedir(field_name)
            metadata_filename = os.path.join(savedir, 'metadata.db')
            metadata_file = shelve.open(metadata_filename)
            metadata_file["finalize_data"] = finalize_data
            metadata_file.close()
    """

    def _update_metadata_file(self, field_name, data, t, timestep, save_as, metadata):
        if on_master_process():
            savedir = self.get_savedir(field_name)
            metadata_filename = os.path.join(savedir, 'metadata.db')
            metadata_file = shelve.open(metadata_filename)

            # Store some data the first time
            if "type" not in metadata_file and data != None:
                # Data about type and formats
                metadata_file["type"] = type(data).__name__
                metadata_file["saveformats"] = list(set(save_as+metadata_file.get("saveformats", [])))
                # Data about function space
                if isinstance(data, Function):
                    metadata_file["element"] = repr(data.element(),)
                    metadata_file["element_degree"] = repr(data.element().degree(),)
                    metadata_file["element_family"] = repr(data.element().family(),)
                    metadata_file["element_value_shape"] = repr(data.element().value_shape(),)
            # Store some data each timestep
            metadata_file[str(timestep)] = metadata
            metadata_file[str(timestep)]["t"] = t

            # Flush file between timesteps
            metadata_file.close()

    def _get_datafile_name(self, field_name, saveformat, timestep):
        # These formats produce a new file each time
        counted_formats = ('xml', 'xml.gz')

        metadata = {}

        # Make filename, with or without save count in name
        if saveformat in counted_formats:
            filename = "%s%d.%s" % (field_name, timestep, saveformat)
            # If we have a new filename each time, store the name in metadata
            #metadata = [('filename', filename)]
            metadata['filename'] = filename
        elif saveformat == "shelve":
            filename = "%s.%s" % (field_name, "db")
        else:
            filename = "%s.%s" % (field_name, saveformat)
            if saveformat == 'hdf5':
                metadata['dataset'] = field_name+str(timestep)

        savedir = self.get_savedir(field_name)
        fullname = os.path.join(savedir, filename)
        return fullname, metadata

    def _update_pvd_file(self, field_name, saveformat, data, timestep, t):
        assert isinstance(data, Function)
        assert saveformat == "pvd"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        key = (field_name, saveformat)
        datafile = self._datafile_cache.get(key)
        if datafile is None:
            datafile = File(fullname)
            self._datafile_cache[key] = datafile
        datafile << data
        return metadata

    def _update_xdmf_file(self, field_name, saveformat, data, timestep, t):
        assert isinstance(data, Function)
        assert saveformat == "xdmf"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        key = (field_name, saveformat)
        datafile = self._datafile_cache.get(key)
        if datafile is None:
            datafile = XDMFFile(mpi_comm_world(), fullname)
            datafile.parameters["rewrite_function_mesh"] = False
            datafile.parameters["flush_output"] = True
            self._datafile_cache[key] = datafile
        datafile << (data, t)
        return metadata

    def _update_hdf5_file(self, field_name, saveformat, data, timestep, t):
        assert saveformat == "hdf5"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        
        # Create "good enough" hash. This is done to avoid data corruption when restarted from
        # different number of processes, different distribution or different function space
        local_hash= sha1()
        local_hash.update(str(data.function_space().mesh().num_cells()))
        local_hash.update(str(data.function_space().ufl_element()))
        local_hash.update(str(data.function_space().dim()))
        local_hash.update(str(MPI.size(mpi_comm_world())))
        
        # Global hash (same on all processes), 10 digits long
        hash = str(int(MPI.sum(mpi_comm_world(), int(local_hash.hexdigest(), 16))%1e10)).zfill(10)
        
        #key = (field_name, saveformat)
        #datafile = self._datafile_cache.get(key)
        #if datafile is None:
        #    datafile = HDF5File(mpi_comm_world(), fullname, 'w')
        #    self._datafile_cache[key] = datafile
        
        # Open HDF5File
        if not os.path.isfile(fullname):
            datafile = HDF5File(mpi_comm_world(), fullname, 'w')
        else:
            datafile = HDF5File(mpi_comm_world(), fullname, 'a')
        
        # Write to hash-dataset if not yet done
        if not datafile.has_dataset(hash) or not datafile.has_dataset(hash+"/"+field_name):
            datafile.write(data, str(hash)+"/"+field_name)
            
        if not datafile.has_dataset("Mesh"):
            datafile.write(data.function_space().mesh(), "Mesh")
        
        # Write vector to file
        # TODO: Link vector when function has been written to hash
        datafile.write(data.vector(), field_name+str(timestep)+"/vector")

        del datafile        
        # Link information about function space from hash-dataset
        hdf5_link(fullname, str(hash)+"/"+field_name+"/x_cell_dofs", field_name+str(timestep)+"/x_cell_dofs")
        hdf5_link(fullname, str(hash)+"/"+field_name+"/cell_dofs", field_name+str(timestep)+"/cell_dofs")
        hdf5_link(fullname, str(hash)+"/"+field_name+"/cells", field_name+str(timestep)+"/cells")
        
        return metadata

    def _update_xml_file(self, field_name, saveformat, data, timestep, t):
        assert saveformat == "xml"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        meshfile = os.path.join(self.get_savedir(field_name), "mesh.hdf5")
        if not os.path.isfile(meshfile):
            hdf5file = HDF5File(mpi_comm_world(), meshfile, 'w')
            hdf5file.write(data.function_space().mesh(), "Mesh")
            del hdf5file
        datafile = File(fullname)
        datafile << data
        return metadata

    def _update_xml_gz_file(self, field_name, saveformat, data, timestep, t):
        assert saveformat == "xml.gz"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        meshfile = os.path.join(self.get_savedir(field_name), "mesh.hdf5")
        if not os.path.isfile(meshfile):
            hdf5file = HDF5File(mpi_comm_world(), meshfile, 'w')
            hdf5file.write(data.function_space().mesh(), "Mesh")
            del hdf5file
        datafile = File(fullname)
        datafile << data
        return metadata

    def _update_txt_file(self, field_name, saveformat, data, timestep, t):
        # TODO: Identify which more well defined data formats we need
        assert saveformat == "txt"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        if on_master_process():
            datafile = open(fullname, 'a')
            datafile.write(str(data))
            datafile.write("\n")
            datafile.close()
        return metadata

    def _update_shelve_file(self, field_name, saveformat, data, timestep, t):
        assert saveformat == "shelve"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        if on_master_process():
            datafile = shelve.open(fullname)
            datafile[str(timestep)] = data
            datafile.close()
            
        return metadata

    def _fetch_play_log(self):
        casedir = self.get_casedir()
        play_log_file = os.path.join(casedir, "play.db")
        play_log = shelve.open(play_log_file)
        return play_log

    def _update_play_log(self, t, timestep):
        if on_master_process():
            play_log = self._fetch_play_log()
            if str(timestep) in play_log:
                play_log.close()
                return
            play_log[str(timestep)] = {"t":float(t)}
            play_log.close()

    def _fill_play_log(self, field, timestep, save_as):
        if on_master_process():
            play_log = self._fetch_play_log()
            timestep_dict = dict(play_log[str(timestep)])
            if "fields" not in timestep_dict:
                timestep_dict["fields"] = {}
            timestep_dict["fields"][field.name] = {"type": field.__class__.shortname(), "save_as": save_as}
            play_log[str(timestep)] = timestep_dict
            play_log.close()

    def store_params(self, params):
        "Store parameters in casedir as params.pickle and params.txt."
        casedir = self._create_casedir()

        pfn = os.path.join(casedir, "params.pickle")
        with open(pfn, 'w') as f:
            pickle.dump(params, f)

        tfn = os.path.join(casedir, "params.txt")
        with open(tfn, 'w') as f:
            f.write(str(params))

    def store_mesh(self, mesh, cell_domains=None, facet_domains=None):
        "Store mesh in casedir to mesh.hdf5 (dataset Mesh) in casedir."
        casedir = self.get_casedir()
        meshfile = HDF5File(mpi_comm_world(), os.path.join(casedir, "mesh.hdf5"), 'w')
        meshfile.write(mesh, "Mesh")
        if cell_domains != None:
            meshfile.write(cell_domains, "CellDomains")
        if facet_domains != None:
            meshfile.write(facet_domains, "FacetDomains")
        del meshfile
    
    def _action_save(self, field, data):
        "Apply the 'save' action to computed field data."
        field_name = field.name

        # Create save folder first time
        self._create_savedir(field_name)

        # Get current time (assuming the cache contains
        # valid 't' and 'timestep' at each step)
        t = self.t
        timestep = self.timestep

        # Collect metadata shared between data types
        metadata = {
            'timestep': timestep,
            'time': t,
            }

        # Rename Functions to get the right name in file
        # (NB! This has the obvious side effect!)
        # TODO: We don't need to cache a distinct Function
        # object like we do for plotting, or?
        if isinstance(data, Function):
            data.rename(field_name, "Function produced by cbcpost.")
        
        # Get list of file formats
        save_as = self._get_save_formats(field, data)
        
        # Write data to file for each filetype
        for saveformat in save_as:
            # Write data to file depending on type
            if saveformat == 'pvd':
                metadata[saveformat] = self._update_pvd_file(field_name, saveformat, data, timestep, t)
            elif saveformat == 'xdmf':
                metadata[saveformat] = self._update_xdmf_file(field_name, saveformat, data, timestep, t)
            elif saveformat == 'xml':
                metadata[saveformat] = self._update_xml_file(field_name, saveformat, data, timestep, t)
            elif saveformat == 'xml.gz':
                metadata[saveformat] = self._update_xml_gz_file(field_name, saveformat, data, timestep, t)
            elif saveformat == 'txt':
                metadata[saveformat] = self._update_txt_file(field_name, saveformat, data, timestep, t)
            elif saveformat == 'hdf5':
                metadata[saveformat] = self._update_hdf5_file(field_name, saveformat, data, timestep, t)
            elif saveformat == 'shelve':
                metadata[saveformat] = self._update_shelve_file(field_name, saveformat, data, timestep, t)
            else:
                error("Unknown save format %s." % (saveformat,))
            self._timer.completed("PP: save %s %s" %(field_name, saveformat))

        # Write new data to metadata file
        self._update_metadata_file(field_name, data, t, timestep, save_as, metadata)
        
        self._fill_play_log(field, timestep, save_as)
    
    
    def update(self, t, timestep, cache, triggered_or_finalized):
        #print cache, triggered_or_finalized
        self.t = t
        self.timestep = timestep
        for field in triggered_or_finalized:
            #print name, cache[name]
            if field.params.save:
                self._action_save(field, cache[field.name])
        
        """
        if self._last_trigger_time[name][1] == timestep or finalize:
            for action in ["save", "plot", "callback"]:
                if field.params[action]:
                    self._apply_action(action, field, data)
        """
        


    