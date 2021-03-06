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
Module to handle all I/O from cbcpost. This includes default saving such as
the play log, as well as all specified Field-saving.

This code is intended for internal usage, and is called from a PostProcessor
instance.
"""
from hashlib import sha1
from dolfin import (Function, MPI, mpi_comm_world, File, HDF5File, XDMFFile,
                    error, dolfin_version)

from cbcpost.utils import safe_mkdir, hdf5_link, on_master_process, cbc_log
from cbcpost.fieldbases import Field
from distutils.version import LooseVersion
import os
import shelve
import pickle
from shutil import rmtree

def _get_save_formats(field, data):
    """Get save formats associated with field.

    Default values are xdmf and hdf5 for dolfin.Function-type data, and
    txt and shelve for types float, int, list, tuple and dict.
    """
    if data is None:
        return []

    if field.params.save_as == Field.default_save_as():
        # Determine proper file formats from data type if not specifically
        # provided
        if isinstance(data, Function):
            save_as = ['xdmf', 'hdf5']
        elif isinstance(data, (float, int, list, tuple, dict)):
            save_as = ['txt', 'shelve']
        else:
            error("Unknown data type %s for field %s, cannot determine \
                  file type automatically."
                  % (type(data).__name__, field.name))
    else:
        if isinstance(field.params.save_as, (list, tuple)):
            save_as = list(field.params.save_as)
        else:
            save_as = [field.params.save_as]
    return save_as


class Saver():
    "Class to handle all saving in cbcpost."
    _playlog = dict()

    def __init__(self, timer, casedir, flush_frequency):
        self._timer = timer
        # Caches for file storage
        self._datafile_cache = {}

        self._metadata_cache = {}
        self._playlog[casedir] = None

        self._casedir = casedir
        self.flush_frequency = flush_frequency

        self._create_casedir()

    def get_casedir(self):
        "Return case directory."
        return self._casedir

    def _clean_casedir(self):
        "Cleans out all files produced by cbcpost in the current casedir."
        MPI.barrier(mpi_comm_world())
        if on_master_process():
            if os.path.isdir(self.get_casedir()):
                try:
                    playlog = self._fetch_playlog()
                except:
                    playlog = None

                if playlog is not None:
                    all_fields = []
                    for v in playlog.values():
                        all_fields += v.get("fields", {}).keys()

                    all_fields = list(set(all_fields))
                    playlog.close()

                    for field in all_fields:
                        rmtree(os.path.join(self.get_casedir(), field))

                    for f in ["mesh.hdf5", "play.db", "params.txt",
                              "params.pickle"]:

                        if os.path.isfile(os.path.join(self.get_casedir(), f)):
                            os.remove(os.path.join(self.get_casedir(), f))

        MPI.barrier(mpi_comm_world())

    def _create_casedir(self):
        "Create case directory"
        casedir = self._casedir
        safe_mkdir(casedir)
        return casedir

    def get_savedir(self, field_name):
        "Returns save directory for given field name"
        return os.path.join(self._casedir, field_name)

    def _create_savedir(self, field_name):
        "Create save directory for field name"
        self._create_casedir()
        savedir = self.get_savedir(field_name)
        safe_mkdir(savedir)
        return savedir

    def _update_metadata_file(self, field_name, data, t, timestep, save_as, metadata):
        "Update metadata shelve file from master process."
        if on_master_process():
            if self._metadata_cache.get(field_name) is None:
                savedir = self.get_savedir(field_name)
                metadata_filename = os.path.join(savedir, 'metadata.db')
                #metadata_file = shelve.open(metadata_filename)
                metadata_file = shelve.open(metadata_filename)
                self._metadata_cache[field_name] = metadata_file

            metadata_file = self._metadata_cache[field_name]

            # Store some data the first time
            if "type" not in metadata_file and data is not None:
                # Data about type and formats
                metadata_file["type"] = type(data).__name__
                metadata_file["saveformats"] = list(set(save_as+metadata_file.get("saveformats", [])))
                # Data about function space
                if isinstance(data, Function):
                    if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
                        element = data.ufl_element()
                    else:
                        element = data.element()

                    metadata_file["element"] = repr(element,)
                    metadata_file["element_degree"] = repr(element.degree(),)
                    metadata_file["element_family"] = repr(element.family(),)
                    metadata_file["element_value_shape"] = repr(element.value_shape(),)
            # Store some data each timestep
            metadata_file[str(timestep)] = metadata
            metadata_file[str(timestep)]["t"] = t

            # Flush file between timesteps
            #metadata_file.close()

    def _get_datafile_name(self, field_name, saveformat, timestep):
        """Get datafile name associated with given field name, saveformat and
        timestep. This is used when saving files to disk."""
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
        "Update pvd file with new data."
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

    def _update_xdmf_file(self, field_name, saveformat, data, timestep, t, rewrite_mesh=False):
        "Update xdmf file with new data."
        assert isinstance(data, Function)
        assert saveformat == "xdmf"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        key = (field_name, saveformat)
        datafile = self._datafile_cache.get(key)
        if datafile is None:
            datafile = XDMFFile(mpi_comm_world(), fullname)
            datafile.parameters["rewrite_function_mesh"] = rewrite_mesh
            datafile.parameters["flush_output"] = True
            self._datafile_cache[key] = datafile
        try:
            datafile.write(data, float(t))
        except:
            datafile << (data, float(t))
        return metadata

    def _update_hdf5_file(self, field_name, saveformat, data, timestep, t):
        """Update hdf5 file with new data.

        This creates a hashed dataset within the dataset to save FunctionSpace
        information only once, and for all subsequent savings only the vector
        is saved and links are created to the FunctionSpace information.

        This ensures that the saving is fully compatible with restart and
        replay on an arbitrary number of processes.
        """
        assert saveformat == "hdf5"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)

        # Create "good enough" hash. This is done to avoid data corruption when restarted from
        # different number of processes, different distribution or different function space
        local_hash = sha1()
        local_hash.update(str(data.function_space().mesh().num_cells()).encode())
        local_hash.update(str(data.function_space().ufl_element()).encode())
        local_hash.update(str(data.function_space().dim()).encode())
        local_hash.update(str(MPI.size(mpi_comm_world())).encode())

        # Global hash (same on all processes), 10 digits long
        global_hash = MPI.sum(mpi_comm_world(), int(local_hash.hexdigest(), 16))
        global_hash = str(int(global_hash%1e10)).zfill(10)

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
        if not datafile.has_dataset(global_hash) or not datafile.has_dataset(global_hash+"/"+field_name):
            datafile.write(data, str(global_hash)+"/"+field_name)

        if not datafile.has_dataset("Mesh"):
            datafile.write(data.function_space().mesh(), "Mesh")

        # Write vector to file
        # TODO: Link vector when function has been written to hash
        datafile.write(data.vector(), field_name+str(timestep)+"/vector")

        # HDF5File.close is broken in 1.4
        if dolfin_version() == "1.4.0+":
            datafile.close()
        del datafile
        # Link information about function space from hash-dataset
        hdf5_link(fullname, str(global_hash)+"/"+field_name+"/x_cell_dofs", field_name+str(timestep)+"/x_cell_dofs")
        hdf5_link(fullname, str(global_hash)+"/"+field_name+"/cell_dofs", field_name+str(timestep)+"/cell_dofs")
        hdf5_link(fullname, str(global_hash)+"/"+field_name+"/cells", field_name+str(timestep)+"/cells")

        return metadata

    def _update_xml_file(self, field_name, saveformat, data, timestep, t):
        "Create new xml file for current timestep with new data."
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
        "Create new xml.gz file for current timestep with new data."
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
        "Update txt file with a string representation of the data."
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
        "Update shelve file with new data."
        assert saveformat == "shelve"
        fullname, metadata = self._get_datafile_name(field_name, saveformat, timestep)
        if on_master_process():
            key = (field_name, saveformat)
            datafile = self._datafile_cache.get(key)
            if datafile is None:
                datafile = shelve.open(fullname)
                self._datafile_cache[key] = datafile

            #datafile = shelve.open(fullname)
            datafile[str(timestep)] = data
            #datafile.close()

        return metadata

    def _fetch_playlog(self, flag='c', writeback=False):
        "Get play log from disk (which is stored as a shelve-object)."
        if self._playlog[self.get_casedir()] is not None:
            self._playlog[self.get_casedir()].close()
            self._playlog[self.get_casedir()] = None
        casedir = self.get_casedir()
        playlog_file = os.path.join(casedir, "play.db")
        playlog = shelve.open(playlog_file, flag=flag, writeback=writeback)
        return playlog

    def _update_playlog(self, t, timestep):
        "Update play log from master process with current time"
        if on_master_process():
            if self._playlog[self.get_casedir()] is None:
                self._playlog[self.get_casedir()] = self._fetch_playlog()
            #playlog = self._fetch_playlog()
            if str(timestep) in self._playlog[self.get_casedir()]:
                #playlog.close()
                return
            self._playlog[self.get_casedir()][str(timestep)] = {"t":float(t)}
            #playlog.close()

    def _fill_playlog(self, field, timestep, save_as):
        "Update play log with the data that has been stored at this timestep"
        if on_master_process():
            if self._playlog[self.get_casedir()] is None:
                self._playlog[self.get_casedir()] = self._fetch_playlog()
            timestep_dict = dict(self._playlog[self.get_casedir()][str(timestep)])
            if "fields" not in timestep_dict:
                timestep_dict["fields"] = {}
            timestep_dict["fields"][field.name] = {"type": field.__class__.shortname(), "save_as": save_as}
            self._playlog[self.get_casedir()][str(timestep)] = timestep_dict
            #playlog.close()

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
        if cell_domains is not None:
            meshfile.write(cell_domains, "CellDomains")
        if facet_domains is not None:
            meshfile.write(facet_domains, "FacetDomains")
        del meshfile

    def _flush_data(self):
        "Flush data to disk"
        if on_master_process():
            for f in self._metadata_cache.values():
                f.sync()
            if self._playlog[self.get_casedir()] is not None:
                self._playlog[self.get_casedir()].sync()

            for key, f in self._datafile_cache.items():
                fieldname, saveformat = key
                if saveformat == "shelve":
                    f.sync()

    def _close_shelves(self):
        "Close all shelve files"
        if on_master_process():
            """
            for k in self._metadata_cache.keys():
                f = self._metadata_cache[k]
                f.sync()
                f.close()
                #self._metadata_cache[k] = None
                self._metadata_cache.pop(k)
            """

            if self._playlog[self.get_casedir()] is not None:
                self._playlog[self.get_casedir()].sync()
                self._playlog[self.get_casedir()].close()
                self._playlog[self.get_casedir()] = None

            #for key, f in self._datafile_cache.iteritems():
            for key in self._datafile_cache.keys():
                fieldname, saveformat = key
                if saveformat == "shelve":
                    f = self._datafile_cache[key]
                    f.sync()
                    f.close()
                    #self._datafile_cache[key] = None
                    self._datafile_cache.pop(key)


    def _action_save(self, field, data, timestep, t):
        "Apply the 'save' action to computed field data."
        field_name = field.name

        # Create save folder first time
        self._create_savedir(field_name)

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
        save_as = _get_save_formats(field, data)
        # Notify user if self.rewrite_mesh has been set without mattering
        if field.params["rewrite_mesh"]:
            if not "xdmf" in save_as:
                cbc_log(30, "rewrite_mesh flag has no effect for field {}".format(field_name))
    
        # Write data to file for each filetype
        for saveformat in save_as:
            # Write data to file depending on type
            if saveformat == 'pvd':
                metadata[saveformat] = self._update_pvd_file(field_name, saveformat, data, timestep, t)
            elif saveformat == 'xdmf':
                metadata[saveformat] = self._update_xdmf_file(field_name, saveformat, data, timestep, t, field.params["rewrite_mesh"])
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
        self._timer.completed("PP: update metadata")
        self._fill_playlog(field, timestep, save_as)

        self._timer.completed("PP: update playlog")


    def update(self, t, timestep, cache, triggered_or_finalized):
        """Iterate through the triggered_or_finalized-list of fields, and save
        the fields with Field.params.save == True"""
        for field in triggered_or_finalized:
            #print name, cache[name]
            if field.params.save:
                cbc_log(20, "Saving field %s" %field.name)
                self._action_save(field, cache[field.name], timestep, t)

        if timestep%self.flush_frequency == 0:
            self._flush_data()
            self._timer.completed("PP: flush data")




