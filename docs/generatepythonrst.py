# Copyright (C) 2011 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#

import os, sys, types

__all__ = ["generate_python_api_documentation"]

def indent(string, num_spaces):
    "Indent given text block given number of spaces"
    return "\n".join(num_spaces*" " + l for l in string.split("\n"))

class Module(object):
    def __init__(self, name, mod_file):
        #print name
        self.name = name
        self.file = mod_file
        self.submodules = []
        self.classes = []
        self.functions = []
    def __hash__(self):
        return self.name

def get_modules(parent, loc, modules):
    for mod in os.listdir(loc):
        f = os.path.join(loc, mod)
        new_mod = None
        mod_to_append = None
        # Add modules (files) to global dict and to parent as submodules.
        if os.path.isfile(f):
            m, e = os.path.splitext(mod)
            if e == ".py" and m != "__init__":
                new_mod = Module(".".join([parent.name, m]), f)
                mod_to_append = m
                
        # Add submodules (directories with '__init__.py' files) to global dict
        # and to parent as submodules.
        if os.path.isdir(f):
            if not "__init__.py" in os.listdir(f):
                continue
            new_mod = Module(".".join([parent.name, mod]), os.path.join(f, "__init__.py"))
            mod_to_append = mod

            # Recursively extract submodules.
            get_modules(new_mod, f, modules)

        if new_mod is not None:
            if new_mod in modules:
                raise RuntimeError("module already present???")
            parent_classes, parent_functions = get_objects(parent)
            classes, functions = get_objects(new_mod)
        
            parent_classes, parent_functions = set(parent_classes), set(parent_functions)
            classes, functions = set(classes), set(functions)
            
            new_mod.classes = list(classes-parent_classes)
            new_mod.functions = list(functions-parent_functions)
            
            if len(new_mod.classes)+len(new_mod.functions)+len(new_mod.submodules) > 0:
                parent.submodules.append(mod_to_append)
                modules.append(new_mod)
            
def get_objects(module):
    """Extract classes and functions defined in a module.
    The function will not return imported classes and functions."""
    classes = []
    functions = []
    objects = {}

    # NOTE: Dirty hack for Python 2.6, in 2.7 it should be possible to use
    # importlib for submodules to test if __all__ is defined.
    if "__all__" in open(module.file, "r").read():
        define_all = True
    else:
        define_all = False

    # Get objects listed in __all__ by developer.
    exec("from %s import *" % module.name, objects)

    for key, val in objects.items():
        if isinstance(val, (types.ClassType, types.TypeType)):
            if define_all or module.name == val.__module__:
                classes.append(key)
        elif isinstance(val, types.FunctionType):
            if define_all or module.name == val.__module__:
                functions.append(key)
        # Anything else we need to catch?
        else:
            pass

    return classes, functions

def index_items(item_type, items):
    return """
%s:

.. toctree::
    :maxdepth: 1

%s

""" % (item_type, indent("\n".join(sorted(items)), 4))

def caption(string, level, top=False):
    markers = level*len(string)
    if top:
        return "%s\n%s\n%s\n" % (markers, string, markers)
    return "%s\n%s\n" % (string, markers)

def label(package_name, name):
    output = ".. _programmers_reference_"
    return output + "%s:\n\n" % name

def write_class(name, module_name):
    output = "\n.. currentmodule:: %s\n\n" % module_name
    output += ".. autoclass:: %s\n" %  name
    output += "   :members:\n"
    output += "   :undoc-members:\n"
    output += "   :show-inheritance:\n"
    output += "\n"

    return output

def write_function(name, module_name):
    output = "\n.. currentmodule:: %s\n\n" % module_name
    output += ".. autofunction:: %s\n" % (name)
    return output
    
def write_object(package_name, directory, module_name, name, obj_type):
    output = ".. Documentation for the %s %s\n\n" % (obj_type, module_name + "." + name)
    output += label(package_name, "_".join(module_name.split(".")[1:] + [name.lower()]))
    
    if obj_type == "class":
        output += write_class(name, module_name)
    else:
        output += name + "\n"
        output += "="*len(name) + "\n"
        output += "\n.. currentmodule:: %s\n\n" % module_name
        output += ".. auto%s:: %s\n" % (obj_type, name)
    outfile = os.path.join(directory, name + ".rst")
    f = open(outfile, "w")
    f.write(output)
    f.close()

def write_documentation(package_name, module, output_dir, version):

    package_version = package_name + "-" + version
    dirs = [output_dir]
    dirs += module.name.split(".")[1:]
    directory = os.path.sep.join(dirs)

    try:
        os.makedirs(directory)
    except:
        pass

    modules = []

    # Special handling of cpp module in dolfin.
    for sub in module.submodules:
        modules.append(sub + "/index")

    classes, functions = module.classes, module.functions

    output = ".. Index file for the %s module.\n\n" % module.name
    output += label(package_name, "_".join(module.name.split(".")[1:] + ["index"]))

    if module.name == package_name and package_name == "dolfin":
        output += caption("Programmer's reference for DOLFIN (Python)", "#")
    elif module.name == package_name:
        output += caption("Programmer's reference", "#")
    else:
        header = "%s module" % module.name
        output += caption(header, "*")
        
        
    if not modules and not classes and not functions:
        return

    outfile = os.path.join(directory, "index.rst")
    f = open(outfile, "w")
    f.write(output)
    f.write("""\n.. automodule:: %s\n\n""" % module.name)
    
#    f.write("""\n
#.. automodule:: %s
#   :no-members:
#   :no-undoc-members:
#   :no-show-inheritance: \n\n""" % module.name)
    

    if modules:
        f.write(index_items("Modules", modules))
    if classes:
        f.write("Classes \n")
        f.write("="*20+"\n")
        for c in classes:
            f.write(write_class(c, module.name))
    if functions:
        f.write("Functions \n")
        f.write("="*20+"\n")
        for function in functions:
            f.write(write_function(function, module.name))
    f.write('\n')
    f.close()


def generate_python_api_documentation(module, output_dir, version):

    print "\nWriting files for module: %s ... " % module.__name__
    submods = [Module(module.__name__,
                      os.path.join(os.path.dirname(module.__file__),
                                   "__init__.py"))]
    
    

    get_modules(submods[0], os.path.dirname(module.__file__), submods)
    print "Writing files for submodules ... "
    for submod in sorted(submods, key=lambda o: o.name):
        print "  Writing files for sub module: ", submod.name
        write_documentation(module.__name__, submod, output_dir, version)
