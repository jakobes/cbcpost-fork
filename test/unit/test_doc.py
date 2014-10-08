
import pytest
import inspect
import xml.etree.ElementTree as etree
from docutils.core import publish_doctree


def parse_rst_text_to_doc_dict(doc):
    doc = doc.split('\n')
    doc = [d.strip() for d in doc]
    doc = '\n'.join(doc)
    
    doctree = publish_doctree(doc)
    d2 = doctree.asdom()
    d3 = etree.fromstring(d2.toxml())
    
    doc_dict = dict()
    for table in d3.findall('table'):
        # Check if it is a correct format for param documentation
        if table[0].attrib['cols'] != '3':
            continue
        
        head = table[0].find('thead')
        if len(head[0]) != 3:
            continue

        headers = [head[0][i][0].text for i in range(3)]
        if headers == ["Key", "Default value", "Description"]:
            # Table has correct format!
            body = table[0].find('tbody')
            
            for row in body.findall('row'):
                k, v = row[0][0].text, row[1][0].text
                assert k not in doc_dict, "Parameter %s documented multiple times" %k
                doc_dict[k] = v

    return doc_dict

import cbcpost
from cbcpost import Parameterized
@pytest.mark.parametrize("Parameterized_subclass", [v for v in cbcpost.__dict__.values()
                                                    if inspect.isclass(v)
                                                    and issubclass(v, Parameterized)
                                                    and not v is Parameterized])
def test_params_doc(Parameterized_subclass):    
    v = Parameterized_subclass

    params = v.default_params()
    required_doc_keys = []
    supercls = v.mro()[1]
    
    required_doc_keys = set(params.keys())
    for supercls in v.mro()[1:]:
        if issubclass(supercls, Parameterized) and not supercls is Parameterized:
            required_doc_keys -= set(supercls.default_params().keys())
    required_doc_keys = list(required_doc_keys)
    required_doc_params = dict([(k, params[k]) for k in required_doc_keys])
    
    doc = v.default_params.__doc__
    if doc == None:
        assert len(required_doc_keys) == 0, ("Found no docstring, but undocumented parameters %s are introduced." \
                                             % (str(required_doc_keys)))
        return
    
    doc_params = parse_rst_text_to_doc_dict(doc)

    assert set(doc_params.keys()) <= set(params.keys())#, "All documented parameters does not exist in default_params"
    assert set(required_doc_params.keys()) <= set(doc_params.keys())#, "Undocumented parameter(s) are introduced" 
    
    for key in doc_params.keys():
        assert params[key] == eval(doc_params[key])


import fnmatch
import os
d = os.path.split(inspect.getabsfile(cbcpost))[0]

pyfiles = []
for root, dirnames, filenames in os.walk(d):
  for filename in fnmatch.filter(filenames, '*.py'):
      pyfiles.append(os.path.join(root, filename))

@pytest.mark.parametrize("pyfile", pyfiles)
def test_header(pyfile):
    s = ["# Copyright (C) 2010-2014 Simula Research Laboratory",
         "#",
         "# This file is part of CBCPOST.",
         "#",
         "# CBCPOST is free software: you can redistribute it and/or modify",
         "# it under the terms of the GNU Lesser General Public License as published by",
         "# the Free Software Foundation, either version 3 of the License, or",
         "# (at your option) any later version.",
         "#",
         "# CBCPOST is distributed in the hope that it will be useful,",
         "# but WITHOUT ANY WARRANTY; without even the implied warranty of",
         "# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the",
         "# GNU Lesser General Public License for more details.",
         "#",
         "# You should have received a copy of the GNU Lesser General Public License",
         "# along with CBCPOST. If not, see <http://www.gnu.org/licenses/>.",
        ]
    
    s = '\n'.join(s) 
    
    text = open(pyfile, 'r').read()
    assert text.find(s) == 0, "%s does not have correct license header." %pyfile

    
def _recursive_getmembers(object, members, predicate=None):
    if object in members:
        return members
    if predicate != None and not predicate(object):
        return members

    members.append(object)
    for member in inspect.getmembers(object[1]):
        _recursive_getmembers(member, members, predicate)
    return members
    
def recursive_getmembers(object, predicate=None):
    return _recursive_getmembers((object.__name__, object), [], predicate)


def object_test(object):
    name, object = object
    if inspect.getmodule(object) == None:
        #print object, "Not accepted. No module."    
        return False
    elif not "cbcpost" in inspect.getmodule(object).__name__:
        #print object, "Not accepted. Wrong module name."    
        return False
    elif inspect.iscode(object):
        #print object, "Not accepted. Is code."
        return False
    elif name in ["__func__", "__self__", "im_func", "im_self", "im_class"]:
        #print object, "Not accepted. Bad name."
        return False
    #print object, "Accepted."
    return True

members = recursive_getmembers(cbcpost, object_test)

@pytest.mark.parametrize("member", [m for m in members
                                    if inspect.isclass(m[1])
                                    and m[0][0] != "_"                           
                                    ])
def test_class_docstring(member):
    name, mem = member
    assert mem.__doc__ != None

@pytest.mark.parametrize("member", [m for m in members
                                    if inspect.ismodule(m[1])                             
                                    ])
def test_module_docstring(member):
    name, mem = member
    assert mem.__doc__ != None


@pytest.mark.parametrize("member", [m for m in members
                                    if inspect.isfunction(m[1])
                                    and m[0][0] != "_"
                                    ])
def test_function_docstring(member):
    name, mem = member
    assert mem.__doc__ != None


@pytest.mark.parametrize("member", [m for m in members
                                    if inspect.ismethod(m[1])
                                    and (m[0][0] != "_"
                                         or m[0] in ["__call__"])
                                    and m[1].__self__ == None
                                    ])
def test_method_docstring(member):
    name, mem = member
    
    if hasattr(mem.im_class, "mro"):
        docs = [mem.__doc__]
        
        for base in mem.im_class.mro():
            docs.append(getattr(base, name, None).__doc__)
            
        #print docs
        assert not all([d==None for d in docs])
    else:
        assert mem.__doc__ != None
