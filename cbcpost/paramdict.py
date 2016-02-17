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
ParamDict is an extension to the standard python dict-type. It has support for
multilevel dicts, and access and assignment through dot-notation (similar to
accessing attributes).
"""

from __future__ import division

import re
import copy

class ParamDict(dict):
    "The base class extending the standard python dict."
    #_parsed_args = ParamDict() # To check which command line arguments are used

    def __init__(self, *args, **kwargs):
        dict.__init__(self)
        self._keys = []
        if not kwargs:
            kwargs = {}
        if args:
            arg, = args
            for item in arg:
                if isinstance(item, tuple):
                    k, v = item
                else:
                    k, v = item, arg[item]
                kwargs[k] = v

        self.update_recursive(kwargs)
        self._keys = sorted(set(self._keys) | set(kwargs))

    # --- Recursive ParamDict aware copy and update functions

    def to_dict(self):
        "Convert hierarchy of ParamDicts into a hierarchy of dicts recursively."
        keys = list(self.iterkeys())
        items = []
        for k in keys:
            v = self[k]
            if isinstance(v, ParamDict):
                v2 = v.to_dict()
            else:
                v2 = v
            items.append((k, v2))
        return dict(items)

    def copy_recursive(self):
        "Copy ParamDict hierarchy recursively, using copy.deepcopy() to copy \
        values."
        keys = list(self.iterkeys())
        items = []
        for k in keys:
            v = self[k]
            if isinstance(v, ParamDict):
                v2 = v.copy_recursive()
            else:
                v2 = copy.deepcopy(v)
            items.append((k, v2))
        return ParamDict(items)

    def replace_shallow(self, params=None, **kwparams):
        "Perform a shallow update where no new keys are allowed."
        if params:
            unknown = set(params.iterkeys()) - set(self.iterkeys())
            if unknown:
                raise RuntimeError("Trying to replace non-existing entries: %s"
                                   % (sorted(unknown),))
            for k, v in params.iteritems():
                self[k] = v
        if kwparams:
            self.replace_shallow(kwparams)
        # Allow use as 'foo(params.replace_shallow(foo,bar))'
        return self

    def replace_recursive(self, params=None, **kwparams):
        "Perform a recursive update where no new keys are allowed."
        def handle(k, v):
            if k not in self:
                raise RuntimeError("Trying to replace non-existing entry: %s"
                                   % (k,))
            if isinstance(v, ParamDict):
                # If it's a ParamDict, recurse
                self[k].replace_recursive(v)
            else:
                # Otherwise abort recursion
                self[k] = v
        if params:
            for k, v in params.iteritems():
                handle(k, v)
        for k, v in kwparams.iteritems():
            handle(k, v)
        # Allow use as 'foo(params.replace_recursive(foo,bar))'
        return self

    def update_shallow(self, params=None, **kwparams):
        "Perform a shallow update, allowing new keys to be introduced."
        if params:
            for k, v in params.iteritems():
                self[k] = v
        if kwparams:
            self.update_shallow(kwparams)
        # Allow use as 'foo(params.update_shallow(foo,bar))'
        return self

    def update_recursive(self, params=None, **kwparams):
        "Perform a recursive update, allowing new keys to be introduced."
        def handle(k, v):
            if isinstance(v, dict):
                # If it's a dict, convert to ParamDict and recurse
                pd = ParamDict()
                pd.update_recursive(v)
                self[k] = pd
            else:
                # Otherwise abort recursion
                self[k] = v
        if params:
            for k, v in params.iteritems():
                handle(k, v)
        for k, v in kwparams.iteritems():
            handle(k, v)
        # Allow use as 'foo(params.update_recursive(foo,bar))'
        return self

    # Default update and replace behaviour is recursive
    update = update_recursive
    replace = replace_recursive

    # --- Attribute access

    def __getitem__(self, name):
        return dict.__getitem__(self, name)

    def __setitem__(self, name, value):
        "Insert item with dict notation, allowing a new key to be added."
        if name not in self:
            assert(isinstance(name, str))
            self._keys.append(name)
        return dict.__setitem__(self, name, value)

    def __delitem__(self, name): # TODO: Add tests for this
        if name in self:
            assert(isinstance(name, str))
            self._keys.remove(name)
        return dict.__delitem__(self, name)

    def __getattr__(self, name):
        if name.startswith("_"):
            return self.__dict__[name]
        else:
            return self[name]

    def __setattr__(self, name, value):
        """Insert item with attribute notation, only allows changing a value
        with existing key.

        """
        if name.startswith("_"):
            self.__dict__[name] = value
        else:
            if name not in self:
                raise RuntimeError("Trying to update non-existing entry: %s"
                                   % (name,))
            self[name] = value

    def pop(self, name, default=None):
        """ Returns Paramdict[name] if the key exists. If the key does not
        exist the default value is returned. """
        if self.has_key(name):
            v = self[name]
            del self[name]
            return v
        else:
            return default

    # --- Pickling and shelving

    def __getstate__(self):
        return (self._keys, list(dict.iteritems(self)))

    def __setstate__(self, state):
        (self._keys, data) = state
        assert len(self._keys) == len(data)
        dict.clear(self)
        dict.update(self, data)

    # --- String rendering

    def __repr__(self):
        return "ParamDict([%s])" % ", ".join("(%r, %r)"
                % (k, self[k]) for k in self._keys)

    def __str__(self):
        "Return formatted string representing the dict"
        return '\n'.join(self._str())

    def _str(self, level=0):
        "Format dict string recursively"
        indent = (level+1)*4*" "
        if level == 0:
            lines = ["Parameters:"]
        else:
            lines = []

        for k in self._keys:
            v = self[k]
            if isinstance(v, ParamDict):
                lines.append("%s%s =" % (indent, k))
                lines.extend(v._str(level+1))
            else:
                lines.append("%s%s = %r" % (indent, k, v))

        return lines

    # TODO: Add .json format support!

    # --- Iteration

    def __iter__(self):
        return iter(self._keys)

    def iteritems(self):
        return ((k, self[k]) for k in self._keys)

    def items(self):
        return list(self.iteritems())

    def iterkeys(self):
        return iter(self._keys)

    def keys(self):
        return list(self._keys)

    def iterdeep(self):
        "Iterate recursively over all parameter items."
        for k, v in self.iteritems():
            if isinstance(v, ParamDict):
                for sk, sv in v.iterdeep():
                    yield ("%s.%s" % (k, sk), sv)
            else:
                yield (k, v)

    # --- Commandline argument translation
    # (should perhaps be placed outside class?)

    def arg_assign(self, name, value):
        "Assign value recursively to self"
        subs = name.split('.')
        subs, vname = subs[:-1], subs[-1]
        p = self
        for s in subs:
            p = p[s]
        if vname in p:
            #value = value.split(',')
            value = re.split("\s*,\s*",value)
            for i,v in enumerate(value):
                try:
                    value[i] = int(v)
                    continue
                except:
                    pass
                
                try:
                    value[i] = float(v)
                    continue
                except:
                    pass
                
                if v in ["True", "true"]:
                    value[i] = True
                elif v in ["False", "false"]:
                    value[i] = False
        
            if len(value) == 1:
                value = value[0]
            else:
                value = tuple(value)
    
            p[vname] = value

    def parse_args(self, args):
        "Parse command line arguments into self"
        #m = re.findall(r'([^ =]+)=([^ ]+)', args)
        #args = args.replace("")
        args = re.sub("\s+=", "=", args)
        args = re.sub("=\s+", "=", args)
        args = re.sub("[\]'\"()[]", "", args)
        
        m = re.findall(r'([^\s]*\S+)=(.+?(?=\s+\S+=|$))', args)
        
        for k, v in m:
            self.arg_assign(k, v)

    def render_args(self):
        "Render arguments (inverse of parse_args)"
        return "  ".join("%s=%r" % (k, v) for k, v in self.iterdeep())
