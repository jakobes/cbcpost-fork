
from cbcflow.utils.common import (cbcflow_warning, hdf5_link, safe_mkdir, on_master_process,
                                  in_serial, Timer, cbcflow_log)

# Copyright (C) 2010-2014 Simula Research Laboratory
#
# This file is part of CBCFLOW.
#
# CBCFLOW is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CBCFLOW is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CBCFLOW. If not, see <http://www.gnu.org/licenses/>.
import keyword

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