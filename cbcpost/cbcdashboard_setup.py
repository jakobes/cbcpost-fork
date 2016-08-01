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

if __name__ == '__main__':
    print "This file is not runnable. To launch dashboard, run cbcdashboard."
    exit(1)
from IPython.display import *

#### Silence deprecation warnings in traitlets
# (for some reason needs to be set twice)
import warnings
def new_warn(*args, **kwargs):
    try:
        warnings.warn(*args, **kwargs)
    except:
        warnings.warn("Unable to suppress warning (%s %s)" %(str(args), str(kwargs)))
    
warnings.warn_explicit = new_warn
warnings.filterwarnings("ignore",category=DeprecationWarning)
import mpld3
mpld3.enable_notebook()
import warnings
warnings.warn_explicit = new_warn
warnings.filterwarnings("ignore",category=DeprecationWarning)

import pickle

from dolfin import *
set_log_level(20)
from cbcpost import *
from cbcpost.utils import Loadable, cbc_log
from IPython.core.display import display_html

import re
import numpy as np
try:
    import seaborn as _sns
except:
    pass
from matplotlib import pyplot as plt
try:
    import matplotlib.style
    matplotlib.style.use("seaborn-notebook")
except:
    pass
import pandas as pd
from ipywidgets.widgets import *
import os
import time
import threading

relwidth=0.9
casedir = os.environ["CBCDASHBOARD_CASEDIR"]
_documentwidth = int(os.environ["CBCDASHBOARD_WIDTH"])
_documentheight = int(os.environ["CBCDASHBOARD_HEIGHT"])
#cwd = None

def set_style(_relwidth=0.9):
    "Sets style of cbcdashboard"
    relwidth = _relwidth
    #### STYLING ####
    from IPython.display import display,Javascript, HTML
    
    display(HTML("""
    <style>.widget-subarea .widget-dropdown .widget_item .widget-combo-btn { display: stretch; width: 100%; white-space: normal !important;}
    </style>
    """))
    display(HTML("<style>.container { width:%d%%; }</style>" %(int(relwidth*100))))

def setup():
    "Setup dashboard"
    current_value = ParamDict()
    widgets = ParamDict()
    gui = None
    read_params = ""

    # Setup sizes    
    documentwidth = int(_documentwidth*relwidth)
    documentheight = int(_documentheight*relwidth)

    _width = [0.2,0.6,0.2]
    dpi = 80
    #pcwidth = [str(100*x)+"%" for x in _width]
    pxwidth = [str(documentwidth*x)+"px" for x in _width]
    inwidth = [documentwidth*x/dpi for x in _width]
    
    _height = [0.05,0.9,0.05]
    #pcheight = [str(100*x)+"%" for x in _height]
    pxheight = [str(documentheight*x)+"px" for x in _height]
    inheight = [documentheight*x/dpi for x in _height]

    # Create and populate gui container
    gui = VBox(width=pxwidth[1], height=pxwidth[1])
    rows = [HBox(width=str(documentwidth)+"px"),
        HBox(width=str(documentwidth)+"px"),
        HBox(width=str(documentwidth)+"px")]
    header = HTML("""<h1 style="background-color: #FFFFFF">cbcdashboard</h1>""")
    
    gui.children = [header]+rows
    
    containers = [[HBox(width=pxwidth[0], height=pxheight[0]),
                   HBox(width=pxwidth[1], height=pxheight[0]),
                   HBox(width=pxwidth[2], height=pxheight[0])],
                  [HBox(width=pxwidth[0], height=pxheight[1]),
                   HBox(width=pxwidth[1], height=pxheight[1]),
                   HBox(width=pxwidth[2], height=pxheight[1])],
                  [HBox(width=pxwidth[0], height=pxheight[2]),
                   HBox(width=pxwidth[1], height=pxheight[2]),
                   HBox(width=pxwidth[2], height=pxheight[2])],
                  ]
    for i in range(3):
        rows[i].children = containers[i]
    #    for c in containers[i]:
    #        c.children = [HTML("Loading...")]
    containers[2][0].children = (HTML("License: LGPLv3 or later"),)
    containers[2][2].children = (HTML("cbcpost version 2016.1"),)
    containers[0][0].children = (HTML("Directory: "+casedir),)
    containers[0][2].children = ()
    
    # Display loading gui
    display(gui)


    # Create variable containers
    coverage = dict()
    history = dict()
    read_params = ""
    
    params = ParamDict(
        updated_plot = False,
        current_plot = None,
        displayed_field = None,
        play = False,
        is_batch = False,
        single_mode=True,
        current_casedir = None,
    )
    
    fields = ParamDict(
        timedependent_function = None,
        constant_function = None,
        timedependent_float = None,
        constant_float = None,
        time = dict(),
    )
    
    current_value = ParamDict(
        value = None,
        #html = None,
        key = None,
        ts = None,
        x = None,
        type = None,
        legend = None,
        xlabel = None,
    )
    
    num_all = 0
    #current_casedir = None
    casedirs = dict()
    
    # Create containers for matplotlib and pandas
    matplotlib_figure = plt.figure(1, figsize=(inwidth[1],inheight[1]))
    matplotlib_widget = HTML(mpld3.fig_to_html(matplotlib_figure))
    
    pandas_widget = FlexBox(width=pxwidth[1], height=pxheight[1])
    pandas_widget.pack = "center"
    pandas_widget.align = "center"
    pandas_widget.children = (HTML(""),)
    pandas_df = pd.DataFrame()

   
    # Check is case directory is a batch directory
    pp = PostProcessor(dict(casedir=casedir))
    read_params = read_parameters(pp)
    params.is_batch = "BatchParams" in read_params
    
    
    # Set current case direcotry based on selections
    def set_current_casedir():
        "Set current case directory"
        num_all = np.count_nonzero([w.value == "all" for w in widgets.batch_params.children])
        if num_all == 0:
            params.current_casedir = casedirs[tuple([(w.description, w.value) for w in widgets.batch_params.children])]
        else:
            params.current_casedir = None
    
    def get_boundary():
        "Get boundary of 3D field (since only boundary is visualized in 3D mode)"
        key = widgets.available_fields.value
        ts = widgets.slider.value
        u = coverage[key][ts]._oldcall()
        bdry = get_boundary._precomputed.get(key)
        if bdry is None:
            bdry = Boundary(key)
            bdry.before_first_compute(lambda x: u)
        return bdry.compute(lambda x: u)
    get_boundary._precomputed = dict()
    
    
    def read_history(cd=None):
        "Read history from casedir cd"
        if cd is None:
            cd = casedir
        
        coverage[cd] = dict()
        cv = coverage[cd]
        
        pp = PostProcessor(dict(casedir=cd))
        replay = Replay(pp)
        history[cd] = replay._fetch_history()
        hist = history[cd]
            
        for ts in hist:
            for k,v in hist[ts].items():
                f = cv.setdefault(k, dict())
                if isinstance(v, Loadable) and v.function is not None:
                    if v.function.ufl_domain().topological_dimension() == 3:
                        v._oldcall = v.__call__
                        v.__call__ = get_boundary
                f[ts] = v
        
        constant_float = set()
        constant_function = set()
        timedependent_float = set()
        timedependent_function = set()
        for f in cv:
            if f == "t":
                fields.time[cd] = np.array(cv[f].values())
                continue
            is_constant = len(cv[f]) == 1       
            is_function = cv[f].values()[0].function is not None
            if is_constant and is_function:
                constant_function.add(f)
            elif is_constant and not is_function:
                constant_float.add(f)
            elif not is_constant and is_function:
                timedependent_function.add(f)
            elif not is_constant and not is_function:
                timedependent_float.add(f)
    
        if fields.constant_float is None and len(constant_float)>0:
            fields.constant_float = list(constant_float)
        else:
            fields.constant_float = list(set(fields.constant_float).intersection(constant_float))
        
        if fields.timedependent_float is None and len(timedependent_float)>0:
            fields.timedependent_float = timedependent_float
        else:
            fields.timedependent_float = list(set(fields.timedependent_float).intersection(timedependent_float))
        
        if fields.constant_function is None and len(constant_function)>0:
            fields.constant_function = list(constant_function)
        else:
            fields.constant_function = list(set(fields.constant_function).intersection(constant_function))
        
        if fields.timedependent_function is None and len(timedependent_function)>0:
            fields.timedependent_function = list(timedependent_function)
        else:
            fields.timedependent_function = list(set(fields.timedependent_function).intersection(timedependent_function))
    
    # Read back casedir(s)    
    if params.is_batch:
        batch_params = read_params["Params"]
    
        for d in os.listdir(casedir):
            subcasedir = os.path.join(casedir,d)
            subparams = read_parameters(PostProcessor(dict(casedir=subcasedir)))
            subparams = extract_batch_params(subparams, batch_params)
            key = []
            if not all(k in subparams for k in batch_params):
                continue
            
            for k in batch_params:
                assert k in subparams
                
                key.append((k, subparams[k]))
            key = tuple(key)
            casedirs[key] = subcasedir
    else:
        casedirs[()] = casedir
    
    N = len(casedirs.values())
    fb = FlexBox(width=pxwidth[1], height=pxheight[0])
    fb.pack = "center"
    fb.align = "center"
    desc = HTML("""<p style="font-align: center"> Loading casedirs 0%% (0/%d)""" %N)
    #desc.align_self = "center"
    progress = FloatProgress(min=0, max=N)
    fb.children = [desc, progress]
    containers[0][1].children = (fb,)
    for i, cd in enumerate(casedirs.values()):
        read_history(cd)
        desc.value = """<p style="font-align: center"> Loading casedirs %d%% (%d/%d)""" %(int((i+1)*100./N), i+1, N)
        progress.value = i
            
    def fix_html(s, name):
        """x3dom.js has some bug that causes infinite recursion.
        A fixed version is moved to the UiO home area, at "http://folk.uio.no/oyvinev/x3dom-full.js"""
        S = re.sub(r"http://www.x3dom.org/download/x3dom.js", "http://folk.uio.no/oyvinev/x3dom-full.js", s, count=1)
        S = re.sub(r"<x3d", r'<x3d id="%s"' %name, S, count=1)
        S = re.sub('width="500.000000px" height="400.000000px"', 'width="%s" height="%s"' %(pxwidth[1], pxheight[1]), S, count=1)
        return S

    def compute_colors(u, p, is_cell_based=False):
        """To avoid re-reading from dolfin.X3DOM, recomputes only colors for functions
        More or less taken directly from X3DOM.cpp in dolfin"""
        vv = u.compute_vertex_values()
        mesh = u.function_space().mesh()
        dim = u.ufl_domain().topological_dimension()
        if not is_cell_based:
            # Compute or get vertex set
            vertex_set = compute_colors._vertex_sets.get(u.id())
            if vertex_set is None:
                vertex_set = set()
                for f in cells(mesh):
                    for v in vertices(f):
                        vertex_set.add(v.index())
                compute_colors._vertex_sets[u.id()] = vertex_set
            
            # Compute magnitude of function, since vectors are not supported in X3Dom
            if len(u.ufl_shape) == 1:
                N =len(vv)/dim
                _vv = vv[:N]**2
                for i in range(1,dim):
                    _vv += vv[i*N:(i+1)*N]**2
                vv = np.sqrt(_vv)
            value_data = np.zeros(len(vertex_set))#[None]*len(vertex_set)
            for i, v in enumerate(vertex_set):
                value_data[i] = vv[v]
    
        else:
            facet_set = compute_colors._facet_sets.get(u.id())
            if facet_set is None:
                if dim == 3:
                    iterator = facets
                    num_facets = mesh.num_facets()
                else:
                    iterator = cells
                    num_facets = mesh.num_cells()
    
                facet_set = np.array(np.zeros((num_facets,dim+1)))
                for f in iterator(mesh):
                    facet_set[f.index(),:] = f.entities(0)
                facet_set = facet_set.astype(int)
                compute_colors._facet_sets[u.id] = facet_set
            values = vv[facet_set]
            value_data = np.mean(values, axis=1)
        

        # Compute new value range, and set if value_range is set to default
        value_min = np.min(value_data)
        value_max = np.max(value_data)

        if widgets.value_range.default is True:
            widgets.value_range.unobserve_all()
            widgets.value_range.min = value_min
            widgets.value_range.max = value_max
            widgets.value_range.value = (value_min, value_max)
            widgets.value_range.step = (value_max-value_min)/200.
            widgets.value_range.observe(widgets.value_range.handler, "value")
        
        value_min, value_max = widgets.value_range.value       
    
        # Compute new colors
        if value_max == value_min:
            scale = 1.0
        else:
            scale = 255.0/(value_max-value_min)
        cindex = (scale*np.abs(value_data - value_min)).astype(int)
        cindex[cindex<0] = 0
        cindex[cindex>255] = 255
        cmap = p.get_color_map().reshape(-1,3)
        colors = cmap[cindex].flatten()
        return colors
    compute_colors._vertex_sets = dict() # Cache for storing vertex sets
    compute_colors._facet_sets = dict() # Cache for storing facet sets
    #compute_colors._edges = dict()
    
    # Insert colors into HTML-representation of x3d-object
    def insert_colors(name, colors):
        """Insert colors into HTML-representation of x3d-object.
        This is done through a simple javascript."""
        colors_str = " ".join(["%.3g" %c for c in colors])

        javascript= """
        var elems = document.getElementsByTagName("x3d")
        var i = 0
        for ( i=0; i<elems.length; ++i) {
            if (elems[i].id == "%s") {
                break;
            }
        }
        try{
            elems[i].getElementsByTagName("color")[0].color = "%s"
        } catch(err) {} """  %(name, colors_str)
        #document.getElementById("%s").color = "%s";"""  %(name,name,name,colors_str)
        display(Javascript(javascript))
        return
    
    def get_combinations():
        "Get combinations based on selected batch params"
        combinations = []
        for ckey, cd in casedirs.iteritems():
            _ckey = ParamDict(ckey)
            
            if all([w.value == _ckey[w.description] or w.value == "all" for w in widgets.batch_params.children]):
                combinations.append(ckey)
        return combinations
    
    def compute_current_value():
        "Compute the currently selected value"
        key = widgets.available_fields.value
        for k in current_value:
            current_value[k] = None
        
        current_value.key = key
    
        children = widgets.batch_params.children
        which_all = dict((w.description,batch_params[w.description]) for w in children if w.value == "all")
        num_all = len(which_all)
        combinations = get_combinations()
        cds = [casedirs[c] for c in combinations]
    
        if key in fields.constant_float and num_all == 0:
            current_value.type = "float"
            current_value.value = coverage[params.current_casedir][key].values()[0]()
    
        elif key in fields.constant_function+fields.timedependent_function:
            current_value.type = "function"
            ts = widgets.slider.value
            current_value.ts = ts
            t = coverage[params.current_casedir]["t"][ts]
            current_value.x = t
            
            u = coverage[params.current_casedir][key][ts]()
            current_value.value = u
    
        elif num_all == 2 and key in fields.constant_float:
            current_value.type = "pandas"
            import pandas as pd
            cname, columns = which_all.items()[0]
            rname, index = which_all.items()[1]
            ##from IPython import embed
            #embed()
            value_data = np.zeros((len(batch_params[rname]),len(batch_params[cname])))
    
            for ci, c in enumerate(columns):
                for ri, r in enumerate(index):
                    for comb, cd in zip(combinations, cds):
                        _comb = ParamDict(comb)
    
                        if _comb[cname] == c and _comb[rname] == r:
                            value_data[ri,ci] = coverage[cd][key].values()[0]()
            
            df = pd.DataFrame(data=value_data,columns=columns, index=index)
            df.columns.name = "%s\%s" %tuple(which_all.keys()[::-1])
            
            current_value.value = df
        else:
            current_value.type = "matplotlib"
            if key in fields.timedependent_float:
                values = [[]]*len(combinations)
                x = [[]]*len(combinations)
                legend = [""]*len(combinations)
                for i, cd in enumerate(cds):
                    comb = ParamDict(combinations[i])
                    #_cd = dict(cd)
                    timesteps = coverage[cd][key].keys()
                    x[i] = np.array(fields.time[cd][timesteps])
                    _values = coverage[cd][key].values()
                    _values = [_v() for _v in _values]
                    legend[i] = ", ".join("%s=%s" %(k,v) for k,v in dict(combinations[i]).items() if k in which_all)
    
                    values[i] = _values
                current_value.legend = legend
                current_value.xlabel = "Time"
                current_value.x = x
            elif key in fields.constant_float:
                x = [np.array(which_all.values()[0])]
                values = [0.0]*len(combinations)
                for i, cd in enumerate(cds):
                    values[i] = coverage[cd][key].values()[0]()
                values = [values]
                current_value.xlabel = which_all.keys()[0]
            current_value.x = x
            current_value.value = values


    #################################
    # Plot fields
    def plot_function_field():
        params.updated_plot = False
        key = current_value.key
        u = current_value.value
    
        p = X3DOMParameters()
        p.set_representation(p.Representation_surface_with_edges)
        
        dm = u.function_space().dofmap()
        rank = len(u.ufl_shape)
        dim = u.ufl_domain().topological_dimension()
        is_cell_based = dm.max_element_dofs() == dim**rank
    
        if widgets[params.current_casedir]["html"][key] is None:
            s = fix_html(X3DOM.html(u, p), key)
            m = re.findall('<shape>.*?</shape>', s, flags=re.DOTALL)
            if len(m) == 2:
                if "indexedLineSet" in m[0]:
                    m = [m[1],m[0]]
                assert "indexedLineSet" in m[1]
                plot_function_field._edges[u.id()] = m[1]
                s = re.sub(m[1], "<!-- INDEXED LINE SET GOES HERE -->", s)
            plot_function_field._html[u.id()] = s
            widgets[params.current_casedir]["html"][key] = HTML(s)
        s = plot_function_field._html[u.id()]
        m = plot_function_field._edges.get(u.id())
        if widgets.edge_cb.value and m is not None:
            s = re.sub("<!-- INDEXED LINE SET GOES HERE -->", m, s)
        widgets[params.current_casedir]["html"][key].value = s
        widgets.container.children = [widgets[params.current_casedir]["html"][key]]
    
        colors = compute_colors(u,p, is_cell_based)
        insert_colors(key, colors)
        params.displayed_field = key
        params.updated_plot = True
    plot_function_field._edges = dict()
    plot_function_field._html = dict()
            
        
        
        
    def plot_float_field():
        value = current_value.value
        key = current_value.key
        html = widgets[params.current_casedir]["html"].get(key)
        html_str = """
                <p style="font-size: 300%%; text-align: center; line-height: %s"> %s
                """ %(pxheight[1], str(value))
        if html is None:
            html = HTML(html_str, width=pxwidth[1], height=pxheight[1])

        html.value = html_str
        widgets.container.children = [html] 
    
    def plot_matplotlib_field(values=None, x=None, legend=None):
        # Cannot use matplotlib inline backend, changing it temporarily
        #backend = plt.get_backend()
        #print "Backend: ", backend
        #plt.switch_backend(u"TkAgg")
        matplotlib_figure.clear()
        plt.figure(matplotlib_figure.number)
        
        key = current_value.key
        x = current_value.x
        value = current_value.value
        legend = current_value.legend
        xlabel = current_value.xlabel
        #plot_type = widgets.plot_type.value
        #print "Plot type: ", plot_type
        for _x,v in zip(x, value):
            plt.plot(_x,v)
            """
            if plot_type == "plot":
                plt.plot(_x,v)
            else:
                base = widgets.log_scale.value
                print "Base: ", base
                if plot_type == "semilogx":
                    plt.semilogx(_x,v, basex=base)
                elif plot_type == "semilogy":
                    plt.semilogy(_x,v, basey=base)
                elif plot_type == "loglog":
                    plt.loglog(_x,v, basex=base, basey=base)
                print _x, v
            """
        if legend is not None and not (len(legend) == 1 and legend[0] == ""):
            plt.legend(legend)
        
        #plt.draw()
        
        plt.xlabel(xlabel)
        plt.ylabel(key)
        plt.tight_layout()
        
        plt.draw()
        #plt.show()
        html = mpld3.fig_to_html(matplotlib_figure)
        #html = mpld3.fig_to_d3(plt.gcf())
        html = html.replace('"text": "None"', '"text": ""')
    
        matplotlib_widget.value = html
        widgets.container.children = [matplotlib_widget]
        
        #plt.switch_backend(backend)
    
    def plot_pandas_field():
        value = current_value.value
        
        styles = [
        dict(selector="table", props=[#("width", "500px"),
                                     ("border-collapse", "collapse"),
                                     ]),
        dict(selector="th", props=[("font-size", "36px"),
                                   ("text-align", "center"),
                                   ("height", "100%"),
                                   ("background-color", "#D7D7D7"),
                                   ("min-width", "150px"),
                                   ("padding", "12px")
                                  ]),                               
        dict(selector="tr", props=[("text-align", "right"), ("padding", "12px"), ("font-size", "28px")])
    
        ]
        mat = value.as_matrix()#/100000
        shape = mat.shape
        mat = mat.flatten()
        mat2 = mat.astype(str)
        precision = 4
        def determine_formatting(df):
            if np.min(df.as_matrix()) < 10**-(precision-2):
                return lambda x: ("%%.%de" %(precision)) %x
            else:
                
                def _format_float(x):
                    s = "%f" %x
                    if "." in s:
                        s = s[:max(s.index('.'),precision+1)]
                    if s[-1] == ".":
                        s = s[:-1]
                    return s
            
                return _format_float
        format_float = determine_formatting(value)
    
        for i in range(len(mat)):
            mat2[i] = format_float(mat[i])
        mat2 = mat2.reshape(shape)
        global pandas_df
        pandas_df = pd.DataFrame(data=mat2, columns=value.columns, index=value.index)
    
        style = pandas_df.style.set_table_styles(styles)
        html = re.sub('<th class="blank">', '<th class="blank"> %s' %value.columns.name, style._repr_html_(), count=1)
        
        pandas_widget.children[0].value = html
        
        widgets.container.children = [pandas_widget]
    
        
    def plot_field():
        "Plot field. This method is called on every scene change in the gui"
        compute_current_value()
        type = current_value.type
        if type == "function":
            cbc_log(20, "Plotting function field")
            widgets.value_range.visible = True
            widgets.edge_cb.visible = True
            widgets.slider.visible = True
            widgets.play_pause_button.visible = True
            #widgets.plot_type.visible = False
            #widgets.log_scale.visible = False
            
            # Bug if these are added on first render, without function plot,
            # they are still shown. Therefore, adding them here
            c = list(containers[2][1].children)
            if widgets.value_range not in c:
                c.append(widgets.value_range)
            if widgets.edge_cb not in c:
                c.append(widgets.edge_cb)
            containers[2][1].children = tuple(c)
            #widgets.time.visible = True
            if current_value.key not in fields.timedependent_function:
                widgets.slider.disabled = True
                widgets.play_pause_button.disabled = True
            else:
                widgets.slider.disabled = False
                widgets.play_pause_button.disabled = False
            plot_function_field()
        elif type == "matplotlib":
            cbc_log(20, "Plotting matplotlib field")
            widgets.value_range.visible = False
            widgets.slider.visible = False
            widgets.play_pause_button.visible = False
            widgets.edge_cb.visible = False
            """
            c = list(containers[2][1].children)
            if not widgets.plot_type in c:
                c.append(widgets.plot_type)
            if not widgets.log_scale in c:
                c.append(widgets.log_scale)
            containers[2][1].children = tuple(c)
            
            if widgets.plot_type.value == "plot":
                widgets.log_scale.visible = False
            else:
                widgets.log_scale.visible = True
            widgets.plot_type.visible = True
            """
            #if current_value.xlabel == "Time":
            #    widgets.time.visible = False
            plot_matplotlib_field()
        elif type == "float":
            cbc_log(20, "Plotting float field")
            widgets.value_range.visible = False
            widgets.edge_cb.visible = False
            widgets.slider.visible = False
            widgets.play_pause_button.visible = False
            #widgets.plot_type.visible = False
            #widgets.log_scale.visible = False
            #widgets.time.visible = True
            plot_float_field()
        elif type == "pandas":
            cbc_log(20, "Plotting pandas field")
            widgets.value_range.visible = False
            widgets.edge_cb.visible = False
            widgets.slider.visible = False
            widgets.play_pause_button.visible = False
            #widgets.plot_type.visible = False
            #widgets.log_scale.visible = False
            #widgets.time.visible = True
            plot_pandas_field()
        else:
            raise RuntimeError("Unable to recognize field type %s" %type)
    
    ###################################
    # Event handlers
    def change_batch_param(change):
        "Event handler for when a batch param is changed"
        #new = change["new"]
        old = change["old"]
        dd = change["owner"]
        
        global num_all
        num_all = np.count_nonzero([c.value == "all" for c in widgets.batch_params.children])
    
        if num_all > 0:
            widgets.slider.disable = True
        else:
            widgets.slider.disable = False
            widgets.slider.value = widgets.slider.value
        set_current_casedir()
        if num_all > 2:
            dd.value = old
        else:
            setup_available_fields()
            plot_field()
    
    def snap_to_timestep(change):
        "Set timesteps to timestep slider"
        cv = coverage.get(params.current_casedir)
        if cv is None:
            slider.disabled = True
            return
        else:
            slider.disabled = False
            key = widgets.available_fields.value
            try:
                ts = change["new"]
                old = change["old"]
            except:
                ts = change.value
                old = ts+1
    
            if ts not in cv[key]:
                possible_ts = np.array(cv[key].keys())
                if widgets.play_pause_button.play:
                    possible_ts = possible_ts[(possible_ts-ts)>0]
                idx = np.argmin(np.abs(possible_ts-ts))
                new_ts = possible_ts[idx]
                slider.value = new_ts
            if slider.value == old:
                return
            else:
                plot_field()
    
    def field_changed(changed):
        "Handle event where selected field changed"
        new = changed["new"]
        old = changed["old"]
        widgets.value_range.default = True
        cv = coverage.get(params.current_casedir)
        if cv is None:
            slider.disabled = True
            plot_field()
            return
        slider.disabled = False
        if new != old:
            
            min_ts, max_ts, step_ts, valid = get_timestepping(new)
            cbc_log(20, "In field-changed: Time-step range=%d-%d" %(min_ts, max_ts))
            widgets.slider.min = -1e16
            widgets.slider.max = 1e16
            widgets.slider.min = min_ts
            widgets.slider.max = max_ts
            widgets.slider.step = step_ts
            #value_range.value="-"
            #value_range.value = None
            if not valid:
                widgets.slider.value = widgets.slider.value
            plot_field()
    
    def value_range_changed(change):
        "Handle value range changed"
        vr = widgets.value_range
        cbc_log(20, "In value range changed: change="+str(change))
        cbc_log(20, "In value range changed: default="+str(vr.default))
        cbc_log(20, "In value range changed: visible="+str(vr.visible))
        cbc_log(20, "In value range changed: value="+str(vr.value))
        r = vr.max-vr.min
        cbc_log(20, "In value range changed: range="+str(r))
        min,max = vr.value
        if r<1e-12 or (abs(min-vr.min)/r<1e-4 and abs(max-vr.max)/r<1e-4):
            vr.default = True
        else:
            if r<1e-12:
                vr.default = True
            else:
                vr.default = False
        cbc_log(20, "In value range changed (after): default="+str(vr.default))
        plot_field()
        
            
    ###########################
    # Helper functions
    def setup_available_fields():
        num_all = np.count_nonzero([c.value == "all" for c in widgets.batch_params.children])
        af = widgets.available_fields
        af.options = []
        current = af.value
        if num_all == 0:
            for k in fields:
                if k == "time":
                    continue
                af.options += fields[k]
                
            af.options = list(sorted(af.options))
        else:
            for k in fields:
                if "float" in k:
                    af.options += fields[k]
        
        if current not in af.options:
            if len(af.options) > 0:
                af.value = available_fields.options[0]
        cbc_log(20, "In setup_available_fields: af.options="+str(af.options))
        cbc_log(20, "In setup_available_fields: af.value="+str(af.value))
    
    def get_timestepping(key):
        cv = coverage[params.current_casedir]
        possible_ts = np.array(cv[key].keys())
        if len(possible_ts) > 1:
            possible_ts.sort()
            step = np.min(possible_ts[1:]-possible_ts[:-1])
        else:
            step = 1
        min_ts = np.min(possible_ts)
        max_ts = np.max(possible_ts)
        return min_ts, max_ts, step, slider.value in possible_ts

    def play(btn):
        while btn.play and slider.value<slider.max:
            params.updated_plot = False
            slider.value += 1
            time.sleep(0.1)
            while not params.updated_plot:
                time.sleep(0.03)
        btn.description = "Play"
        btn.tooltip = "Play sequence"
        btn.play = False
    
    def play_pause(btn):
        if btn.play:
            btn.description = "Play"
            btn.tooltip = "Play sequence"
            btn.play = False
        else:
            btn.description = "Pause"
            btn.tooltip = "Pause playback"
            btn.play = True
            t = threading.Thread(target=play, args=(play_pause_button,))    
            t.start()
    
    def to_html(s, fontsize=None):
        if not isinstance(s, str):
            s = str(s)
        if fontsize is not None:
            _s = '<pre style="font-size: %dpx">' %fontsize
            s =_s + s#+"<\pre>"
        else:
            s = '<pre>' + s
        return s   
    
    # Create all widgets
    batch_params_container = VBox(width=pxwidth[2], height=pxheight[1])
    if params.is_batch:
        batch_params_container.visible = False
        children = []
        for k,v in batch_params.items():
            w = Dropdown(description=k,
                        options=["all"]+list(v),
                        value=v[0],
                        width=str(float(pxwidth[2][:-2])-100)+"px",
                         )
            w.observe(change_batch_param, names="value")
            w.handler = change_batch_param
            children.append(w)
        batch_params_container.children = children
    
    batch_params_container.handler = None
    widgets["batch_params"] = batch_params_container
    set_current_casedir()
    
    
    # Define slider for timesteps
    slider = IntSlider(min=0, max=1, step=1, value=0, height=pxheight[0])
    slider.handler = snap_to_timestep    
    slider.observe(snap_to_timestep, names='value')
    slider.on_displayed(snap_to_timestep)
    widgets["slider"] = slider
    
    # Define drop down list of available fields
    available_fields = Dropdown(
        width=str(float(pxwidth[1][:-2])/3)+"px",
        font_size=64,
    )
    
    available_fields.handler = field_changed
    available_fields.observe(field_changed, names="value")
    widgets["available_fields"] = available_fields

    # Define value range slider  
    value_range = FloatRangeSlider()
    value_range.handler = value_range_changed
    value_range.observe(value_range_changed,"value")
    value_range.default = False
    widgets["value_range"] = value_range

    # Define play button
    
    play_pause_button = Button(description="Play", width="80px", height=str(float(pxheight[0][:-2])-7)+"px")
    play_pause_button.play = False
    
    play_pause_button.on_click(play_pause)
    play_pause_button.handler = play_pause
    widgets["play_pause_button"] = play_pause_button

    widgets["parameters"] = HTML(to_html(read_params, fontsize=11), width=pxwidth[0], height=pxheight[1])
    widgets.parameters.handler = None
    widgets.parameters.layout.overflow_y = "scroll"
    widgets.parameters.layout.overflow_x = "scroll"
    container = Box(height=pxheight[1], width=pxwidth[1])
    widgets["container"] = container
    widgets.container.handler = None
    for c in casedirs.values():
        widgets[c] = dict()
        widgets[c]["html"] = dict()
        for k in fields.timedependent_function+fields.constant_function:
            widgets[c]["html"][k] = None

    def edge_cb_changed(change):
        plot_field()
    edge_cb = Checkbox(description="Edges", value=False)
    edge_cb.handler = edge_cb_changed
    edge_cb.observe(edge_cb_changed, names="value")
    widgets["edge_cb"] = edge_cb
    """
    # TODO: Log scale does not seem to work in mpld3
    def plot_type_changed(change):
        if change["new"] == "plot":
            widgets.log_scale.visible = False
        else:
            widgets.log_scale.visible = True
        plot_field()
    
    plot_type = Dropdown(description="Plot type",
                          options=["plot", "loglog", "semilogx", "semilogy"],
                          value = "plot"
    )
    plot_type.observe(plot_type_changed, names="value")
    widgets["plot_type"] = plot_type
    
    log_scale = FloatText(description="Log base", value=10.0)
    
    def log_value_wait(change):
        time.sleep(1)
        f = change["owner"]
        if f.value != change["new"]:
            return
        plot_field()

    def log_value_change(change):
        t = threading.Thread(target=wait, args=(change,))
        t.daemon = True
        t.start()   

    log_scale.observe(log_value_change, names="value")
    widgets["log_scale"] = log_scale
    """
    # Populate gui with newly created widget    
    containers[0][1].children = (widgets.available_fields,widgets.play_pause_button,widgets.slider)#, widgets.time)
    containers[1][0].children = (widgets.parameters,)
    containers[1][1].children = (widgets.container,)
    containers[1][2].children = (widgets.batch_params,)
    
    # Bug that shows children even when display=False. Set them when required first time.
    #containers[2][1].children = (widgets.value_range,widgets.edge_cb)
    containers[2][1].children = ()
    
    setup_available_fields()
    value_range.value = (value_range.min, value_range.max)
    
    figures = dict(pandas=pandas_df, matplotlib=matplotlib_figure)

    return current_value, widgets, figures, gui

def read_parameters(pp):
    "Read parameters from postprocessor"
    read_params = ParamDict()
    try:
        read_params = pickle.loads(open(os.path.join(pp.get_casedir(), "params.pickle"), 'r').read())
    except:
        _read_params = "Unable to \nload parameters"
        read_params.__str__ = lambda: _read_params
    return read_params

# Extract the parameters which are run in batch mode
def extract_batch_params(params, batch_params):
    "Extract parameters from batch case directory"
    sub_params = ParamDict()
    for k,v in params.iterdeep():
        k = k.split('.')[-1]
        if k in batch_params:
            sub_params[k] = v
    return sub_params
