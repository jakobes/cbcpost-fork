# -*- coding: utf-8 -*-
#
# cbcpost documentation build configuration file, created by
# sphinx-quickstart on Tue Feb  4 08:51:54 2014.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os
class Mock(object):

    __all__ = []

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        elif name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        else:
            return Mock()

# No need to actually import these, just avoid import errors when using sphinx autodoc/autoclass/autmodule
MOCK_MODULES = ['dolfin', 'ufl', 'cbcpost.dol', 'numpy', 'scipy', 'scipy.interpolate',
                'scipy.integrate', 'scipy.special', 'matplotlib', 'matplotlib.pyplot',
                'scipy.spatial', 'scipy.spatial.ckdtree']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

if not os.path.isdir('_static'): os.mkdir('_static')

# on_rtd is whether we are on readthedocs.org
# Insert cwd in path to be able to import from generate_api_doc
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd: sys.path.insert(0, os.getcwd())

# Insert .. to import cbcpost
sys.path.insert(0, os.path.abspath('..'))

# Generate all rst files
from generate_api_doc import generate_dolfin_doc
generate_dolfin_doc("..", "cbcpost")

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.mathjax', 'sphinx.ext.inheritance_diagram', 'sphinx.ext.graphviz']

inheritance_graph_attrs = dict(rankdir="LR", size='"6.0, 8.0"',
                                     fontsize=14, ratio='compress')
inheritance_node_attrs = dict(shape='ellipse', fontsize=14, height=0.75,
                                )

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'cbcpost'
copyright = u'2014, Martin Alnaes and Oeyvind Evju'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.3'
# The full version, including alpha/beta/rc tags.
release = '1.3.0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ["cbcpost."]

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'default'
html_theme = 'agogo_mod'


headerbg = """
background: {0}; /* Old browsers */
background: -moz-linear-gradient(top, {0} 0%, {1} 44%, {2} 100%); /* FF3.6+ */
background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,{0}), color-stop(44%,{1}), color-stop(100%,{2})); /* Chrome,Safari4+ */
background: -webkit-linear-gradient(top, {0} 0%,{1} 44%,{2} 100%); /* Chrome10+,Safari5.1+ */
background: -o-linear-gradient(top, {0} 0%,{1} 44%,{2} 100%); /* Opera 11.10+ */
background: -ms-linear-gradient(top, {0} 0%,{1} 44%,{2} 100%); /* IE10+ */
background: linear-gradient(to bottom, {0} 0%,{1} 44%,{2} 100%); /* W3C */
filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='{0}', endColorstr='{2}',GradientType=0 ); /* IE6-9 */
""".format("#aa1133", "#8f0222", "#6d0019")


def swap(text, ch1, ch2):
    text = text.replace(ch2, '!',)
    text = text.replace(ch1, ch2)
    text = text.replace('!', ch1)
    return text
  
#footerbg = swap(headerbg, "top", "bottom")
footerbg = headerbg
headerbg = swap(headerbg, "top", "bottom")


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "pagewidth": "90em",
    "documentwidth": "70em",
    "textalign": "left",
    "headerbg": headerbg,
    "footerbg": footerbg,
    "headercolor1": "#999999",
    "headercolor2": "#999999",
    "headerlinkcolor": "#FFFFFF",
    "bodyfont": "helvetica",
    "headerfont": "helvetica",
    }
"""
agogo – A theme created by Andi Albrecht. The following options are supported:

    bodyfont (CSS font family): Font for normal text.
    headerfont (CSS font family): Font for headings.
    pagewidth (CSS length): Width of the page content, default 70em.
    documentwidth (CSS length): Width of the document (without sidebar), default 50em.
    sidebarwidth (CSS length): Width of the sidebar, default 20em.
    bgcolor (CSS color): Background color.
    headerbg (CSS value for “background”): background for the header area, default a grayish gradient.
    footerbg (CSS value for “background”): background for the footer area, default a light gray gradient.
    linkcolor (CSS color): Body link color.
    headercolor1, headercolor2 (CSS color): colors for <h1> and <h2> headings.
    headerlinkcolor (CSS color): Color for the backreference link in headings.
    textalign (CSS text-align value): Text alignment for the body, default is justify.
"""


# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['.']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'cbcpostdoc'


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'cbcpost.tex', u'cbcpost Documentation',
   u'Martin Alnaes and Oeyvind Evju', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'cbcpost', u'cbcpost Documentation',
     [u'Martin Alnaes and Oeyvind Evju'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'cbcpost', u'cbcpost Documentation',
   u'Martin Alnaes and Oeyvind Evju', 'cbcpost', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False
