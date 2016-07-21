#!/usr/bin/env python

import os, sys, platform
from setuptools import setup

# Version number
major = 2016
minor = 1
maintenance = 0

scripts = [
    os.path.join("scripts", "cbcbatch"),
    os.path.join("scripts", "cbcdashboard.ipynb"),
    ]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*' % os.path.split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)

with open("README", 'r') as file:
    readme = file.read()

setup(name = "cbcpost",
      version = "%d.%d.%d" % (major, minor, maintenance),
      description = "cbcpost -- Postprocessing framework from the Center of Biomedical Computing",
      long_description = readme,
      author = "Oeyvind Evju, Martin Sandve Alnaes",
      author_email = "cbcflow@simula.no",
      url = 'https://bitbucket.org/simula_cbc/cbcpost',
      classifiers = [
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      scripts = scripts,
      packages = ["cbcpost",
                  "cbcpost.fieldbases",
                  "cbcpost.metafields",
                  "cbcpost.utils",
                  ],
      package_dir = {"cbcpost": "cbcpost"},
    )

