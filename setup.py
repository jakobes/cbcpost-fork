#!/usr/bin/env python

import os, sys, platform
from setuptools import setup

# Version number
major = 1
minor = 3
maintenance = 0

with open("README", 'r') as file:
    readme = file.read()

setup(name = "cbcpost",
      version = "%d.%d.%d" % (major, minor, maintenance),
      description = "cbcpost -- Postprocessing framework from the Center of Biomedical Computing",
      long_description = readme,
      author = "Oyvind Evju, Martin Sandve Alnaes",
      author_email = "cbcpost@simula.no", 
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
      packages = ["cbcpost",
                  "cbcpost.fieldbases",
                  "cbcpost.metafields",
                  "cbcpost.utils",
                  ],
      package_dir = {"cbcpost": "cbcpost"},
    )

