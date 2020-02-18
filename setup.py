# Installation script for python
from setuptools import setup, find_packages
import os
import re

PACKAGE = 'qibo'

def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join('src', PACKAGE, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)

setup(name='qibo',
      version=get_version(),
      description='Quantum computing framework',
      author = 'TII-Qilimanjaro team',
      author_email='',
      url='https://github.com/Quantum-TII/qibo',
      package_dir={'':'src'},
      packages=find_packages('src'),
      zip_safe=False,
      classifiers=[
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      install_requires=[
          'numpy',
          'tensorflow',
          ],
      extras_require={
          'docs' : [
            'sphinx_rtd_theme',
            'recommonmark',
            'sphinxcontrib-bibtex',
            ],
          },
      python_requires='>=3.6,<3.8',
      long_description='See readthedocs webpage with the documentation'
)
