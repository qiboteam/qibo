# Installation script for python
from setuptools import setup, find_packages
import os
import re

PACKAGE = "qibo"


# Returns the qibo version
def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]


# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# read backend versions
BACKENDFILE = os.path.join("src", PACKAGE, "backends", "__init__.py")
with open(BACKENDFILE, 'r') as f:
    content = f.readlines()
    for line in content:
        if 'TF_MIN_VERSION' in line:
            TF_MIN_VERSION = str(line.split()[2].replace("'", ""))
            break


setup(
    name="qibo",
    version=get_version(),
    description="A framework for quantum computing with hardware acceleration.",
    author="The Qibo team",
    author_email="",
    url="https://github.com/qiboteam/qibo",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.out", "*.yml"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=requirements,
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme", "recommonmark", "sphinxcontrib-bibtex", "sphinx_markdown_tables", "nbsphinx", "IPython"],
        "tests": ["pytest", "cirq", "ply", "sklearn"],
        # Backends dependencies
        "qibotf": ["qibotf"],
        "qibojit": ["qibojit"],
        "tensorflow": [f"tensorflow>={TF_MIN_VERSION}"],
    },
    python_requires=">=3.6.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
