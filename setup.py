# Installation script for python
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import subprocess
import os
import re
import sys

PACKAGE = "qibo"


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


class Build(_build_py):
    def run(self):
        commands = [
            ["make", "-j", "%s" % os.cpu_count(),
             "-C", "src/qibo/tensorflow/custom_operators/"],]
           #["make", "-C", "src/qibo/tensorflow/custom_functions/"]]
        for command in commands:
            if subprocess.call(command) != 0:
                sys.exit(-1)
        _build_py.run(self)


setup(
    name="qibo",
    version=get_version(),
    description="Quantum computing framework",
    author="TII-Qilimanjaro team",
    author_email="",
    url="https://github.com/Quantum-TII/qibo",
    packages=find_packages("src", exclude=["qibo/tensorflow/custom_functions"]),
    package_dir={"": "src"},
    cmdclass={"build_py": Build,},
    package_data={"": ['./**/*.so']},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=["numpy", "tensorflow>=2.1", "scipy", "cma", "cffi"],
    extras_require={
        "docs": ["sphinx_rtd_theme", "recommonmark", "sphinxcontrib-bibtex"]
    },
    python_requires=">=3.6",
    long_description="See readthedocs webpage with the documentation",
)
