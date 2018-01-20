# -*- coding: utf-8 -*-

from setuptools import find_packages
from numpy.distutils.core import setup, Extension

VERSION = "1.4.1"
DISTNAME = "fteikpy"
DESCRIPTION = "FTeikPy"
LONG_DESCRIPTION = """FTeikPy is a Python module that computes accurate first arrival traveltimes in 2-D and 3-D heterogeneous isotropic velocity model."""
AUTHOR = "Mark NOBLE, Keurfon LUU"
AUTHOR_EMAIL = "mark.noble@mines-paristech.fr, keurfon.luu@mines-paristech.fr"
URL = "https://github.com/keurfonluu/fteikpy"
LICENSE = "MIT License"
REQUIREMENTS = [
    "numpy",
    "matplotlib",
]
CLASSIFIERS = [
    "Programming Language :: Python",
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
]

FFLAGS = "-O3 -ffast-math -funroll-loops -fno-protect-parens -fopenmp"

ext1 = Extension(
    name = "fteikpy._fteik2d",
    sources = ["fteikpy/f90/fteik2d.f90"],
    extra_f90_compile_args = FFLAGS.split(),
    extra_link_args = [ "-lgomp" ],
    )

ext2 = Extension(
    name = "fteikpy._fteik3d",
    sources = ["fteikpy/f90/fteik3d.f90"],
    extra_f90_compile_args = FFLAGS.split(),
    extra_link_args = [ "-lgomp" ],
    )

ext3 = Extension(
    name = "fteikpy._lay2vel",
    sources = ["fteikpy/f90/lay2vel.f90"],
    extra_f90_compile_args = FFLAGS.split(),
    f2py_options = [],
    )

if __name__ == "__main__":
    setup(
        name = DISTNAME,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        url = URL,
        license = LICENSE,
        install_requires = REQUIREMENTS,
        classifiers = CLASSIFIERS,
        version = VERSION,
        packages = find_packages(),
        include_package_data = True,
        ext_modules = [ ext1, ext2, ext3 ],
    )
