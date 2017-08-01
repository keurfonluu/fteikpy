#!/bin/bash

# PARAMETERS
F2PY="f2py"
FC="gfortran"
FFLAGS="-O3 -ffast-math -march=native -funroll-loops -fno-protect-parens -flto -fopenmp"
F90DIR="f90/"

# COMMANDS
$F2PY -c -m _fteik2d --fcompiler=$FC --f90flags="$FFLAGS" -lgomp "$F90DIR"FTeik2d.f90
$F2PY -c -m _fteik3d --fcompiler=$FC --f90flags="$FFLAGS" -lgomp "$F90DIR"FTeik3d.f90
$F2PY -c -m _interpolate --fcompiler=$FC --f90flags="$FFLAGS" "$F90DIR"interpolate.f90
