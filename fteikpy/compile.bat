@echo OFF

:: PARAMETERS
set F2PY=f2py
set FC=gfortran
set FFLAGS="-O3 -ffast-math -march=native -funroll-loops -fno-protect-parens -flto -fopenmp"
set F90DIR=f90/

:: COMMANDS
call %F2PY% -c -m _fteik2d --fcompiler=%FC% --f90flags=%FFLAGS% -lgomp %F90DIR%FTeik2d.f90
call %F2PY% -c -m _fteik3d --fcompiler=%FC% --f90flags=%FFLAGS% -lgomp %F90DIR%FTeik3d.f90
call %F2PY% -c -m _lay2vel --fcompiler=%FC% --f90flags=%FFLAGS% %F90DIR%lay2vel.f90
