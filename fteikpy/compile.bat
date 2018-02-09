@echo OFF

:: PARAMETERS
set F2PY=f2py
set FC=gfortran
set FLAGS="-O3 -ffast-math -funroll-loops -fno-protect-parens -fopenmp"
set F90DIR=f90/

:: COMMANDS
call %F2PY% -c -m _fteik2d --fcompiler=%FC% --f90flags=%FLAGS% -lgomp %F90DIR%fteik2d.f90
call %F2PY% -c -m _fteik3d --fcompiler=%FC% --f90flags=%FLAGS% -lgomp %F90DIR%fteik3d.f90
call %F2PY% -c -m _lay2vel --fcompiler=%FC% --f90flags=%FLAGS% %F90DIR%lay2vel.f90
call %F2PY% -c -m _bspline --fcompiler=%FC% --f90flags=%FLAGS% -lgomp %F90DIR%bspline.f90
