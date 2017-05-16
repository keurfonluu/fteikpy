!
!   Copyright or Copr. Mines Paristech, France - Mark NOBLE, Alexandrine GESRET
!   FTeik2d_3.0.f90 - First release Aug 2011
!
!   FTeik2d_3 has been written by:
!     - Mark Noble <mark.noble@mines-paristech.fr>
!     - Alexandrine Gesret <alexandrine.gesret@mines-paristech.fr>
!
!   This software is a computer program (subroutine) whose purpose is to
!   compute traveltimes in a 2D heterogeneous velocity model by solving
!   by finite difference approximation the Eikonal equation. This package
!   is written in fortran 90 and is composed of 4 elements, the functions
!   "t_ana", "t_anad", the subroutine "FTeik2d_3" and an include file
!   "include_FT2d_3".
!
!   This software is governed by the CeCILL-C license under French law and
!   abiding by the rules of distribution of free software.  You can  use,
!   modify and/ or redistribute the software under the terms of the CeCILL-C
!   license as circulated by CEA, CNRS and INRIA at the following URL
!   "http://www.cecill.info".
!
!   As a counterpart to the access to the source code and  rights to copy,
!   modify and redistribute granted by the license, users are provided only
!   with a limited warranty  and the software's author,  the holder of the
!   economic rights,  and the successive licensors  have only  limited
!   liability.
!
!   In this respect, the user's attention is drawn to the risks associated
!   with loading,  using,  modifying and/or developing or reproducing the
!   software by the user in light of its specific status of free software,
!   that may mean  that it is complicated to manipulate,  and  that  also
!   therefore means  that it is reserved for developers  and  experienced
!   professionals having in-depth computer knowledge. Users are therefore
!   encouraged to load and test the software's suitability as regards their
!   requirements in conditions enabling the security of their systems and/or
!   data to be ensured and,  more generally, to use and operate it in the
!   same conditions as regards security.
!
!   The fact that you are presently reading this means that you have had
!   knowledge of the CeCILL-C license and that you accept its terms.
!   For more information, see the file COPYRIGHT-GB or COPYRIGHT-FR.
!
!________________________________________________________________________
!
!Local Operators 1D, 2D
!
! define index of velocity nodes to use, don't touch!
!____________________________________________________
    i1=i-sgnvz
    j1=j-sgnvx
!
! Get local times of surrounding points
!______________________________________
    tv = tt(i-sgntz,j)
    te = tt(i,j-sgntx)
    tev = tt(i-sgntz,j-sgntx)
!
! get analytical solution, if using pertubation
!______________________________________________
 !   if (tmin <= eps) then
      t0c = t_anad(tzc,txc,i,j,dz,dx,zsa,xsa,vzero)
!
! Convert times into pertubations
!________________________________
      tauv = tv - t_ana(i-sgntz,j,dz,dx,zsa,xsa,vzero)
      taue = te - t_ana(i,j-sgntx,dz,dx,zsa,xsa,vzero)
      tauev = tev - t_ana(i-sgntz,j-sgntx,dz,dx,zsa,xsa,vzero)
!    endif
!
! 1D operators, (refracted times),set times to BIG
!_________________________________________________
    t1d=Big
    t1=Big ; t2=Big
!
!V plane
!_______
    vref = 1.d0 / dble( max( vel(i1,max(j-1,1)),vel(i1,min(j,nx-1))))
    t1= tv + dz * vref
!
!WE plane
!________
    vref = 1.d0 / dble( max( vel(max(i-1,1),j1),vel(min(i,nz-1),j1) ) )
    t2= te + dx * vref
!
! End of 1D operators (just take smallest time)
!______________________________________________
    t1d=min(t1,t2)
!
!2D operators, and diagonal operators
!____________
    t2d=Big; t1=Big ; t2=Big ; t3=Big ; tdiag=Big
    vref=1.d0 / dble( vel(i1,j1) )
!
! Diagonal operator
!____________________
 tdiag= tev + vref*(dx**2.d0 + dz**2.d0)**0.5d0
!
! choose spherical or plane wave
! First test for Plane wave
!_______________________________

 if ( ( abs(i-zsi) .gt. epsin .or. abs(j-xsi) .gt. epsin ) ) then
!
! 4 Point operator, if possible otherwise do three points
!________________________________________________________________________
    if ( (tv .le. te+dx*vref) .and. (te .le. tv+dz*vref) &
        .and. (te-tev .ge. 0.d0) .and. (tv-tev .ge. 0.d0) ) then
           ta=tev+te-tv
           tb=tev-te+tv
           t1=((tb*dz2i+ta*dx2i)+sqrt(4.d0*(vref**2.d0)*(dz2i+dx2i) &
             - dz2i*dx2i*(ta-tb)**2.d0))/(dz2i+dx2i)
!
! Two 3 point operators
!______________________
    elseif ( ((te-tev).le.dz**2.d0*vref/sqrt(dx**2.d0+dz**2.d0)).and. ((te-tev).gt.0.d0) ) then
            t2=te+dx*sqrt(vref**2.d0-((te-tev)/dz)**2.d0)

    elseif ( ((tv-tev).le.dx**2.d0*vref/sqrt(dx**2.d0+dz**2.d0)).and. ((tv-tev).gt.0.d0)) then
            t3=tv+dz*sqrt(vref**2.d0-((tv-tev)/dx)**2.d0)
    endif
!
! Test for spherical
!___________________  
 else
!
! do spherical operator if conditions ok
!_______________________________________
   if ( (tv < te+dx*vref) .and. (te < tv+dz*vref) &
       .and. (te-tev .ge. 0.d0) .and. (tv-tev .ge. 0.d0) ) then
       ta = tauev+taue-tauv   ! X
       tb = tauev-taue+tauv   ! Z
       apoly=dz2i+dx2i
       bpoly=4.d0 *(sgnrx*txc*dxi+sgnrz*tzc*dzi)-2.d0*(ta*dx2i + tb*dz2i)
       cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
             -4.d0*(sgnrx*txc*dxi*ta+sgnrz*tzc*dzi*tb)+4.d0*(vzero**2.d0-vref**2.d0)
       dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
       if (dpoly >= 0.d0) t1=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
       if ((t1-tv <0.d0 .or. t1-te <0.d0))t1=Big
    endif
!
 endif
!
! End of 2D operators
!____________________
    t2d=min(t1,t2,t3)
!
! shortest path of 1D and 2D and diagonal
!________________________________________
!    tt(i,j) = min(tt(i,j),t1d,t2d,tdiag)
    tt(i,j) = min(tt(i,j),t1d,t2d)
!
! End of Calculation for this sweep
!__________________________________
!

