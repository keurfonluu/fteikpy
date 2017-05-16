!
!   Copyright or Copr. Mines Paristech, France - Mark NOBLE, Alexandrine GESRET
!   FTeik3d_2.0.f90 - First release Aug 2011
!
!   FTeik3d has been written by:
!     - Mark Noble <mark.noble@mines-paristech.fr>
!     - Alexandrine Gesret <alexandrine.gesret@mines-paristech.fr>
!
!   This software is a computer program (subroutine) whose purpose is to
!   compute traveltimes in a 3D heterogenious velocity model by solving
!   by finite difference approximation the Eikonal equation. This package
!   is written in fortran 90 and is composed of 4 elements, the functions
!   "t_ana", "t_anad", the subroutine "FTeik3d_2" and an include file
!   "include_FT3d_2".
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
!Local Operators 1D, 2D and 3D
!
! define index of velocity nodes to use, don't touch!
!____________________________________________________
    i1=i-sgnvz
    j1=j-sgnvx
    k1=k-sgnvy
!
! Get local times of surrounding points
!______________________________________
    tv = dble( tt(i-sgntz,j,k) )
    te = dble( tt(i,j-sgntx,k) )
    tn = dble( tt(i,j,k-sgnty) )
    tev = dble( tt(i-sgntz,j-sgntx,k) )
    ten = dble( tt(i,j-sgntx,k-sgnty) )
    tnv = dble( tt(i-sgntz,j,k-sgnty) )
    tnve = dble( tt(i-sgntz,j-sgntx,k-sgnty) )
!
! check time to see when to switch to plane approximation
!________________________________________________________
    tmin=min(tv,te,tn)
!
! get analytical solution, if using pertubation
!______________________________________________
    if (tmin <= eps .or. kk < 1) then
      t0c = t_anad(tzc,txc,tyc,i,j,k,dz,dx,dy,zsa,xsa,ysa,vzero)
!
! Convert times into pertubations
!________________________________
      tauv = tv - t_ana(i-sgntz,j,k,dz,dx,dy,zsa,xsa,ysa,vzero)
      taue = te - t_ana(i,j-sgntx,k,dz,dx,dy,zsa,xsa,ysa,vzero)
      taun = tn - t_ana(i,j,k-sgnty,dz,dx,dy,zsa,xsa,ysa,vzero)
      tauev = tev - t_ana(i-sgntz,j-sgntx,k,dz,dx,dy,zsa,xsa,ysa,vzero)
      tauen = ten - t_ana(i,j-sgntx,k-sgnty,dz,dx,dy,zsa,xsa,ysa,vzero)
      taunv = tnv - t_ana(i-sgntz,j,k-sgnty,dz,dx,dy,zsa,xsa,ysa,vzero)
      taunve = tnve - t_ana(i-sgntz,j-sgntx,k-sgnty,dz,dx,dy,zsa,xsa,ysa,vzero)
    endif
!
! 1D operators, (refracted times),set times to BIG
!_________________________________________________
    t1d=Big
    t1=Big ; t2=Big ; t3=Big
!
!V plane
!_______
    vref = 1.d0 / dble( max( vel(i1,max(j-1,1),max(k-1,1)),vel(i1,max(j-1,1),min(k,ny-1)), &
                      vel(i1,min(j,nx-1),max(k-1,1)),vel(i1,min(j,nx-1),min(k,ny-1))))
    t1= tv + dz * vref
!
!WE plane
!________
    vref = 1.d0 / dble( max( vel(max(i-1,1),j1,max(k-1,1)),vel(min(i,nz-1),j1,max(k-1,1)), &
                      vel(max(i-1,1),j1,min(k,ny-1)),vel(min(i,nz-1),j1,min(k,ny-1))))
    t2= te + dx * vref
!
!NS plane
!________
    vref = 1.d0 / dble( max( vel(max(i-1,1),max(j-1,1),k1),vel(max(i-1,1),min(j,nx-1),k1), &
                      vel(min(i,nz-1),max(j-1,1),k1),vel(min(i,nz-1),min(j,nx-1),k1)))
    t3= tn + dy * vref
!
! End of 1D operators (just take smallest time)
!______________________________________________
    t1d=min(t1,t2,t3)
!
!2D operators
!____________
    t2d=Big; t1=Big ; t2=Big ; t3=Big
!
!ZX (VE) plane
!_____________
    vref=1.d0 / dble( max( vel(i1,j1,max(k-1,1)),vel(i1,j1,min(k,ny-1)) ))
!
! Check condition and the choose between plane or spherical approximation
!________________________________________________________________________
    if ( (tv < te+dx*vref) .and. (te < tv+dz*vref) ) then
      if (k /= ysi .or. tmin > eps .or. kk > 0) then
        ta=tev+te-tv
        tb=tev-te+tv
        t1=((tb*dz2i+ta*dx2i)+sqrt(4.d0*(vref**2.d0)*(dz2i+dx2i) &
            - dz2i*dx2i*(ta-tb)**2.d0))/(dz2i+dx2i)
      else
        ta = tauev+taue-tauv   ! X
        tb = tauev-taue+tauv   ! Z
        apoly=dz2i+dx2i
        bpoly=4.d0 *(sgnrx*txc*dxi+sgnrz*tzc*dzi)-2.d0*(ta*dx2i + tb*dz2i)
        cpoly=((ta**2)*dx2i)+((tb**2)*dz2i) &
              -4.d0*(sgnrx*txc*dxi*ta+sgnrz*tzc*dzi*tb)+4.d0*(vzero**2-vref**2+tyc**2)
        dpoly=(bpoly**2)-4.d0*apoly*cpoly
        if (dpoly >= 0.d0) then
          t1=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
        endif
        if ((t1-tv <0.d0 .or. t1-te <0.d0))t1=Big
      endif
    endif
!
!ZY (VN) plane
!_____________
    vref=1.d0 / dble( max( vel(i1,max(j-1,1),k1),vel(i1,min(j,nx-1),k1) ))
!
! Check condition and the choose between plane or spherical approximation
!________________________________________________________________________
    if ( (tv < tn+dy*vref) .and. (tn < tv+dz*vref) ) then
      if (j /= xsi .or. tmin > eps .or. kk > 0)then
        ta=tv-tn+tnv
        tb=tn-tv+tnv
        t2=((ta*dz2i+tb*dy2i)+sqrt(4.d0*(vref**2.d0)*(dz2i+dy2i) &
            -dz2i*dy2i*(ta-tb)**2.d0))/(dz2i+dy2i)
      else
        ta=tauv-taun+taunv   !Z
        tb=taun-tauv+taunv   !Y
        apoly=dz2i+dy2i
        bpoly=4.d0*(sgnry*tyc*dyi+sgnrz*tzc*dzi)-2.d0 *(ta*dz2i + tb*dy2i)
        cpoly=((ta**2)*dz2i)+((tb**2)*dy2i) &
              -4.d0*(sgnrz*tzc*dzi*ta+sgnry*tyc*dyi*tb)+4.d0*(vzero**2-vref**2+txc**2)
        dpoly=(bpoly**2)-4.d0*apoly*cpoly
        if (dpoly >= 0.d0) then
          t2=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
        endif
        if ((t2 -tv <0.d0 .or. t2-tn <0.d0))t2=Big
      endif
    endif
!
!XY (EN) plane
!_____________
    vref=1.d0/dble(max(vel(max(i-1,1),j1,k1),vel(min(i,nz-1),j1,k1)))
!
! Check condition and the choose between plane or spherical approximation
!________________________________________________________________________
    if ( (te < tn+dy*vref) .and. (tn < te+dx*vref) ) then
      if (i /= zsi .or. tmin > eps .or. kk > 0) then
        ta=te-tn+ten
        tb=tn-te+ten
        t3 =((ta*dx2i+tb*dy2i)+sqrt(4.d0*(vref**2.d0)*(dx2i+dy2i) &
            -dx2i*dy2i*(ta-tb)**2.d0))/(dx2i+dy2i)
      else
        ta=taue-taun+tauen  !X
        tb=taun-taue+tauen  !Y
        apoly=dx2i+dy2i
        bpoly=4.d0*(sgnry*tyc*dyi+sgnrx*txc*dxi)-2.d0*(ta*dx2i + tb*dy2i)
        cpoly=((ta**2)*dx2i)+((tb**2)*dy2i) &
              -4.d0*(sgnrx*txc*dxi*ta+sgnry*tyc*dyi*tb)+4.d0*(vzero**2-vref**2+tzc**2)
        dpoly=(bpoly**2)-4.d0*apoly*cpoly
        if (dpoly >= 0.d0) then
          t3=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
        endif
      if ((t3-te <0.d0 .or. t3-tn <0.d0))t3=Big
      endif
    endif
!
! End of 2D operators
!____________________
    t2d=min(t1,t2,t3)
!
! 3D operator
!____________
    t3d=Big

    if ( min(t1d,t2d) > max(tv,te,tn) ) then
      vref= 1.d0 / dble(vel(i1,j1,k1))
      if (tmin > eps .or. kk > 0) then
        ta=te-0.5d0*tn+0.5d0*ten-0.5d0*tv+0.5d0*tev-tnv+tnve !X
        tb=tv-0.5d0*tn+0.5d0*tnv-0.5d0*te+0.5d0*tev-ten+tnve !Z
        tc=tn-0.5d0*te+0.5d0*ten-0.5d0*tv+0.5d0*tnv-tev+tnve !Y
        t2=vref**2.d0*dsum*9.d0
        t3=dz2dx2*(ta-tb)**2.d0+dz2dy2*(tb-tc)**2.d0+dx2dy2*(ta-tc)**2.d0
        if ( t2 >= t3 )then
          t1=tb*dz2i+ta*dx2i+tc*dy2i
          t3d=(t1+sqrt(t2-t3))/dsum
        endif
      else
        ta=taue-0.5d0*taun+0.5d0*tauen-0.5d0*tauv+0.5d0*tauev-taunv+taunve !X
        tb=tauv-0.5d0*taun+0.5d0*taunv-0.5d0*taue+0.5d0*tauev-tauen+taunve !Z
        tc=taun-0.5d0*taue+0.5d0*tauen-0.5d0*tauv+0.5d0*taunv-tauev+taunve !Y
        apoly=dx2i+dz2i+dy2i
        bpoly=6.d0 *(sgnrz*tzc*dzi+sgnry*tyc*dyi+sgnrx*txc*dxi) &
              -2.d0*(ta*dx2i+tb*dz2i+tc*dy2i)
        cpoly=((ta**2)*dx2i)+((tb**2)*dz2i)+((tc**2)*dy2i) &
              -6.d0*(sgnrx*txc*dxi*ta+sgnrz*tzc*dzi*tb+sgnry*tyc*dyi*tc) &
              +9.d0*(vzero**2-vref**2)
        dpoly=(bpoly**2)-4.d0*apoly*cpoly
        if (dpoly >= 0.d0) then
          t3d =(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
        endif
        if (t3d-te < 0.d0 .or. t3d-tn < 0.d0 .or. t3d-tv < 0.d0)t3d=Big
      endif
    endif
!
! End of 3D,Choose shortest path of 1,2 and 3d times
!___________________________________________________
    t1 = min(dble(tt(i,j,k)),t1d,t2d,t3d)
    tt(i,j,k)=sngl(t1)
!
! End of Calculation for this sweep
!__________________________________
