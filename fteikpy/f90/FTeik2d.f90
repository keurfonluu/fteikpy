module eikonal2d

  use omp_lib

  implicit none
  
contains
!
!   Copyright or Copr. Mines Paristech, France - Mark NOBLE, Alexandrine GESRET
!   FTeik3d_2.0.f90 - First release Aug 2011
! 
!   FTeik3d has been written by:
!     - Mark Noble <mark.noble@mines-paristech.fr>
!     - Alexandrine Gesret <alexandrine.gesret@mines-paristech.fr>
! 
!   This software is a computer program (subroutine) whose purpose is to
!   compute traveltimes in a 2D heterogeneous velocity model by solving
!   by finite difference approximation the Eikonal equation. This package
!   is written in fortran 90 and is composed of 4 elements, the functions
!   "t_ana", "t_anad", the subroutine "FTeik2d_3" and an include file
!   "include_FTeik2d_3".
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
!            ARGUMENTS REQUIRED TO CALL THE SUBROUTINE FTeik3d_2
!     call FTeik2d_3(vel,ttout,nz,nx,zsin,xsin,dzin,dxin,epsin,nsweep)
! 
!     WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
! 
!     WARNING        :Time field array and velocity field array have
!                     the same dimension. Velocities are defined at center of
!                     cell, whereas times are computed on the corners. The last row
!                     and last column of velocity field are not used in the computation
! 
!     integer - nz,nx  :Dimensions of the time field array tt
!                           tt(nz,nx)
!                           No dimension may be lower than 3.
! 
!     real - dzin,dxin :Mesh spacing along the 3 axis
! 
!     real - ttout                :Travel time field array: ttout(nz,nx)
! 
!     real - vel               :Velocity field array: vel(nz,nx)
! 
!     real - zsin,xsin :Point source coordinates referred expressed in meters
!                         Licit ranges: [0.0,(nz-1.)*dzin][0.0,(nx-1.)*dxin]
! 
!     integer - nsweep         :Number of sweeps over model. 1 is in general enough
! 
!     integer - epsin :  radius in number of grid points arround source where then
!                     spherical approximation will be used
!
!________________________________________________________________________
!
  subroutine FTeik2d(vel,ttout,nz,nx,zsin,xsin,dzin,dxin,nsweep,epsin)

     implicit none
  !
  ! Parameter for double precision
  !_______________________________
     integer,parameter   :: PRES=kind(1.d0)
  !
  ! Size of traveltime map
  !_______________________
     integer,intent(in)   ::   nz,nx
  !
  ! Number of sweeps to do
  !_______________________
     integer,intent(in)   ::   nsweep
     real,intent(in)   ::   dzin,dxin
     real,intent(in)   ::   zsin,xsin
     integer,intent(in)   ::   epsin

     real,intent(out),dimension(nz,nx)   ::   ttout
     real,intent(in),dimension(nz,nx)    ::   vel
  !
  ! Work array to do all calculations in double
  !___________________________________________
     real(kind=PRES),allocatable,dimension(:,:)  :: tt

     integer   ::   i,j,kk,i1,j1
     integer   :: ierr,iflag

     real(kind=PRES), parameter   ::   Big = 99999.d0
     real(kind=PRES), parameter   ::   zerr = 1.d-04
     integer,parameter ::   nmax=15000
     real(kind=PRES)   ::   td(nmax)

     real(kind=PRES)   ::   dz,dx,dzu,dzd,dxw,dxe
     real(kind=PRES)   ::   zsrc,xsrc,zsa,xsa
     integer   ::   zsi,xsi
     integer   ::   sgntz,sgntx,sgnvz,sgnvx

     real(kind=PRES)   :: vzero,vref
     real(kind=PRES)   :: t1d,t2d,t1,t2,t3,tdiag,ta,tb
     real(kind=PRES)   :: tv,te,tev
     real(kind=PRES)   :: tauv,taue,tauev

     real(kind=PRES)   ::   dzi,dxi,dz2i,dx2i,dsum,dz2dx2

     real(kind=PRES)   ::   sgnrz,sgnrx
     real(kind=PRES)   ::   t0c,tzc,txc
     real(kind=PRES)   ::   apoly,bpoly,cpoly,dpoly
  !
  !  Check grid size
  !_________________
     if (nz < 3 .or. nx < 3 ) goto 993
     if (max(nz,nx) > nmax) goto 996
  !   
  !  Check grid spacing
  !____________________
     dz=dble(dzin) ; dx=dble(dxin)
     if (dz <= 0. .or. dx <= 0. ) goto 994
  !
  ! Check sweep
  !____________
     if (nsweep < 1) goto 995
  !   
  ! Check velocity field
  !_____________________
     if (minval(vel) <= 0.) goto 992
  !   
  ! Check source position
  !______________________
     zsrc=dble(zsin) ; xsrc=dble(xsin)
     if ( zsrc < 0.d0 .or. zsrc > (dfloat(nz-1)*dz) ) goto 990
     if ( xsrc < 0.d0 .or. xsrc > (dfloat(nx-1)*dx) ) goto 990
  !
  ! Convert src pos to grid position and try and take into account machine precision
  !_________________________________
     zsa = (zsrc/dz)+1.d0 ; xsa=(xsrc/dx)+1.d0
  !
  ! Try to handle edges simply for source due to precision
  !________________________________________
     if ( zsa >= dfloat(nz) ) zsa=zsa - zerr
     if ( xsa >= dfloat(nx) ) xsa=xsa - zerr

  !
  !  Grid points to initialise source
  !__________________________________
     zsi = int(zsa)
     xsi = int(xsa)

     vzero=1.d0 / dble(vel(zsi,xsi))
  !
  ! Allocate work array for traveltimes
  !____________________________________
     allocate(tt(nz,nx),stat=ierr)
     if (ierr .ne. 0) stop "Error FTeik2d.f90 in allocating tt array"
  !
  ! Set traveltime map to BIG value
  !________________________________
     tt=Big
  !
  ! do our best to initialize source
  !________________________________
      dzu=abs( zsa-dfloat(zsi) )  ; dzd=abs( dfloat(zsi+1)-zsa )
      dxw=abs( xsa-dfloat(xsi) )  ; dxe=abs( dfloat(xsi+1)-xsa )
      iflag=0
  ! source seems close enough to a grid point in X and Y direction
      if (min(dzu,dzd) < zerr .and. min(dxw,dxe) < zerr) then
          zsa=dnint(zsa)  ;  xsa=dnint(xsa)    
          iflag=1
      endif
  !
  ! At least one of coordinates not close to any grid point in X and Y direction
      if (min(dzu,dzd) > zerr .or. min(dxw,dxe) > zerr) then
         if (min(dzu,dzd) < zerr) zsa=dnint(zsa)
         if (min(dxw,dxe) < zerr) xsa=dnint(xsa)
         iflag=2
      endif
  !
  ! Oups we are lost, not sure this happens - fixe Src to nearest grid point
      if (iflag /= 1 .and. iflag /= 2) then
         zsa=dnint(zsa)  ;  xsa=dnint(xsa)    
         iflag=3
      endif

  ! We know where src is - start first propagation
      select case(iflag)
        case(1)
          tt(nint(zsa),nint(xsa))=0.d0
        case(3)
          tt(nint(zsa),nint(xsa))=0.d0
        case(2)
            dzu=abs( zsa-dfloat(zsi) )  ; dzd=abs( dfloat(zsi+1)-zsa )
            dxw=abs( xsa-dfloat(xsi) )  ; dxe=abs( dfloat(xsi+1)-xsa )
            ! first initialize 4 points around source
            tt(zsi,xsi) =  t_ana(zsi,xsi,dz,dx,zsa,xsa,vzero)
            tt(zsi+1,xsi) = t_ana(zsi+1,xsi,dz,dx,zsa,xsa,vzero)
            tt(zsi,xsi+1) =  t_ana(zsi,xsi+1,dz,dx,zsa,xsa,vzero)
            tt(zsi+1,xsi+1) = t_ana(zsi+1,xsi+1,dz,dx,zsa,xsa,vzero)

                ! X
                td=Big
                td(xsi+1)= vzero * dxe * dx
                dx2i = 1.d0 / (dx**2.d0)

                do j = xsi+2,nx
                vref=1.d0 / dble( vel(zsi,j-1) )
                td(j)=td(j-1) + dx * vref
                tv = td(j) ; tev=td(j-1)
                tauv = tv - vzero * abs((dfloat(j)-xsa)) * dx
                tauev = tev - vzero * abs((dfloat(j-1)-xsa)) * dx
                !
                te=tt(zsi+1,j-1)
                taue = te - t_ana(zsi+1,j-1,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,zsi+1,j,dz,dx,zsa,xsa,vzero)
                sgntz=1 ; sgntx=1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dz2i = 1.d0 / (dzd**2.d0) * dz
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dx+sgnrz*tzc/dzd)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dx*ta+sgnrz*tzc/dzd*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(zsi+1,j)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
                !
                te=tt(zsi,j-1)
                taue = te - t_ana(zsi,j-1,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,zsi,j,dz,dx,zsa,xsa,vzero)
                sgntz=-1 ; sgntx=1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dz2i = 1.d0 / (dzu**2.d0) * dz
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dx+sgnrz*tzc/dzu)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dx*ta+sgnrz*tzc/dzu*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(zsi,j)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
                enddo


               ! X
                td(xsi)= vzero * dxw * dx
                do j=xsi-1,1,-1
                vref=1.d0 / dble( vel(zsi,j) )
                td(j)=td(j+1) + dx * vref
                tv = td(j) ; tev=td(j+1)
                tauv = tv - vzero * abs((dfloat(j)-xsa)) * dx
                tauev = tev - vzero * abs((dfloat(j+1)-xsa)) * dx
                !
                te=tt(zsi+1,j+1)
                taue = te - t_ana(zsi+1,j+1,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,zsi+1,j,dz,dx,zsa,xsa,vzero)
                sgntz=1 ; sgntx=-1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dz2i = 1.d0 / (dzd**2.d0) * dz
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dx+sgnrz*tzc/dzd)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dx*ta+sgnrz*tzc/dzd*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(zsi+1,j)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
                !
                te=tt(zsi,j+1)
                taue = te - t_ana(zsi,j+1,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,zsi,j,dz,dx,zsa,xsa,vzero)
                sgntz=-1 ; sgntx=-1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dz2i = 1.d0 / (dzu**2.d0) * dz
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dx+sgnrz*tzc/dzu)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dx*ta+sgnrz*tzc/dzu*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(zsi,j)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
                enddo

                ! Z
                td=Big
                td(zsi+1)= vzero * dzd * dz
                dz2i = 1.d0 / (dz**2.d0)
                do i = zsi+2,nz
                vref=1.d0 / dble( vel(i-1,xsi) )
                td(i)=td(i-1) + dz * vref
                te = td(i) ; tev=td(i-1)
                taue = te - vzero * abs((dfloat(i)-zsa)) * dz
                tauev = tev - vzero * abs((dfloat(i-1)-zsa)) * dz
                !
                tv=tt(i-1,xsi+1)
                tauv = tv - t_ana(i-1,xsi+1,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,i,xsi+1,dz,dx,zsa,xsa,vzero)
                sgntz=1 ; sgntx=1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dx2i = 1.d0 / (dxe**2.d0) * dx
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dxe+sgnrz*tzc/dz)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dxe*ta+sgnrz*tzc/dz*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(i,xsi+1)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
                !
                tv=tt(i-1,xsi)
                tauv = tv - t_ana(i-1,xsi,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,i,xsi,dz,dx,zsa,xsa,vzero)
                sgntz=1 ; sgntx=-1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dx2i = 1.d0 / (dxw**2.d0) * dx
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dxw+sgnrz*tzc/dz)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dxw*ta+sgnrz*tzc/dz*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(i,xsi)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
                enddo

            ! Z
                td(zsi)= vzero * dzu * dz
                do i = zsi-1,1,-1
                vref=1.d0 / dble( vel(i,xsi) )
                td(i)=td(i+1) + dz * vref
                te = td(i) ; tev=td(i+1)
                taue = te - vzero * abs((dfloat(i)-zsa)) * dz
                tauev = tev - vzero * abs((dfloat(i+1)-zsa)) * dz

                tv=tt(i+1,xsi+1)
                tauv = tv - t_ana(i+1,xsi+1,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,i,xsi+1,dz,dx,zsa,xsa,vzero)
                sgntz=-1 ; sgntx=1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dx2i = 1.d0 / (dxe**2.d0) * dx
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dxe+sgnrz*tzc/dz)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dxe*ta+sgnrz*tzc/dz*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(i,xsi+1)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c

                tv=tt(i+1,xsi)
                tauv = tv - t_ana(i+1,xsi,dz,dx,zsa,xsa,vzero)
                t0c = t_anad(tzc,txc,i,xsi,dz,dx,zsa,xsa,vzero)
                sgntz=-1 ; sgntx=-1 ; sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
                ta = tauev+taue-tauv; tb = tauev-taue+tauv ; dx2i = 1.d0 / (dxw**2.d0) * dx
                apoly=dz2i+dx2i
                bpoly=4.d0 *(sgnrx*txc/dxw+sgnrz*tzc/dz)-2.d0*(ta*dx2i + tb*dz2i)
                cpoly=((ta**2.d0)*dx2i)+((tb**2.d0)*dz2i) &
                      -4.d0*(sgnrx*txc/dxw*ta+sgnrz*tzc/dz*tb)+4.d0*(vzero**2.d0-vref**2.d0)
                dpoly=(bpoly**2.d0)-4.d0*apoly*cpoly
                if (dpoly >= 0.d0) tt(i,xsi)=(sqrt(dpoly)-bpoly)/2.d0/apoly+t0c
                enddo

        end select
  ! 
  ! Pre-calculate a few constants concerning mesh spacing
  !______________________________________________________
      dzi = 1.d0 /dz
      dxi = 1.d0 /dx
      dz2i = 1.d0 / (dz**2.d0)
      dx2i = 1.d0 / (dx**2.d0)
      dsum = dz2i + dx2i
      dz2dx2 = 1.d0/(dz**2.d0 * dx**2.d0)
  !
  ! Ready to do at least one global sweep
     do kk = 1,nsweep
  !
  ! First sweeping: Top->Bottom ; West->East
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=1 
      sgnvz=1 ; sgnvx=1
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)

      do j = 2,nx
      do i = 2,nz
           include 'Include_FTeik2d.f'
      enddo
      enddo
  !goto 911
  !
  ! Second sweeping: Top->Bottom; East->West
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=-1
      sgnvz=1 ; sgnvx=0
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
      do j = nx-1,1,-1
      do i = 2,nz
           include 'Include_FTeik2d.f'
      enddo
      enddo
  !
  ! Third sweep: Bottom->Top ; West->East
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=1
      sgnvz=0 ; sgnvx=1
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
      do j = 2,nx
      do i = nz-1,1,-1
           include 'Include_FTeik2d.f'
      enddo
      enddo
  !
  ! Fourth sweeping: Bottom->Top; East->West
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=-1
      sgnvz=0 ; sgnvx=0
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx)
      do j = nx-1,1,-1
      do i = nz-1,1,-1
           include 'Include_FTeik2d.f'
      enddo
      enddo
  !911 continue
  !
  ! End loop for global sweeps
  enddo
     ttout=sngl(tt)
     deallocate(tt)
  !
  ! That's all
  ! It's easy
     return
  !
  990   continue
        write(*,'(/)')
        write(*,'(''=================================================='')')
        write(*,'(5x,''ERROR FTeik2d, source out of bounds '')')
        write(*,'(''=================================================='')')
        stop
  !
  992   continue
        write(*,'(/)')
        write(*,'(''=================================================='')')
        write(*,'(5x,''ERROR FTeik2d, Velocities are strange '')')
        write(*,'(10x,''EQUAL or SMALLER to ZERO '')')
        write(*,'(''=================================================='')')
        stop
        
  993   continue
        write(*,'(/)')
        write(*,'(''=================================================='')')
        write(*,'(5x,''ERROR FTeik2d, Grid size nz,nx too small '')')
        write(*,'(''=================================================='')')
        stop
              
  994   continue
        write(*,'(/)')
        write(*,'(''================================================='')')
        write(*,'(5x,''ERROR FTeik2d, Grid spacing dz,dx too small '')')
        write(*,'(''================================================='')')
        stop
                    
  995   continue
        write(*,'(/)')
        write(*,'(''================================================='')')
        write(*,'(5x,''ERROR FTeik2d, Sweep number wrong '')')
        write(*,'(''================================================='')')
        stop
                    
  996   continue
        write(*,'(/)')
        write(*,'(''================================================='')')
        write(*,'(5x,''ERROR FTeik2d, Must increase size of NMAX '')')
        write(*,'(10x,'' Need to recompile routine'')')
        write(*,'(''================================================='')')
        stop
        
  contains
  
    ! Functions to calculate analytical times in homgeneous model
    real(kind=kind(1.d0)) function t_ana(i,j,dz,dx,zsa,xsa,vzero)

      implicit none

      integer,parameter  :: PRES=kind(1.d0)
      integer, intent(in)  :: i,j

      real(kind=PRES),intent(in)  :: dz,dx,zsa,xsa,vzero

      t_ana =vzero*(((dfloat(i)-zsa)*dz)**2.d0+((dfloat(j)-xsa)*dx)**2.d0)**0.5d0

    end function t_ana
   
    ! Functions to calculate analytical times in homogeneous model, + derivative of times
    real(kind=kind(1.d0)) function t_anad(tzc,txc,i,j,dz,dx,zsa,xsa,vzero)

      implicit none

      integer,parameter  :: PRES=kind(1.d0)
      integer, intent(in)  :: i,j

      real(kind=PRES),intent(in)  :: dz,dx,zsa,xsa,vzero
      real(kind=PRES)  :: d0
      real(kind=PRES),intent(out)   :: tzc,txc

      d0= ((dfloat(i)-zsa)*dz)**2.d0+((dfloat(j)-xsa)*dx)**2.d0

      t_anad = vzero * (d0**0.5d0)

      if ( d0 > 0.d0) then
        tzc = (d0**(-0.5d0)) *(dfloat(i)-zsa)*dz * vzero
        txc = (d0**(-0.5d0)) *(dfloat(j)-xsa)*dx * vzero
      else
        tzc = 0.d0
        txc = 0.d0
      endif

    end function t_anad
   
  end subroutine FTeik2d
  
  subroutine solve(vel, ttout, nz, nx, zsin, xsin, nsrc, dzin, dxin, nsweep, n_threads)
    integer, intent(in) :: nz, nx, nsrc, nsweep, n_threads
    real, intent(in) :: vel(nz,nx), zsin(nsrc), xsin(nsrc), dzin, dxin
    real, intent(out) :: ttout(nz, nx, nsrc)
    integer :: k
    
    call omp_set_num_threads(n_threads)
    
    !$omp parallel default(shared)
    !$omp do schedule(runtime)
    do k = 1, nsrc
      call FTeik2d(vel, ttout(:,:,k), nz, nx, zsin(k), xsin(k), dzin, dxin, nsweep, 5)
    end do
    !$omp end parallel
    return
  end subroutine solve

end module eikonal2d
