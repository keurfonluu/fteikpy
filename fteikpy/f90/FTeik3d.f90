module eikonal3d

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
  !   compute traveltimes in a 3D heterogenious velocity model by solving
  !   by finite difference approximation the Eikonal equation. This package
  !   is written in fortran 90 and is composed of 4 elements, the functions
  !   "t_ana", "t_anad", the subroutine "FTeik3d_2" and an include file
  !   "include_FTeik3d_2".
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
  !     call FTeik3d_2(vel,tt,nz,nx,ny,zsin,xsin,ysin,dzin,dxin,dyin,nsweep,epsin)
  ! 
  !     WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
  ! 
  !     WARNING        :Time field array and velocity field array do not have
  !                     the same dimension. Velocities are defined at center of
  !                     cell, whereas times are computed on the corners.
  ! 
  !     integer - nz,nx,ny  :Dimensions of the time field array tt
  !                           tt(nz,nx,ny)
  !                           No dimension may be lower than 3.
  ! 
  !     integer - dzin,dxin,dyin :Mesh spacing along the 3 axis
  ! 
  !     real - tt                :Travel time field array: tt(nz,nx,ny)
  ! 
  !     real - vel               :Velocity field array: vel(nz-1,nx-1,ny-1)
  ! 
  !     real - zsin,xsin,ysin :Point source coordinates referred expressed in meters
  !                         Licit ranges: [0.0,(nz-1.)*dz][0.0,(nx-1.)*dx] [0.,(ny-1.)*dy]
  ! 
  !     integer - nsweep         :Number of sweeps over model. 1 is in general enough
  ! 
  !     real - epsin :  radius in number of grid points arround source where then
  !                     spherical approximation will be used
  !
  !________________________________________________________________________
  !
  subroutine FTeik3d(vel,tt,nz,nx,ny,zsin,xsin,ysin,dzin,dxin,dyin,nsweep,epsin)

     implicit none
  !
  ! Parameter for double precision
  !_______________________________
     integer,parameter   :: PRES=kind(1.d0)
  !
  ! Size of traveltime map
  !_______________________
     integer,intent(in)   ::   nz,nx,ny
  !
  ! Number of sweeps to do
  !_______________________
     integer,intent(in)   ::   nsweep
     real,intent(in)   ::   dzin,dxin,dyin
     real,intent(in)   ::   zsin,xsin,ysin
     real,intent(in)   ::   epsin

     real,intent(out),dimension(nz,nx,ny)   ::   tt
     real,intent(in),dimension(nz-1,nx-1,ny-1)   ::   vel

     integer   ::   i,j,k,kk,i1,j1,k1

     real(kind=PRES), parameter   ::   Big = 99999.d0

     real(kind=PRES)   ::   dz,dx,dy
     real(kind=PRES)   ::   zsrc,xsrc,ysrc,zsa,xsa,ysa
     real(kind=PRES)   ::   eps,tmin
     integer   ::   zsi,xsi,ysi
     integer   ::   sgntz,sgntx,sgnty,sgnvz,sgnvx,sgnvy

     real(kind=PRES)   :: vzero,vref
     real(kind=PRES)   :: t1d,t2d,t3d,t1,t2,t3,ta,tb,tc
     real(kind=PRES)   :: tv,te,tn,ten,tnv,tev,tnve
     real(kind=PRES)   :: tauv,taue,taun,tauen,taunv,tauev,taunve

     real(kind=PRES)   ::   dzi,dxi,dyi,dz2i,dx2i,dy2i,dsum,dz2dx2,dz2dy2,dx2dy2

     real(kind=PRES)   ::   sgnrz,sgnrx,sgnry
     real(kind=PRES)   ::   t0c,tzc,txc,tyc
     real(kind=PRES)   ::   apoly,bpoly,cpoly,dpoly
  !
  !  Check grid size
  !_________________
     if (nz < 3 .or. nx < 3 .or. ny < 3 ) goto 993
  !   
  !  Check grid spacing
  !____________________
     dz=dble(dzin) ; dx=dble(dxin) ; dy=dble(dyin)
     if (dz <= 0. .or. dx <= 0. .or. dy <= 0.) goto 994
  !
  ! Check sweep
  !____________
     if (nsweep < 0) goto 995
  !   
  ! Check velocity field
  !_____________________
     if (minval(vel) <= 0.) goto 992
  !   
  ! Check source position
  !______________________
     zsrc=dble(zsin) ; xsrc=dble(xsin) ; ysrc=dble(ysin)
     if ( zsrc < 0.d0 .or. zsrc > (dfloat(nz-1)*dz) ) goto 990
     if ( xsrc < 0.d0 .or. xsrc > (dfloat(nx-1)*dx) ) goto 990
     if ( ysrc < 0.d0 .or. ysrc > (dfloat(ny-1)*dy) ) goto 990
  !
  ! Convert src pos to grid position
  !_________________________________
     zsa = (zsrc/dz)+1.d0 ; xsa=(xsrc / dx)+1.d0 ; ysa=(ysrc/dy)+1.d0
  !
  ! Trick to handle edges simply for source
  !________________________________________
     if ( zsa == 1.d0 ) zsa=zsa + 0.0001d0
     if ( zsa >= nz ) zsa=zsa - 0.0001d0
     if ( xsa == 1.d0 ) xsa=xsa + 0.0001d0
     if ( xsa >= nx ) xsa=xsa - 0.0001d0
     if ( ysa == 1.d0 ) ysa=ysa + 0.0001d0
     if ( ysa >= ny ) ysa=ysa - 0.0001d0
  !
  !  Grid points to initialise source
  !__________________________________
     zsi = int(zsa)
     xsi = int(xsa)
     ysi = int(ysa)
     vzero=1.d0 / dble(vel(zsi,xsi,ysi))
  !
  ! Set spherical approximation radius, convert grid number to time
  !________________________________________________________________
     eps=dble(epsin)
     if (int(eps) > min(nz,nx,ny)) goto 996
     eps = eps * vzero * dble(min(dz,dx,dy))
  !
  ! Set traveltime map to BIG value
  !________________________________
     tt=Big
  !
  ! Initialise points around source
  !________________________________

     tt(zsi,xsi,ysi) = sngl( t_ana(zsi,xsi,ysi,dz,dx,dy,zsa,xsa,ysa,vzero) )
     tt(zsi+1,xsi,ysi) = sngl( t_ana(zsi+1,xsi,ysi,dz,dx,dy,zsa,xsa,ysa,vzero) )
     tt(zsi,xsi+1,ysi) = sngl( t_ana(zsi,xsi+1,ysi,dz,dx,dy,zsa,xsa,ysa,vzero) )
     tt(zsi,xsi,ysi+1) = sngl( t_ana(zsi,xsi,ysi+1,dz,dx,dy,zsa,xsa,ysa,vzero) )
     tt(zsi+1,xsi+1,ysi) = sngl( t_ana(zsi+1,xsi+1,ysi,dz,dx,dy,zsa,xsa,ysa,vzero) )
     tt(zsi+1,xsi,ysi+1) = sngl( t_ana(zsi+1,xsi,ysi+1,dz,dx,dy,zsa,xsa,ysa,vzero) )
     tt(zsi,xsi+1,ysi+1) = sngl( t_ana(zsi,xsi+1,ysi+1,dz,dx,dy,zsa,xsa,ysa,vzero) )
     tt(zsi+1,xsi+1,ysi+1) = sngl( t_ana(zsi+1,xsi+1,ysi+1,dz,dx,dy,zsa,xsa,ysa,vzero) )
  ! 
  ! Pre-calculate a few constants concerning mesh spacing
  !______________________________________________________
      dzi = 1.d0 /dz
      dxi = 1.d0 /dx
      dyi = 1.d0 /dy
      dz2i = 1.d0 / (dz**2)
      dx2i = 1.d0 / (dx**2)
      dy2i = 1.d0 / (dy**2)
      dsum = dz2i + dx2i + dy2i
      dz2dx2 = 1.d0/(dz**2 * dx**2)
      dz2dy2 = 1.d0/(dz**2 * dy**2)
      dx2dy2 = 1.d0/(dx**2 * dy**2)
  !
  ! Set sweep variable to 0 for first run kk=0
    kk=0
  !
  ! First sweeping: Top->Bottom ; West->East ; South->North
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=1 ; sgnty=1
      sgnvz=1 ; sgnvx=1 ; sgnvy=1
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = max(2,ysi),ny
      do j = max(2,xsi),nx
      do i = max(2,zsi),nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Second sweeping: Top->Bottom; East->West; South->North
  ! Set direction variables
  !________________________________________________________

      sgntz=1 ; sgntx=-1 ; sgnty=1
      sgnvz=1 ; sgnvx=0 ; sgnvy=1
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = max(2,ysi),ny
      do j = xsi+1,1,-1
      do i = max(2,zsi),nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Third sweeping: Top->Bottom; West->East; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=1 ; sgnty=-1
      sgnvz=1 ; sgnvx=1 ; sgnvy=0
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = ysi+1,1,-1
      do j = max(2,xsi),nx
      do i = max(2,zsi),nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Fouth sweeping: Top->Bottom; East->West ; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=-1 ; sgnty=-1
      sgnvz=1 ; sgnvx=0 ; sgnvy=0
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = ysi+1,1,-1
      do j = xsi+1,1,-1
      do i = max(2,zsi),nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Fifth sweep: Bottom->Top ; West->East ; South->North
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=1 ; sgnty=1
      sgnvz=0 ; sgnvx=1 ; sgnvy=1
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = max(2,ysi),ny
      do j = max(2,xsi),nx
      do i = zsi+1,1,-1
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Six sweeping: Bottom->Top; East->West; South->North
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=-1 ; sgnty=1
      sgnvz=0 ; sgnvx=0 ; sgnvy=1
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = max(2,ysi),ny
      do j = xsi+1,1,-1
      do i = zsi+1,1,-1
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Seven sweeping: Bottom->Top; West->East; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=1 ; sgnty=-1
      sgnvz=0 ; sgnvx=1 ; sgnvy=0
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = ysi+1,1,-1
      do j = max(2,xsi),nx
      do i = zsi+1,1,-1
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Eighth sweeping: Bottom->Top; East->West ; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=-1 ; sgnty=-1
      sgnvz=0 ; sgnvx=0 ; sgnvy=0
      sgnrz=dfloat(sgntz) ; sgnrx=dfloat(sgntx) ; sgnry=dfloat(sgnty)

      do k = ysi+1,1,-1
      do j = xsi+1,1,-1
      do i = zsi+1,1,-1
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  ! End of first run
  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Ready to do at least one global sweep
     do kk = 1,nsweep
  !
  ! First sweeping: Top->Bottom ; West->East ; South->North
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=1 ; sgnty=1
      sgnvz=1 ; sgnvx=1 ; sgnvy=1

      do k = 2,ny
      do j = 2,nx
      do i = 2,nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Second sweeping: Top->Bottom; East->West; South->North
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=-1 ; sgnty=1
      sgnvz=1 ; sgnvx=0 ; sgnvy=1

      do k = 2,ny
      do j = nx-1,1,-1
      do i = 2,nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Third sweeping: Top->Bottom; West->East; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=1 ; sgnty=-1
      sgnvz=1 ; sgnvx=1 ; sgnvy=0

      do k = ny-1,1,-1
      do j = 2,nx
      do i = 2,nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Fouth sweeping: Top->Bottom; East->West ; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=1 ; sgntx=-1 ; sgnty=-1
      sgnvz=1 ; sgnvx=0 ; sgnvy=0

      do k = ny-1,1,-1
      do j = nx-1,1,-1
      do i = 2,nz
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Fifth sweep: Bottom->Top ; West->East ; South->North
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=1 ; sgnty=1
      sgnvz=0 ; sgnvx=1 ; sgnvy=1

      do k=2,ny
      do j = 2,nx
      do i = nz-1,1,-1
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Six sweeping: Bottom->Top; East->West; South->North
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=-1 ; sgnty=1
      sgnvz=0 ; sgnvx=0 ; sgnvy=1

      do k = 2,ny
      do j = nx-1,1,-1
      do i = nz-1,1,-1
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Seven sweeping: Bottom->Top; West->East; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=1 ; sgnty=-1
      sgnvz=0 ; sgnvx=1 ; sgnvy=0

      do k = ny-1,1,-1
      do j = 2,nx
      do i = nz-1,1,-1
           include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! Eighth sweeping: Bottom->Top; East->West ; North->South
  ! Set direction variables
  !________________________________________________________
      sgntz=-1 ; sgntx=-1 ; sgnty=-1
      sgnvz=0 ; sgnvx=0 ; sgnvy=0

      do k = ny-1,1,-1
      do j = nx-1,1,-1
      do i = nz-1,1,-1
            include 'Include_FTeik3d.f'
      enddo
      enddo
      enddo
  !
  ! End loop for global sweeps
  enddo
  !
  ! That's all
  ! It's easy
     return
  !
  990   continue
        write(*,'(/)')
        write(*,'(''=================================================='')')
        write(*,'(5x,''ERROR FTAeik3d, source out of bounds '')')
        write(*,'(''=================================================='')')
        stop
  !
  992   continue
        write(*,'(/)')
        write(*,'(''=================================================='')')
        write(*,'(5x,''ERROR FTAeik3d, Velocities are strange '')')
        write(*,'(''=================================================='')')
        stop
        
  993   continue
        write(*,'(/)')
        write(*,'(''=================================================='')')
        write(*,'(5x,''ERROR FTAeik3d, Grid size nz,nx,ny too small '')')
        write(*,'(''=================================================='')')
        stop
              
  994   continue
        write(*,'(/)')
        write(*,'(''================================================='')')
        write(*,'(5x,''ERROR FTAeik3d, Grid spacing dz,dx,dy too small '')')
        write(*,'(''================================================='')')
        stop
                    
  995   continue
        write(*,'(/)')
        write(*,'(''================================================='')')
        write(*,'(5x,''ERROR FTAeik3d, Sweep number wrong '')')
        write(*,'(''================================================='')')
        stop

  996   continue
        write(*,'(/)')
        write(*,'(''================================================='')')
        write(*,'(5x,''ERROR FTAeik3d, epsin bigger than model '')')
        write(*,'(''================================================='')')
        stop

  contains
  
    ! Functions to calculate analytical times in homgeneous model
    real(kind=kind(1.d0)) function t_ana(i,j,k,dz,dx,dy,zsa,xsa,ysa,vzero)

      implicit none

      integer,parameter  :: PRES=kind(1.d0)
      integer, intent(in)  :: i,j,k

      real(kind=PRES),intent(in)  :: dz,dx,dy,zsa,xsa,ysa,vzero

      t_ana =vzero*(((dfloat(i)-zsa)*dz)**2.+((dfloat(j)-xsa)*dx)**2.+((dfloat(k)-ysa)*dy)**2.)**0.5

    end function t_ana
   
    ! Functions to calculate analytical times in homgeneous model, + derivative of times
    real(kind=kind(1.d0)) function t_anad(tzc,txc,tyc,i,j,k,dz,dx,dy,zsa,xsa,ysa,vzero)

      implicit none

      integer,parameter  :: PRES=kind(1.d0)
      integer, intent(in)  :: i,j,k

      real(kind=PRES),intent(in)  :: dz,dx,dy,zsa,xsa,ysa,vzero
      real(kind=PRES)  :: d0
      real(kind=PRES),intent(out)   :: tzc,txc,tyc

      d0=((dfloat(i)-zsa)*dz)**2.+((dfloat(j)-xsa)*dx)**2.+((dfloat(k)-ysa)*dy)**2.

      t_anad = vzero * (d0**0.5)

      if ( d0 > 0.d0) then
        tzc = (d0**(-0.5)) *(dfloat(i)-zsa)*dz * vzero
        txc = (d0**(-0.5)) *(dfloat(j)-xsa)*dx * vzero
        tyc = (d0**(-0.5)) *(dfloat(k)-ysa)*dy * vzero
      else
        tzc = 0.d0
        txc = 0.d0
        tyc = 0.d0
      endif

    end function t_anad
    
  end subroutine FTeik3d
  
  subroutine solve(vel, ttout, nz, nx, ny, zsin, xsin, ysin, nsrc, dzin, dxin, dyin, nsweep, n_threads)
    integer, intent(in) :: nz, nx, ny, nsrc, nsweep, n_threads
    real, intent(in) :: vel(nz-1,nx-1,ny-1), zsin(nsrc), xsin(nsrc), ysin(nsrc), dzin, dxin, dyin
    real, intent(out) :: ttout(nz, nx, ny, nsrc)
    integer :: k
    
    call omp_set_num_threads(n_threads)
    
    !$omp parallel default(shared)
    !$omp do schedule(runtime)
    do k = 1, nsrc
      call FTeik3d(vel, ttout(:,:,:,k), nz, nx, ny, zsin(k), xsin(k), ysin(k), &
                   dzin, dxin, dyin, nsweep, 5.)
    end do
    !$omp end parallel
    return
  end subroutine solve
  
  function interp3(source, x, y, z, v, xq, yq, zq) result(vq)
      real :: vq
      real, intent(in) :: xq, yq, zq
      real, dimension(:), intent(in) :: source, x, y, z
      real, dimension(:,:,:), intent(in) :: v
      integer :: nx, ny, nz, i1, i2, j1, j2, k1, k2
      real :: x1, x2, y1, y2, z1, z2, v111, v211, v121, v221, &
        v112, v212, v122, v222, d111, d211, d121, d221, d112, d212, d122, d222
      real :: N(8), ax(8), ay(8), az(8), av(8), ad(8)

      nx = size(v, 1)
      ny = size(v, 2)
      nz = size(v, 3)
      i1 = minloc(xq - x, dim = 1, mask = xq .ge. x)
      j1 = minloc(yq - y, dim = 1, mask = yq .ge. y)
      k1 = minloc(zq - z, dim = 1, mask = zq .ge. z)
      i2 = i1 + 1
      j2 = j1 + 1
      k2 = k1 + 1

      if ( i1 .eq. nx .and. j1 .ne. ny .and. k1 .ne. nz ) then
        x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = z(k2)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
        d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = 0.
        d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = 0.
        d122 = sqrt(sum((source-[x1,y2,z2])**2)); d222 = 0.
        v111 = v(i1,j1,k1); v211 = 1.; v121 = v(i1,j2,k1); v221 = 1.
        v112 = v(i1,j1,k2); v212 = 1.; v122 = v(i1,j2,k2); v222 = 1.
      else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
        x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = z(k2)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
        d121 = 0.; d221 = 0.
        d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = sqrt(sum((source-[x2,y1,z2])**2))
        d122 = 0.; d222 = 0.
        v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = 1.; v221 = 1.
        v112 = v(i1,j1,k2); v212 = v(i2,j1,k2); v122 = 1.; v222 = 1.
      else if ( i1 .ne. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
        x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
        d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = sqrt(sum((source-[x2,y2,z1])**2))
        d112 = 0.; d212 = 0.
        d122 = 0.; d222 = 0.
        v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = v(i1,j2,k1); v221 = v(i2,j2,k1)
        v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
      else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
        x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = z(k2)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
        d121 = 0.; d221 = 0.
        d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = 0.
        d122 = 0.; d222 = 0.
        v111 = v(i1,j1,k1); v211 = 1.; v121 = 1.; v221 = 1.
        v112 = v(i1,j1,k2); v212 = 1.; v122 = 1.; v222 = 1.
      else if ( i1 .eq. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
        x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
        d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = 0.
        d112 = 0.; d212 = 0.
        d122 = 0.; d222 = 0.
        v111 = v(i1,j1,k1); v211 = 1.; v121 = v(i1,j2,k1); v221 = 1.
        v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
      else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
        x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
        d121 = 0.; d221 = 0.
        d112 = 0.; d212 = 0.
        d122 = 0.; d222 = 0.
        v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = 1.; v221 = 1.
        v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
      else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
        x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
        d121 = 0.; d221 = 0.
        d112 = 0.; d212 = 0.
        d122 = 0.; d222 = 0.
        v111 = v(i1,j1,k1); v211 = 1.; v121 = 1.; v221 = 1.
        v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
      else
        x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = z(k2)
        d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
        d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = sqrt(sum((source-[x2,y2,z1])**2))
        d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = sqrt(sum((source-[x2,y1,z2])**2))
        d122 = sqrt(sum((source-[x1,y2,z2])**2)); d222 = sqrt(sum((source-[x2,y2,z2])**2))
        v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = v(i1,j2,k1); v221 = v(i2,j2,k1)
        v112 = v(i1,j1,k2); v212 = v(i2,j1,k2); v122 = v(i1,j2,k2); v222 = v(i2,j2,k2)
      end if

      ax = [ x2, x1, x2, x1, x2, x1, x2, x1 ]
      ay = [ y2, y2, y1, y1, y2, y2, y1, y1 ]
      az = [ z2, z2, z2, z2, z1, z1, z1, z1 ]
      av = [ v111, v211, v121, v221, v112, v212, v212, v222 ]
      ad = [ d111, d211, d121, d221, d112, d212, d212, d222 ]
      N = abs( (ax - xq) * (ay - yq) * (az - zq) ) / abs( (x2 - x1) * (y2 - y1) * (z2 -z1) )
      vq = sqrt(sum((source-[xq,yq,zq])**2)) / dot_product(ad / av, N)
      return
    end function interp3
  
end module eikonal3d
