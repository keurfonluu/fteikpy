!=======================================================================
! Created by
!     Keurfon Luu <keurfon.luu@mines-paristech.fr>
!     MINES ParisTech - Centre de Géosciences
!     PSL - Research University
!=======================================================================

module fteik3d

  use omp_lib

  implicit none

  integer(kind = 4), parameter :: scheme = 1
  real(kind = 8), parameter :: zerr = 1.d-4
  real(kind = 8), parameter :: Big = 99999.d0

contains

  subroutine solver3d(slow, tt, nz, nx, ny, zsrc, xsrc, ysrc, dz, dx, dy, n_sweep, ttgrad)
    real(kind = 8), dimension(nz,nx,ny), intent(in) :: slow
    real(kind = 8), dimension(nz,nx,ny), intent(out) :: tt
    real(kind = 8), dimension(3,nz,nx,ny), intent(out), optional :: ttgrad
    integer(kind = 4), intent(in) :: nz, nx, ny, n_sweep
    real(kind = 8), intent(in) :: zsrc, xsrc, ysrc, dz, dx, dy

    integer(kind = 4) :: i, j, k, kk, iflag
    integer(kind = 4) :: zsi, xsi, ysi
    integer(kind = 4) :: sgntz, sgntx, sgnty, sgnvz, sgnvx, sgnvy

    real(kind = 8) :: ttz(nz,nx,ny), ttx(nz,nx,ny), tty(nz,nx,ny)
    real(kind = 8) :: zsa, xsa, ysa
    real(kind = 8) :: tzc, txc, tyc
    real(kind = 8) :: vzero

    ! Check inputs
    if ( nz .lt. 3 .or. nx .lt. 3 .or. ny .lt. 3 ) stop "Error: grid size nz, nx, ny too small"
    if ( dz .le. 0.d0 .or. dx .le. 0.d0 .or. dy .le. 0.d0 ) stop "Error: grid spacing dz, dx, dy too small"
    if ( n_sweep .lt. 0 ) stop "Error: wrong sweep number"
    if ( minval(slow) .le. 0.d0 .or. maxval(slow) .ge. 1.d0 ) stop "Error: slownesses are strange"
    if ( zsrc .lt. 0.d0 .or. zsrc .gt. dfloat(nz-1) * dz &
         .or. xsrc .lt. 0.d0 .or. xsrc .gt. dfloat(nx-1) * dx &
         .or. ysrc .lt. 0.d0 .or. ysrc .gt. dfloat(ny-1) * dy ) &
      stop "Error: source out of bounds"

    ! Convert src to grid position and try and take into account machine precision
    zsa = zsrc / dz + 1.d0
    xsa = xsrc / dx + 1.d0
    ysa = ysrc / dy + 1.d0

    ! Try to handle edges simply for source due to precision
    if ( zsa .ge. dfloat(nz) ) zsa = zsa - zerr
    if ( xsa .ge. dfloat(nx) ) xsa = xsa - zerr
    if ( ysa .ge. dfloat(ny) ) ysa = ysa - zerr

    ! Grid points to initialize source
    zsi = int(zsa)
    xsi = int(xsa)
    ysi = int(ysa)
    vzero = slow(zsi,xsi,ysi)

    ! Allocate work array for traveltimes
    tt = Big
    if ( present(ttgrad) ) then
      ttz = 0.d0
      ttx = 0.d0
      tty = 0.d0
    end if

    ! Initialize points around source
    tt(zsi,xsi,ysi) = t_ana(zsi, xsi, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt(zsi+1,xsi,ysi) = t_ana(zsi+1, xsi, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt(zsi,xsi+1,ysi) = t_ana(zsi, xsi+1, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt(zsi,xsi,ysi+1) = t_ana(zsi, xsi, ysi+1, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt(zsi+1,xsi+1,ysi) = t_ana(zsi+1, xsi+1, ysi, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt(zsi+1,xsi,ysi+1) = t_ana(zsi+1, xsi, ysi+1, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt(zsi,xsi+1,ysi+1) = t_ana(zsi, xsi+1, ysi+1, dz, dx, dy, zsa, xsa, ysa, vzero)
    tt(zsi+1,xsi+1,ysi+1) = t_ana(zsi+1, xsi+1, ysi+1, dz, dx, dy, zsa, xsa, ysa, vzero)

    ! Full sweeps
    do kk = 1, n_sweep
      call sweep3d(scheme, slow, tt, ttz, ttx, tty, nz, nx, ny, dz, dx, dy, &
                   zsi, xsi, ysi, tzc, txc, tyc, zsa, xsa, ysa, vzero)
    end do

    if ( present(ttgrad) ) then
      ttgrad(1,:,:,:) = ttz
      ttgrad(2,:,:,:) = ttx
      ttgrad(3,:,:,:) = tty
    end if
    return
  contains

    ! Function to calculate analytical times in homogeneous model
    real(kind = 8) function t_ana(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero)
      integer(kind = 4), intent(in) :: i, j, k
      real(kind = 8), intent(in) :: dz, dx, dy, zsa, xsa, ysa, vzero

      t_ana = vzero * ( ( ( dfloat(i) - zsa ) * dz )**2.d0 &
                      + ( ( dfloat(j) - xsa ) * dx )**2.d0 &
                      + ( ( dfloat(k) - ysa ) * dy )**2.d0 )**0.5d0
      return
    end function t_ana

    ! Function to calculate analytical times in homogeneous model + derivatives of times
    real(kind = 8) function t_anad(tzc, txc, tyc, i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero)
      integer(kind = 4), intent(in) :: i, j, k
      real(kind = 8), intent(in) :: dz, dx, dy, zsa, xsa, ysa, vzero
      real(kind = 8) :: d0
      real(kind = 8), intent(out) :: tzc, txc, tyc

      d0 = ( ( dfloat(i) - zsa ) * dz )**2.d0 &
           + ( ( dfloat(j) - xsa ) * dx )**2.d0 &
           + ( ( dfloat(k) - ysa ) * dy )**2.d0
      t_anad = vzero * (d0**0.5d0)
      if ( d0 .gt. 0.d0 ) then
        tzc = ( d0**(-0.5d0) ) * ( dfloat(i) - zsa ) * dz * vzero
        txc = ( d0**(-0.5d0) ) * ( dfloat(j) - xsa ) * dx * vzero
        tyc = ( d0**(-0.5d0) ) * ( dfloat(k) - ysa ) * dy * vzero
      else
        tzc = 0.d0
        txc = 0.d0
        tyc = 0.d0
      end if
      return
    end function t_anad

    ! Function to perform sweep
    subroutine sweep3d(scheme, slow, tt, ttz, ttx, tty, nz, nx, ny, dz, dx, dy, &
                       zsi, xsi, ysi, tzc, txc, tyc, zsa, xsa, ysa, vzero)
      real(kind = 8), intent(in) :: slow(nz,nx,ny)
      real(kind = 8), intent(inout) :: tt(nz,nx,ny), ttz(nz,nx,ny), ttx(nz,nx,ny), tty(nz,nx,ny)
      integer(kind = 4), intent(in) :: scheme, nz, nx, ny, zsi, xsi, ysi
      real(kind = 8), intent(in) :: dz, dx, dy, zsa, xsa, ysa, vzero
      real(kind = 8), intent(inout) :: tzc, txc, tyc
      integer(kind = 4) :: i, j, k, sgntz, sgntx, sgnty, sgnvz, sgnvx, sgnvy
      integer(kind = 4) :: i1, j1, k1, imin
      real(kind = 8) :: dzi, dxi, dyi, dz2i, dx2i, dy2i, dz2dx2, dz2dy2, dx2dy2, dsum
      real(kind = 8) :: vref, time_sol(8)
      real(kind = 8) :: tv, te, tn, ten, tnv, tev, tnve
      real(kind = 8) :: t1d, t2d, t3d, t1, t2, t3, ta, tb, tc
      real(kind = 8) :: t1d1, t1d2, t1d3, t2d1, t2d2, t2d3
      real(kind = 8) :: apoly, bpoly, cpoly, dpoly

      ! Precalculate constants
      dzi = 1.d0 / dz
      dxi = 1.d0 / dx
      dyi = 1.d0 / dy
      dz2i = 1.d0 / (dz*dz)
      dx2i = 1.d0 / (dx*dx)
      dy2i = 1.d0 / (dy*dy)
      dz2dx2 = dz2i * dx2i
      dz2dy2 = dz2i * dy2i
      dx2dy2 = dx2i * dy2i
      dsum = dz2i + dx2i + dy2i

      select case(scheme)
      ! Standard sweeping scheme
      case(1)
        ! First sweeping: Top->Bottom ; West->East ; South->North
        sgntz = 1 ; sgntx = 1 ; sgnty = 1
        sgnvz = 1 ; sgnvx = 1 ; sgnvy = 1
        do k = 2, ny
          do j = 2, nx
            do i = 2, nz
              include "Include_FTeik3d.f"
            end do
          end do
        end do

        ! Second sweeping: Top->Bottom ; East->West ; South->North
        sgntz = 1 ; sgntx = -1 ; sgnty = 1
        sgnvz = 1 ; sgnvx = 0 ; sgnvy = 1
        do k = 2, ny
          do j = nx-1, 1, -1
            do i = 2, nz
              include "Include_FTeik3d.f"
            end do
          end do
        end do

        ! Third sweeping: Top->Bottom ; West->East ; North->South
        sgntz = 1 ; sgntx = 1 ; sgnty = -1
        sgnvz = 1 ; sgnvx = 1 ; sgnvy = 0
        do k = ny-1, 1, -1
          do j = 2, nx
            do i = 2, nz
              include "Include_FTeik3d.f"
            end do
          end do
        end do

        ! Fouth sweeping: Top->Bottom ; East->West ; North->South
        sgntz = 1 ; sgntx = -1 ; sgnty = -1
        sgnvz = 1 ; sgnvx = 0 ; sgnvy = 0
        do k = ny-1, 1, -1
          do j = nx-1, 1, -1
            do i = 2, nz
              include "Include_FTeik3d.f"
            end do
          end do
        end do

        ! Fifth sweeping: Bottom->Top ; West->East ; South->North
        sgntz = -1 ; sgntx = 1 ; sgnty = 1
        sgnvz = 0 ; sgnvx = 1 ; sgnvy = 1
        do k = 2, ny
          do j = 2, nx
            do i = nz-1, 1, -1
              include "Include_FTeik3d.f"
            end do
          end do
        end do

        ! Sixth sweeping: Bottom->Top ; East->West ; South->North
        sgntz = -1 ; sgntx = -1 ; sgnty = 1
        sgnvz = 0 ; sgnvx = 0 ; sgnvy = 1
        do k = 2, ny
          do j = nx-1, 1, -1
            do i = nz-1, 1, -1
              include "Include_FTeik3d.f"
            end do
          end do
        end do

        ! Seventh sweeping: Bottom->Top ; West->East ; North->South
        sgntz = -1 ; sgntx = 1 ; sgnty = -1
        sgnvz = 0 ; sgnvx = 1 ; sgnvy = 0
        do k = ny-1, 1, -1
          do j = 2, nx
            do i = nz-1, 1, -1
              include "Include_FTeik3d.f"
            end do
          end do
        end do

        ! Eighth sweeping: Bottom->Top ; East->West ; North->South
        sgntz = -1 ; sgntx = -1 ; sgnty = -1
        sgnvz = 0 ; sgnvx = 0 ; sgnvy = 0
        do k = ny-1, 1, -1
          do j = nx-1, 1, -1
            do i = nz-1, 1, -1
              include "Include_FTeik3d.f"
            end do
          end do
        end do
      end select
      return
    end subroutine sweep3d

  end subroutine solver3d

  subroutine solve(slow, tt, nz, nx, ny, zs, xs, ys, dz, dx, dy, n_sweep, nsrc, n_threads)
    integer(kind = 4), intent(in) :: nz, nx, ny, nsrc, n_sweep
    integer(kind = 4), intent(in), optional :: n_threads
    real(kind = 8), intent(in) :: slow(nz,nx,ny), zs(nsrc), xs(nsrc), ys(nsrc), dz, dx, dy
    real(kind = 8), intent(out) :: tt(nsrc,nz,nx,ny)
    integer(kind = 4) :: k

    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    !$omp parallel default(shared)
    !$omp do schedule(runtime)
    do k = 1, nsrc
      call solver3d(slow, tt(k,:,:,:), nz, nx, ny, zs(k), xs(k), ys(k), dz, dx, dy, n_sweep)
    end do
    !$omp end parallel
    return
  end subroutine solve

  function interp3(x, y, z, v, xq, yq, zq) result(vq)
    real(kind = 8) :: vq
    real(kind = 8), intent(in) :: xq, yq, zq
    real(kind = 8), dimension(:), intent(in) :: x, y, z
    real(kind = 8), dimension(:,:,:), intent(in) :: v
    integer(kind = 4) :: nx, ny, nz, i1, i2, j1, j2, k1, k2
    real(kind = 8) :: x1, x2, y1, y2, z1, z2, v111, v211, v121, v221, &
      v112, v212, v122, v222
    real(kind = 8) :: N(8), ax(8), ay(8), az(8), av(8)

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
      x1 = x(i1)
      x2 = 2.d0 * x1 - x(nx-1)
      y1 = y(j1)
      y2 = y(j2)
      z1 = z(k1)
      z2 = z(k2)
      v111 = v(i1,j1,k1)
      v211 = 0.d0
      v121 = v(i1,j2,k1)
      v221 = 0.d0
      v112 = v(i1,j1,k2)
      v212 = 0.d0
      v122 = v(i1,j2,k2)
      v222 = 0.d0
    else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
      x1 = x(i1)
      x2 = x(i2)
      y1 = y(j1)
      y2 = 2.d0 * y1 - y(ny-1)
      z1 = z(k1)
      z2 = z(k2)
      v111 = v(i1,j1,k1)
      v211 = v(i2,j1,k1)
      v121 = 0.d0
      v221 = 0.d0
      v112 = v(i1,j1,k2)
      v212 = v(i2,j1,k2)
      v122 = 0.d0
      v222 = 0.d0
    else if ( i1 .ne. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
      x1 = x(i1)
      x2 = x(i2)
      y1 = y(j1)
      y2 = y(j2)
      z1 = z(k1)
      z2 = 2.d0 * z1 - z(nz-1)
      v111 = v(i1,j1,k1)
      v211 = v(i2,j1,k1)
      v121 = v(i1,j2,k1)
      v221 = v(i2,j2,k1)
      v112 = 0.d0
      v212 = 0.d0
      v122 = 0.d0
      v222 = 0.d0
    else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
      x1 = x(i1)
      x2 = 2.d0 * x1 - x(nx-1)
      y1 = y(j1)
      y2 = 2.d0 * y1 - y(ny-1)
      z1 = z(k1)
      z2 = z(k2)
      v111 = v(i1,j1,k1)
      v211 = 0.d0
      v121 = 0.d0
      v221 = 0.d0
      v112 = v(i1,j1,k2)
      v212 = 0.d0
      v122 = 0.d0
      v222 = 0.d0
    else if ( i1 .eq. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
      x1 = x(i1)
      x2 = 2.d0 * x1 - x(nx-1)
      y1 = y(j1)
      y2 = y(j2)
      z1 = z(k1)
      z2 = 2.d0 * z1 - z(nz-1)
      v111 = v(i1,j1,k1)
      v211 = 0.d0
      v121 = v(i1,j2,k1)
      v221 = 0.d0
      v112 = 0.d0
      v212 = 0.d0
      v122 = 0.d0
      v222 = 0.d0
    else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
      x1 = x(i1)
      x2 = x(i2)
      y1 = y(j1)
      y2 = 2.d0 * y1 - y(ny-1)
      z1 = z(k1)
      z2 = 2.d0 * z1 - z(nz-1)
      v111 = v(i1,j1,k1)
      v211 = v(i2,j1,k1)
      v121 = 0.d0
      v221 = 0.d0
      v112 = 0.d0
      v212 = 0.d0
      v122 = 0.d0
      v222 = 0.d0
    else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
      x1 = x(i1)
      x2 = 2.d0 * x1 - x(nx-1)
      y1 = y(j1)
      y2 = 2.d0 * y1 - y(ny-1)
      z1 = z(k1)
      z2 = 2.d0 * z1 - z(nz-1)
      v111 = v(i1,j1,k1)
      v211 = 0.d0
      v121 = 0.d0
      v221 = 0.d0
      v112 = 0.d0
      v212 = 0.d0
      v122 = 0.d0
      v222 = 0.d0
    else
      x1 = x(i1)
      x2 = x(i2)
      y1 = y(j1)
      y2 = y(j2)
      z1 = z(k1)
      z2 = z(k2)
      v111 = v(i1,j1,k1)
      v211 = v(i2,j1,k1)
      v121 = v(i1,j2,k1)
      v221 = v(i2,j2,k1)
      v112 = v(i1,j1,k2)
      v212 = v(i2,j1,k2)
      v122 = v(i1,j2,k2)
      v222 = v(i2,j2,k2)
    end if

    ax = [ x2, x1, x2, x1, x2, x1, x2, x1 ]
    ay = [ y2, y2, y1, y1, y2, y2, y1, y1 ]
    az = [ z2, z2, z2, z2, z1, z1, z1, z1 ]
    av = [ v111, v211, v121, v221, v112, v212, v212, v222 ]
    N = dabs( (ax - xq) * (ay - yq) * (az - zq) ) / dabs( (x2 - x1) * (y2 - y1) * (z2 -z1) )
    vq = dot_product(av, N)
    return
  end function interp3

  ! Apparent velocity interpolation: returns NaN when query point is
  ! near source (within [zsrc-dz, zsrc], [xsrc-dx, xsrc], [ysrc-dy, ysrc])
  ! as traveltime at source is zero. Commented until fixed.
  ! function interp3(source, x, y, z, v, xq, yq, zq) result(vq)
  !   real :: vq
  !   real, intent(in) :: xq, yq, zq
  !   real, dimension(:), intent(in) :: source, x, y, z
  !   real, dimension(:,:,:), intent(in) :: v
  !   integer :: nx, ny, nz, i1, i2, j1, j2, k1, k2
  !   real :: x1, x2, y1, y2, z1, z2, v111, v211, v121, v221, &
  !     v112, v212, v122, v222, d111, d211, d121, d221, d112, d212, d122, d222
  !   real :: N(8), ax(8), ay(8), az(8), av(8), ad(8)
  !
  !   if ( all(source .eq. [ xq, yq, zq ]) ) then
  !     vq = 0.
  !   else
  !     nx = size(v, 1)
  !     ny = size(v, 2)
  !     nz = size(v, 3)
  !     i1 = minloc(xq - x, dim = 1, mask = xq .ge. x)
  !     j1 = minloc(yq - y, dim = 1, mask = yq .ge. y)
  !     k1 = minloc(zq - z, dim = 1, mask = zq .ge. z)
  !     i2 = i1 + 1
  !     j2 = j1 + 1
  !     k2 = k1 + 1
  !
  !     if ( i1 .eq. nx .and. j1 .ne. ny .and. k1 .ne. nz ) then
  !       x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = z(k2)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
  !       d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = 0.
  !       d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = 0.
  !       d122 = sqrt(sum((source-[x1,y2,z2])**2)); d222 = 0.
  !       v111 = v(i1,j1,k1); v211 = 1.; v121 = v(i1,j2,k1); v221 = 1.
  !       v112 = v(i1,j1,k2); v212 = 1.; v122 = v(i1,j2,k2); v222 = 1.
  !     else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
  !       x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = z(k2)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
  !       d121 = 0.; d221 = 0.
  !       d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = sqrt(sum((source-[x2,y1,z2])**2))
  !       d122 = 0.; d222 = 0.
  !       v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = 1.; v221 = 1.
  !       v112 = v(i1,j1,k2); v212 = v(i2,j1,k2); v122 = 1.; v222 = 1.
  !     else if ( i1 .ne. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
  !       x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
  !       d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = sqrt(sum((source-[x2,y2,z1])**2))
  !       d112 = 0.; d212 = 0.
  !       d122 = 0.; d222 = 0.
  !       v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = v(i1,j2,k1); v221 = v(i2,j2,k1)
  !       v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
  !     else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
  !       x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = z(k2)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
  !       d121 = 0.; d221 = 0.
  !       d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = 0.
  !       d122 = 0.; d222 = 0.
  !       v111 = v(i1,j1,k1); v211 = 1.; v121 = 1.; v221 = 1.
  !       v112 = v(i1,j1,k2); v212 = 1.; v122 = 1.; v222 = 1.
  !     else if ( i1 .eq. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
  !       x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
  !       d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = 0.
  !       d112 = 0.; d212 = 0.
  !       d122 = 0.; d222 = 0.
  !       v111 = v(i1,j1,k1); v211 = 1.; v121 = v(i1,j2,k1); v221 = 1.
  !       v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
  !     else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
  !       x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
  !       d121 = 0.; d221 = 0.
  !       d112 = 0.; d212 = 0.
  !       d122 = 0.; d222 = 0.
  !       v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = 1.; v221 = 1.
  !       v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
  !     else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
  !       x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = 0.
  !       d121 = 0.; d221 = 0.
  !       d112 = 0.; d212 = 0.
  !       d122 = 0.; d222 = 0.
  !       v111 = v(i1,j1,k1); v211 = 1.; v121 = 1.; v221 = 1.
  !       v112 = 1.; v212 = 1.; v122 = 1.; v222 = 1.
  !     else
  !       x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = z(k2)
  !       d111 = sqrt(sum((source-[x1,y1,z1])**2)); d211 = sqrt(sum((source-[x2,y1,z1])**2))
  !       d121 = sqrt(sum((source-[x1,y2,z1])**2)); d221 = sqrt(sum((source-[x2,y2,z1])**2))
  !       d112 = sqrt(sum((source-[x1,y1,z2])**2)); d212 = sqrt(sum((source-[x2,y1,z2])**2))
  !       d122 = sqrt(sum((source-[x1,y2,z2])**2)); d222 = sqrt(sum((source-[x2,y2,z2])**2))
  !       v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = v(i1,j2,k1); v221 = v(i2,j2,k1)
  !       v112 = v(i1,j1,k2); v212 = v(i2,j1,k2); v122 = v(i1,j2,k2); v222 = v(i2,j2,k2)
  !     end if
  !
  !     ax = [ x2, x1, x2, x1, x2, x1, x2, x1 ]
  !     ay = [ y2, y2, y1, y1, y2, y2, y1, y1 ]
  !     az = [ z2, z2, z2, z2, z1, z1, z1, z1 ]
  !     av = [ v111, v211, v121, v221, v112, v212, v212, v222 ]
  !     ad = [ d111, d211, d121, d221, d112, d212, d212, d222 ]
  !     N = abs( (ax - xq) * (ay - yq) * (az - zq) ) / abs( (x2 - x1) * (y2 - y1) * (z2 -z1) )
  !     vq = sqrt(sum((source-[xq,yq,zq])**2)) / dot_product(ad / av, N)
  !   end if
  !   return
  ! end function interp3

  function slow2tt3(slow, nz, nx, ny, dz, dx, dy, zs, xs, ys, zr, xr, yr, &
                    n_sweep, nsrc, nrcv, n_threads) result(tt)
    real(kind = 8) :: tt(nrcv,nsrc)
    integer(kind = 4), intent(in) :: nz, nx, ny, n_sweep, nsrc, nrcv
    integer(kind = 4), intent(in), optional :: n_threads
    real(kind = 8), intent(in) :: slow(nz,nx,ny), dz, dx, dy, zs(nsrc), &
      xs(nsrc), ys(nsrc), zr(nrcv), xr(nrcv), yr(nrcv)
    integer(kind = 4) :: i, j, k, n1, n2
    real(kind = 8), dimension(:), allocatable :: ax, az, ay
    real(kind = 8), dimension(:,:), allocatable :: tcalc, rcv, src
    real(kind = 8), dimension(:,:,:), allocatable :: tmp

    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    ! Switch sources and stations to minimize calls to eikonals
    n1 = min(nrcv, nsrc)
    n2 = max(nrcv, nsrc)
    if ( n1 .eq. nsrc ) then
      rcv = reshape([ zr, xr, yr ], shape = [ n2, 3 ], order = [ 1, 2 ])
      src = reshape([ zs, xs, ys ], shape = [ n1, 3 ], order = [ 1, 2 ])
    else
      rcv = reshape([ zs, xs, ys ], shape = [ n2, 3 ], order = [ 1, 2 ])
      src = reshape([ zr, xr, yr ], shape = [ n1, 3 ], order = [ 1, 2 ])
    end if

    ! Initialize tcalc
    allocate(tcalc(n2,n1))

    ! Compute traveltimes using an eikonal solver
    az = dz * [ ( dfloat(k-1), k = 1, nz ) ]
    ax = dx * [ ( dfloat(k-1), k = 1, nx ) ]
    ay = dy * [ ( dfloat(k-1), k = 1, ny ) ]
    !$omp parallel default(shared) private(tmp, j)
    !$omp do schedule(runtime)
    do i = 1, n1
      allocate(tmp(nz,nx,ny))
      call solver3d(slow, tmp, nz, nx, ny, src(i,1), src(i,2), src(i,3), dz, dx, dy, n_sweep)
      do j = 1, n2
        tcalc(j,i) = interp3(az, ax, ay, tmp, rcv(j,1), rcv(j,2), rcv(j,3))
      end do
      deallocate(tmp)
    end do
    !$omp end parallel

    ! Transpose to reshape to [ nrcv, nsrc ]
    if ( n1 .eq. nrcv ) then
      tt = transpose(tcalc)
    else
      tt = tcalc
    end if
    return
  end function slow2tt3

end module fteik3d
