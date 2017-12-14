!=======================================================================
! Created by
!     Keurfon Luu <keurfon.luu@mines-paristech.fr>
!     MINES ParisTech - Centre de Géosciences
!     PSL - Research University
!=======================================================================

module fteik2d

  use omp_lib

  implicit none

  integer(kind = 4), parameter :: nmax = 15000
  real(kind = 8), parameter :: Big = 99999.d0, zerr = 1.d-4

contains

  subroutine solver2d(slow, tt, nz, nx, zsrc, xsrc, dz, dx, n_sweep, ttgrad)
    real(kind = 8), dimension(nz,nx), intent(in) :: slow
    real(kind = 8), dimension(nz,nx), intent(out) :: tt
    real(kind = 8), dimension(nz,nx,2), intent(out), optional :: ttgrad
    integer(kind = 4), intent(in) :: nz, nx, n_sweep
    real(kind = 8), intent(in) :: zsrc, xsrc, dz, dx

    integer(kind = 4) :: imin, i, j, kk, i1, j1, ierr, iflag
    integer(kind = 4) :: zsi, xsi
    integer(kind = 4) :: sgntz, sgntx, sgnvz, sgnvx

    real(kind = 8) :: ttz(nz,nx), ttx(nz,nx)
    real(kind = 8) :: t1d1, t1d2, time_sol(4), td(nmax)
    real(kind = 8) :: dzu, dzd, dxw, dxe
    real(kind = 8) :: zsa, xsa
    real(kind = 8) :: vzero, vref
    real(kind = 8) :: t1d, t2d, t1, t2, t3, tdiag, ta, tb
    real(kind = 8) :: tv, te, tev
    real(kind = 8) :: tauv, taue, tauev
    real(kind = 8) :: dzi, dxi, dz2i, dx2i, dsum, dz2dx2
    real(kind = 8) :: sgnrz, sgnrx
    real(kind = 8) :: t0c, tzc, txc
    real(kind = 8) :: apoly, bpoly, cpoly, dpoly
    integer(kind = 4) :: epsin = 5

    ! Check inputs
    if ( nz .lt. 3 .or. nx .lt. 3 ) stop "Error: grid size nz, nx too small"
    if ( max(nz, nx) .gt. nmax ) stop "Error: must increase size of NMAX"
    if ( dz .le. 0.d0 .or. dx .le. 0.d0 ) stop "Error: grid spacing dz, dx too small"
    if ( n_sweep .lt. 1 ) stop "Error: wrong sweep number"
    if ( minval(slow) .le. 0.d0 .or. maxval(slow) .ge. 1.d0 ) stop "Error: slownesses are strange"
    if ( zsrc .lt. 0.d0 .or. zsrc .gt. dfloat(nz-1) * dz &
         .or. xsrc .lt. 0.d0 .or. xsrc .gt. dfloat(nx-1) * dx ) &
      stop "Error: source out of bounds"

    ! Convert src to grid position and try and take into account machine precision
    zsa = zsrc / dz + 1.d0
    xsa = xsrc / dx + 1.d0

    ! Try to handle edges simply for source due to precision
    if ( zsa .ge. dfloat(nz) ) zsa = zsa - zerr
    if ( xsa .ge. dfloat(nx) ) xsa = xsa - zerr

    ! Grid points to initialize source
    zsi = int(zsa)
    xsi = int(xsa)
    vzero = slow(zsi,xsi)

    ! Allocate work array for traveltimes
    tt = Big

    ! Do our best to initialize source
    dzu = dabs( zsa - dfloat(zsi) )
    dzd = dabs( dfloat(zsi+1) - zsa )
    dxw = dabs( xsa - dfloat(xsi) )
    dxe = dabs( dfloat(xsi+1) - xsa )
    iflag = 0

    ! Source seems close enough to a grid point in X and Y direction
    if ( min(dzu, dzd) .lt. zerr .and. min(dxw, dxe) .lt. zerr) then
      zsa = dnint(zsa)
      xsa = dnint(xsa)
      iflag = 1
    end if

    ! At least one of coordinates not close to any grid point in X and Y direction
    if ( min(dzu, dzd) .gt. zerr .or. min(dxw, dxe) .gt. zerr) then
      if ( min(dzu, dzd) .lt. zerr) zsa = dnint(zsa)
      if ( min(dxw, dxe) .lt. zerr) xsa = dnint(xsa)
      iflag = 2
    end if

    ! Oops we are lost, not sure this happens - fix src to nearest grid point
    if ( iflag .ne. 1 .and. iflag .ne. 2 ) then
      zsa = dnint(zsa)
      xsa = dnint(xsa)
      iflag = 3
    end if

    ! We know where src is - start first propagation
    select case(iflag)
    case(1)
      tt(nint(zsa),nint(xsa)) = 0.d0
    case(3)
      tt(nint(zsa),nint(xsa)) = 0.d0
    case(2)
      dzu = dabs( zsa - dfloat(zsi) )
      dzd = dabs( dfloat(zsi+1) - zsa )
      dxw = dabs( xsa - dfloat(xsi) )
      dxe = dabs( dfloat(xsi+1) - xsa )

      ! First initialize 4 points around source
      tt(zsi,xsi) = t_ana(zsi, xsi, dz, dx, zsa, xsa, vzero)
      tt(zsi+1,xsi) = t_ana(zsi+1, xsi, dz, dx, zsa, xsa, vzero)
      tt(zsi,xsi+1) = t_ana(zsi, xsi+1, dz, dx, zsa, xsa, vzero)
      tt(zsi+1,xsi+1) = t_ana(zsi+1 ,xsi+1, dz, dx, zsa, xsa, vzero)

      td = Big
      td(xsi+1) = vzero * dxe * dx
      dx2i = 1.d0 / (dx*dx)

      do j = xsi+2, nx
        vref = slow(zsi,j-1)
        td(j) = td(j-1) + dx * vref
        tauv = td(j) - vzero * dabs( dfloat(j) - xsa ) * dx
        tauev = td(j-1) - vzero * dabs( dfloat(j-1) - xsa ) * dx

        sgntz = 1
        sgntx = 1
        taue = tt(zsi+1,j-1) - t_ana(zsi+1, j-1, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, zsi+1, j, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dz2i = 1.d0 / (dzd*dzd) * dz
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dx + sgnrz * tzc / dzd ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dx * ta + sgnrz * tzc / dzd * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(zsi+1,j) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c

        sgntz = -1
        sgntx = 1
        taue = tt(zsi,j-1) - t_ana(zsi, j-1, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, zsi, j, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dz2i = 1.d0 / (dzu*dzu) * dz
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dx + sgnrz * tzc / dzu ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dx * ta + sgnrz * tzc / dzu * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(zsi,j) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c
      end do

      td(xsi) = vzero * dxw * dx
      do j = xsi-1, 1, -1
        vref = slow(zsi,j)
        td(j) = td(j+1) + dx * vref
        tauv = td(j) - vzero * dabs( dfloat(j) - xsa ) * dx
        tauev = td(j+1) - vzero * dabs( dfloat(j+1) - xsa ) * dx

        sgntz = 1
        sgntx = -1
        taue = tt(zsi+1,j+1) - t_ana(zsi+1, j+1, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, zsi+1, j, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dz2i = 1.d0 / (dzd*dzd) * dz
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dx + sgnrz * tzc / dzd ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dx * ta + sgnrz * tzc / dzd * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(zsi+1,j) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c

        sgntz = -1
        sgntx = -1
        taue = tt(zsi+1,j+1) - t_ana(zsi+1, j+1, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, zsi, j, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dz2i = 1.d0 / (dzu*dzu) * dz
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dx + sgnrz * tzc / dzu ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dx * ta + sgnrz * tzc / dzu * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(zsi,j) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c
      end do

      td = Big
      td(zsi+1) = vzero * dzd * dz
      dz2i = 1.d0 / (dz*dz)
      do i = zsi+2, nz
        vref = slow(i-1,xsi)
        td(i) = td(i-1) + dz * vref
        taue = td(i) - vzero * dabs(dfloat(i) - zsa) * dz
        tauev = td(i-1) - vzero * dabs(dfloat(i-1) - zsa) * dz

        sgntz = 1
        sgntx = 1
        tauv = tt(i-1,xsi+1) - t_ana(i-1, xsi+1, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, i, xsi+1, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dx2i = 1.d0 / (dxe*dxe) * dx
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dxe + sgnrz * tzc / dz ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dxe * ta + sgnrz * tzc / dz * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(i,xsi+1) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c

        sgntz = 1
        sgntx = -1
        tauv = tt(i-1,xsi) - t_ana(i-1, xsi, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, i, xsi, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dx2i = 1.d0 / (dxw*dxw) * dx
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dxw + sgnrz * tzc / dz ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dxw * ta + sgnrz * tzc / dz * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(i,xsi) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c
      end do

      td(zsi) = vzero * dzu * dz
      do i = zsi-1,1,-1
        vref = slow(i,xsi)
        td(i) = td(i+1) + dz * vref
        taue = td(i) - vzero * dabs( dfloat(i) - zsa ) * dz
        tauev = td(i+1) - vzero * dabs( dfloat(i+1) - zsa) * dz

        sgntz = -1
        sgntx = 1
        tauv = tt(i+1,xsi+1) - t_ana(i+1, xsi+1, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, i, xsi+1, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dx2i = 1.d0 / (dxe*dxe) * dx
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dxe + sgnrz * tzc / dz ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dxe * ta + sgnrz * tzc / dz * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(i,xsi+1) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c

        sgntz = -1
        sgntx = -1
        tauv = tt(i+1,xsi) - t_ana(i+1, xsi, dz, dx, zsa, xsa, vzero)
        t0c = t_anad(tzc, txc, i, xsi, dz, dx, zsa, xsa, vzero)
        sgnrz = dfloat(sgntz)
        sgnrx = dfloat(sgntx)
        ta = tauev + taue - tauv
        tb = tauev - taue + tauv
        dx2i = 1.d0 / (dxw*dxw) * dx
        apoly = dz2i + dx2i
        bpoly = 4.d0 * ( sgnrx * txc / dxw + sgnrz * tzc / dz ) - 2.d0 * ( ta * dx2i + tb * dz2i )
        cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
                - 4.d0 * ( sgnrx * txc / dxw * ta + sgnrz * tzc / dz * tb ) &
                + 4.d0 * ( vzero*vzero - vref*vref )
        dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
        if ( dpoly .ge. 0.d0 ) tt(i,xsi) = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c
      end do
    end select

    ! Precalculate constants
    dzi = 1.d0 / dz
    dxi = 1.d0 / dx
    dz2i = 1.d0 / (dz*dz)
    dx2i = 1.d0 / (dx*dx)
    dsum = dz2i + dx2i
    dz2dx2 = dz2i * dx2i

    ! Ready to do at least one global sweep
    do kk = 1, n_sweep
      ! First sweeping: Top->Bottom ; West->East
      sgntz = 1
      sgntx = 1
      sgnvz = 1
      sgnvx = 1
      sgnrz = dfloat(sgntz)
      sgnrx = dfloat(sgntx)

      do j = 2, nx
        do i = 2, nz
          include "Include_FTeik2d_grad.f"
        end do
      end do

      ! Second sweeping: Top->Bottom ; East->West
      sgntz = 1
      sgntx = -1
      sgnvz = 1
      sgnvx = 0
      sgnrz = dfloat(sgntz)
      sgnrx = dfloat(sgntx)
      do j = nx-1, 1, -1
        do i = 2, nz
          include "Include_FTeik2d_grad.f"
        end do
      end do

      ! Third sweep: Bottom->Top ; West->East
      sgntz = -1
      sgntx = 1
      sgnvz = 0
      sgnvx = 1
      sgnrz = dfloat(sgntz)
      sgnrx = dfloat(sgntx)
      do j = 2, nx
        do i = nz-1, 1, -1
          include "Include_FTeik2d_grad.f"
        end do
      end do

      ! Fourth sweeping: Bottom->Top ; East->West
      sgntz = -1
      sgntx = -1
      sgnvz = 0
      sgnvx = 0
      sgnrz = dfloat(sgntz)
      sgnrx = dfloat(sgntx)
      do j = nx-1, 1, -1
        do i = nz-1, 1, -1
          include "Include_FTeik2d_grad.f"
        end do
      end do

    end do

    if ( present(ttgrad) ) then
      ttgrad(:,:,1) = ttz
      ttgrad(:,:,2) = ttx
    end if
    return
  contains

    ! Functions to calculate analytical times in homgenous model
    real(kind = 8) function t_ana(i, j, dz, dx, zsa, xsa, vzero)
      integer(kind = 4), intent(in) :: i, j
      real(kind = 8), intent(in) :: dz, dx, zsa, xsa, vzero

      t_ana = vzero * ( ( ( dfloat(i) - zsa ) * dz )**2.d0 &
                      + ( ( dfloat(j) - xsa ) * dx )**2.d0 )**0.5d0
      return
    end function t_ana

    ! Functions to calculate analytical times in homgenous model + derivatives of times
    real(kind = 8) function t_anad(tzc, txc, i, j, dz, dx, zsa, xsa, vzero)
      integer(kind = 4), intent(in) :: i, j
      real(kind = 8), intent(in) :: dz, dx, zsa, xsa, vzero
      real(kind = 8) :: d0
      real(kind = 8), intent(out) :: tzc, txc

      d0 = ( ( dfloat(i) - zsa ) *dz )**2.d0 &
           + ( ( dfloat(j) - xsa ) *dx )**2.d0
      t_anad = vzero * (d0**0.5d0)
      if ( d0 .gt. 0.d0 ) then
        tzc = ( d0**(-0.5d0) ) * ( dfloat(i) - zsa ) * dz * vzero
        txc = ( d0**(-0.5d0) ) * ( dfloat(j) - xsa ) * dx * vzero
      else
        tzc = 0.d0
        txc = 0.d0
      end if
      return
    end function t_anad

  end subroutine solver2d

  subroutine solve(slow, tt, nz, nx, zs, xs, dz, dx, n_sweep, nsrc, n_threads)
    integer(kind = 4), intent(in) :: nz, nx, n_sweep, nsrc
    integer(kind = 4), intent(in), optional :: n_threads
    real(kind = 8), intent(in) :: slow(nz,nx), zs(nsrc), xs(nsrc), dz, dx
    real(kind = 8), intent(out) :: tt(nz, nx, nsrc)
    integer(kind = 4) :: k

    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    !$omp parallel default(shared)
    !$omp do schedule(runtime)
    do k = 1, nsrc
      call solver2d(slow, tt(:,:,k), nz, nx, zs(k), xs(k), dz, dx, n_sweep)
    end do
    !$omp end parallel
    return
  end subroutine solve

  function interp2(source, x, y, v, xq, yq) result(vq)
    real(kind = 8) :: vq
    real(kind = 8), intent(in) :: xq, yq
    real(kind = 8), dimension(:), intent(in) :: source, x, y
    real(kind = 8), dimension(:,:), intent(in) :: v
    integer(kind = 4) :: nx, ny, i1, i2, j1, j2
    real(kind = 8) :: N(4), ax(4), ay(4), av(4), ad(4)
    real(kind = 8) :: x1, x2, y1, y2, v11, v21, v12, v22, d11, d21, d12, d22

    nx = size(v, 1)
    ny = size(v, 2)
    i1 = minloc(xq - x, dim = 1, mask = xq .ge. x)
    j1 = minloc(yq - y, dim = 1, mask = yq .ge. y)
    i2 = i1 + 1
    j2 = j1 + 1

    if ( i1 .eq. nx .and. j1 .ne. ny ) then
      x1 = x(i1)
      x2 = 2.d0 * x1 - x(nx-1)
      y1 = y(j1)
      y2 = y(j2)
      d11 = dsqrt( sum( ( source - [x1,y1] ) * ( source - [x1,y1] ) ) )
      d21 = 0.d0
      d12 = dsqrt( sum( ( source - [x1,y2] ) * ( source - [x1,y2] ) ) )
      d22 = 0.d0
      v11 = v(i1,j1)
      v21 = 1.d0
      v12 = v(i1,j2)
      v22 = 1.d0
    else if ( i1 .ne. nx .and. j1 .eq. ny ) then
      x1 = x(i1)
      x2 = x(i2)
      y1 = y(j1)
      y2 = 2.d0 * y1 - y(ny-1)
      d11 = dsqrt( sum( ( source - [x1,y1] ) * ( source - [x1,y1] ) ) )
      d21 = dsqrt( sum( ( source - [x2,y1] ) * ( source - [x2,y1] ) ) )
      d12 = 0.d0
      d22 = 0.d0
      v11 = v(i1,j1)
      v21 = v(i2,j1)
      v12 = 1.d0
      v22 = 1.d0
    else if ( i1 .eq. nx .and. j1 .eq. ny ) then
      x1 = x(i1)
      x2 = 2.d0 * x1 - x(nx-1)
      y1 = y(j1)
      y2 = 2.d0 * y1 - y(ny-1)
      d11 = dsqrt( sum( ( source - [x1,y1] ) * ( source - [x1,y1] ) ) )
      d21 = 0.d0
      d12 = 0.d0
      d22 = 0.d0
      v11 = v(i1,j1)
      v21 = 1.d0
      v12 = 1.d0
      v22 = 1.d0
    else
      x1 = x(i1)
      x2 = x(i2)
      y1 = y(j1)
      y2 = y(j2)
      d11 = dsqrt( sum( ( source - [x1,y1] ) * ( source - [x1,y1] ) ) )
      d21 = dsqrt( sum( ( source - [x2,y1] ) * ( source - [x2,y1] ) ) )
      d12 = dsqrt( sum( ( source - [x1,y2] ) * ( source - [x1,y2] ) ) )
      d22 = dsqrt( sum( ( source - [x2,y2] ) * ( source - [x2,y2] ) ) )
      v11 = v(i1,j1)
      v21 = v(i2,j1)
      v12 = v(i1,j2)
      v22 = v(i2,j2)
    end if

    ax = [ x2, x1, x2, x1 ]
    ay = [ y2, y2, y1, y1 ]
    av = [ v11, v21, v12, v22 ]
    ad = [ d11, d21, d12, d22 ]
    N = dabs( (ax - xq) * (ay - yq) ) / dabs( (x2 - x1) * (y2 - y1) )
    vq = dsqrt( sum( ( source - [xq,yq] ) * ( source - [xq,yq] ) ) ) / dot_product(ad / av, N)
    return
  end function interp2

  function lay2tt(slow, nz, nx, dz, dx, zs, xs, ys, zr, xr, yr, &
                    n_sweep, nsrc, nrcv, n_threads) result(tt)
    real(kind = 8) :: tt(nrcv,nsrc)
    integer(kind = 4), intent(in) :: nz, nx, n_sweep, nsrc, nrcv
    integer(kind = 4), intent(in), optional :: n_threads
    real(kind = 8), intent(in) :: slow(nz,nx), dz, dx, zs(nsrc), xs(nsrc), ys(nsrc), &
      zr(nrcv), xr(nrcv), yr(nrcv)
    integer(kind = 4) :: i, j, k, n1, n2, nmax
    real(kind = 8) :: dhorz, dmax
    real(kind = 8), dimension(:), allocatable :: ax, az
    real(kind = 8), dimension(:,:), allocatable :: tmp, tcalc, rcv, src

    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    ! Switch sources and stations to minimize calls to eikonals
    n1 = min(nrcv, nsrc)
    n2 = max(nrcv, nsrc)
    if (n1 .eq. nsrc) then
      rcv = reshape([ xr, yr, zr ], shape = [ n2, 3 ], order = [ 1, 2 ])
      src = reshape([ xs, ys, zs ], shape = [ n1, 3 ], order = [ 1, 2 ])
    else
      rcv = reshape([ xs, ys, zs ], shape = [ n2, 3 ], order = [ 1, 2 ])
      src = reshape([ xr, yr, zr ], shape = [ n1, 3 ], order = [ 1, 2 ])
    end if

    ! Initialize tcalc
    allocate(tcalc(n2,n1))

    ! Compute traveltimes using an eikonal solver
    az = dz * [ ( dfloat(k-1), k = 1, nz ) ]
    !$omp parallel default(shared) private(tmp, dmax, nmax, ax, j, k, dhorz)
    !$omp do schedule(runtime)
    do i = 1, n1
      dmax = maxval( dsqrt( ( src(i,1) - rcv(:,1) ) * ( src(i,1) - rcv(:,1) ) + &
                            ( src(i,2) - rcv(:,2) ) * ( src(i,2) - rcv(:,2) ) ) )
      nmax = floor( dmax / dx ) + 2
      allocate(tmp(nz,nmax))
      ax = dx * [ ( dfloat(k-1), k = 1, nmax ) ]
      call solver2d(slow(:,:nmax), tmp, nz, nmax, src(i,3), 0.d0, dz, dx, n_sweep)
      do j = 1, n2
        dhorz = dsqrt( sum( ( src(i,1:2) - rcv(j,1:2) ) * ( src(i,1:2) - rcv(j,1:2) ) ) )
        tcalc(j,i) = interp2([ 0.d0, src(i,3) ], az, ax, tmp, rcv(j,3), dhorz)
      end do
      deallocate(tmp)
    end do
    !$omp end parallel

    ! Transpose to reshape to [ nrcv, nsrc ]
    if (n1 .eq. nrcv) then
      tt = transpose(tcalc)
    else
      tt = tcalc
    end if
    return
  end function lay2tt

end module fteik2d
