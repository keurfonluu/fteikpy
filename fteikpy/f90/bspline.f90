!=======================================================================
! Created by
!     Keurfon Luu <keurfon.luu@mines-paristech.fr>
!     MINES ParisTech - Centre de Géosciences
!     PSL - Research University
!=======================================================================

module bspline

  use omp_lib

  implicit none

contains

  recursive function deboor(i, k, x, t) result(db)
    real :: db
    integer, intent(in) :: i, k
    real, intent(in) :: x
    real, dimension(:), intent(in) :: t
    real :: A1, A2

    if ( k .eq. 1 ) then
      if ( x .ne. t(size(t)) ) then
        if ( ( x .ge. t(i) ) .and. ( x .lt. t(i+1) ) ) then
          db = 1.
        else
          db = 0.
        end if
      else
        if ( ( x .ge. t(i) ) .and. ( x .le. t(i+1) ) ) then
          db = 1.
        else
          db = 0.
        end if
      end if
    else
      if ( t(i+k-1) - t(i) .ne. 0. ) then
        A1 = ( x - t(i) ) / ( t(i+k-1) - t(i) )
      else
        A1 = 0.
      end if
      if ( t(i+k) - t(i+1) .ne. 0. ) then
        A2 = ( t(i+k) - x ) / ( t(i+k) - t(i+1) )
      else
        A2 = 0.
      end if
      db = A1 * deboor(i, k-1, x, t) + A2 * deboor(i+1, k-1, x, t)
    end if
    return
  end function deboor

  subroutine bsplrep1(x, y, xq, yq, n, nq, order)
    integer, intent(in) :: n, nq
    real, intent(in) :: x(n), y(n)
    real, intent(out) :: xq(nq), yq(nq)
    integer, intent(in), optional :: order
    integer :: i, iq, j, k
    real :: w
    real, dimension(:), allocatable :: t, t1, t2, t3, y1

    k = 4
    if ( present(order) ) k = order

    allocate(t1(k-1), t3(k-1))
    xq = 0.
    yq = 0.
    t1 = 0.
    t2 = 1. / float(n-k+1) * [ ( i-1, i = 1, n-k+2 ) ]
    t3 = 1.
    t = [ t1, t2, t3 ]
    y1 = 1. / float(nq-1) * [ ( i-1, i = 1, nq ) ]

    do iq = 1, nq
      j = max( k, maxloc( y1(iq) - t, 1, mask = y1(iq) .le. t ) - 1 )
      do i = j-k+1, j
        w = deboor(i, k, y1(iq), t)
        xq(iq) = xq(iq) + x(i) * w
        yq(iq) = yq(iq) + y(i) * w
      end do
    end do
    return
  end subroutine bsplrep1

  function spline1(x, y, xq, n, nq) result(yq)
    integer, intent(in) :: n, nq
    real, intent(in) :: x(n), y(n), xq(nq)
    integer :: i, x1
    real :: yq(nq)
    real, dimension(:), allocatable :: w, h, z, a, b, c, d

    allocate(w(n-1), h(n-1), z(n), a(n-1), b(n-1), c(n-1), d(n-1))

    ! Compute h and b
    do i = 1, n-1
      w(i) = x(i+1) - x(i)
      h(i) = ( y(i+1) - y(i) ) / w(i)
    end do

    ! Compute z
    z(1) = 0.
    do i = 1, n-2
      z(i+1) = 3. * ( h(i+1) - h(i) ) / ( w(i+1) + w(i) )
    end do
    z(n) = 0.

    ! Basis functions
    do i = 1, n-1
      a(i) = ( z(i+1) - z(i) ) / ( 6. * w(i) )
      b(i) = 0.5 * z(i)
      c(i) = h(i) - w(i) * ( z(i+1) + 2.*z(i) ) / 6.
      d(i) = y(i)
    end do

    ! Evaluate
    do i = 1, nq
      x1 = max( 1, minloc( xq(i) - x, 1, mask = xq(i) .gt. x) )
      yq(i) = d(x1) + ( xq(i) - x(x1) ) &
              * ( c(x1) + ( xq(i) - x(x1) ) &
              * ( b(x1) + ( xq(i) - x(x1) ) &
              * a(x1) ) )
    end do
    return
  end function spline1

  function bspline1(x, y, xq, n, nq, order) result(yq)
    integer, intent(in) :: n, nq
    real, intent(in) :: x(n), y(n), xq(nq)
    integer, intent(in), optional :: order
    integer :: k
    real :: yq(nq)
    real, dimension(:), allocatable :: bspl_x, bspl_y

    k = 4
    if ( present(order) ) k = order

    allocate(bspl_x(nq), bspl_y(nq))
    call bsplrep1(x, y, bspl_x, bspl_y, n, nq, k)
    yq = spline1(bspl_x, bspl_y, xq, nq, nq)
    return
  end function bspline1

  function bspline2(x, y, z, xq, yq, m, n, mq, nq, order, n_threads) result(zq)
    integer, intent(in) :: m, n, mq, nq
    real, intent(in) :: x(n), y(m), z(m,n), xq(mq,nq), yq(mq,nq)
    integer, intent(in), optional :: order, n_threads
    integer :: i, k
    real :: zq(mq,nq), tmp(m,nq)

    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    k = 4
    if ( present(order) ) k = order

    !$omp parallel default(shared)
    !$omp do schedule(runtime)
    do i = 1, m
      tmp(i,:) = bspline1(x, z(i,:), xq(i,:), n, nq, k)
    end do
    !$omp end parallel

    !$omp parallel default(shared)
    !$omp do schedule(runtime)
    do i = 1, nq
      zq(:,i) = bspline1(y, tmp(:,i), yq(:,i), m, mq, k)
    end do
    !$omp end parallel
    return
  end function bspline2

end module bspline
