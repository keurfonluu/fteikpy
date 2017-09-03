module lay2vel

  implicit none
  
contains

  subroutine lay2vel1(vel, lay, nlay, dz, nz)
    integer, intent(in) :: nlay, nz
    real, intent(in) :: lay(nlay,2), dz
    real, intent(out) :: vel(nz)
    integer :: i, ztop, zbot
    
    ztop = 1
    do i = 1, nlay
      zbot = nint( lay(i,2) / dz )
      vel(ztop:zbot) = lay(i,1)
      ztop = zbot
    end do
    return
  end subroutine lay2vel1
  
  subroutine lay2vel2(vel, lay, nlay, dz, nz, nx)
    integer, intent(in) :: nlay, nz, nx
    real, intent(in) :: lay(nlay,2), dz
    real, intent(out) :: vel(nz, nx)
    integer :: ix
    real :: vel1d(nz)
    
    call lay2vel1(vel1d, lay, nlay, dz, nz)
    do ix = 1, nx
      vel(:,ix) = vel1d
    end do
    return
  end subroutine lay2vel2

  subroutine lay2vel3(vel, lay, nlay, dz, nz, nx, ny)
    integer, intent(in) :: nlay, nz, nx, ny
    real, intent(in) :: lay(nlay,2), dz
    real, intent(out) :: vel(nz, nx, ny)
    integer :: iy
    real :: vel2d(nz,nx)
    
    call lay2vel2(vel2d, lay, nlay, dz, nz, nx)
    do iy = 1, ny
      vel(:,:,iy) = vel2d
    end do
    return
  end subroutine lay2vel3
  
end module lay2vel
