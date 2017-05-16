function interp2(x, y, v, xq, yq) result(vq)
  real :: vq
  real, intent(in) :: xq, yq
  real, dimension(:), intent(in) :: x, y
  real, dimension(:,:), intent(in) :: v
  integer :: i, x1(1), y1(1), x2(1), y2(1), ix(4), iy(4)
  real :: vn, xr(2), yr(2), N(4), vr(4)
  
  x1 = minloc(xq - x, mask = xq .ge. x)
  y1 = minloc(yq - y, mask = yq .ge. y)
  x2 = maxloc(xq - x, mask = xq .lt. x)
  y2 = maxloc(yq - y, mask = yq .lt. y)
  vn = abs( (x(x2(1)) - x(x1(1))) &
          * (y(y2(1)) - y(y1(1))) )
  xr = x( [ x1(1), x2(1) ] )
  yr = y( [ y1(1), y2(1) ] )
  ix = [ 2, 1, 2, 1 ]
  iy = [ 2, 2, 1, 1 ]
  do i = 1, 4
    N(i) = abs( (xr(ix(i)) - xq) * (yr(iy(i)) - yq) )
  end do
  vr = reshape(v( [ x1(1), x2(1) ], &
                  [ y1(1), y2(1) ] ), shape = [ 4 ])
  vq = dot_product(vr, N/vn)
  return
end function interp2

function interp3(x, y, z, v, xq, yq, zq) result(vq)
    real :: vq
    real, intent(in) :: xq, yq, zq
    real, dimension(:), intent(in) :: x, y, z
    real, dimension(:,:,:), intent(in) :: v
    integer :: i, x1(1), y1(1), z1(1), x2(1), y2(1), z2(1), &
      ix(8), iy(8), iz(8)
    real :: vn, xr(2), yr(2), zr(2), N(8), vr(8)
    
    x1 = minloc(xq - x, mask = xq .ge. x)
    y1 = minloc(yq - y, mask = yq .ge. y)
    z1 = minloc(zq - z, mask = zq .ge. z)
    x2 = maxloc(xq - x, mask = xq .lt. x)
    y2 = maxloc(yq - y, mask = yq .lt. y)
    z2 = maxloc(zq - z, mask = zq .lt. z)
    vn = abs( (x(x2(1)) - x(x1(1))) &
            * (y(y2(1)) - y(y1(1))) &
            * (z(z2(1)) - z(z1(1))) )
    xr = x( [ x1(1), x2(1) ] )
    yr = y( [ y1(1), y2(1) ] )
    zr = z( [ z1(1), z2(1) ] )
    ix = [ 2, 1, 2, 1, 2, 1, 2, 1 ]
    iy = [ 2, 2, 1, 1, 2, 2, 1, 1 ]
    iz = [ 2, 2, 2, 2, 1, 1, 1, 1 ]
    do i = 1, 8
      N(i) = abs( (xr(ix(i)) - xq) * (yr(iy(i)) - yq) * (zr(iz(i)) - zq) )
    end do
    vr = reshape(v( [ x1(1), x2(1) ], &
                    [ y1(1), y2(1) ], &
                    [ z1(1), z2(1) ] ), shape = [ 8 ])
    vq = dot_product(vr, N/vn)
    return
  end function interp3