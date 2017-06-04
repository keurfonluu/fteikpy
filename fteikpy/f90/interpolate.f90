function interp2(x, y, v, xq, yq) result(vq)
  real :: vq
  real, intent(in) :: xq, yq
  real, dimension(:), intent(in) :: x, y
  real, dimension(:,:), intent(in) :: v
  integer :: nx, ny, i1, i2, j1, j2
  real :: x1, x2, y1, y2, v11, v21, v12, v22
  real :: N(4), ax(4), ay(4), av(4)

  nx = size(v, 1)
  ny = size(v, 2)
  i1 = minloc(xq - x, dim = 1, mask = xq .ge. x)
  j1 = minloc(yq - y, dim = 1, mask = yq .ge. y)
  i2 = i1 + 1
  j2 = j1 + 1

  if ( i1 .eq. nx .and. j1 .ne. ny ) then
    x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2)
    v11 = v(i1,j1); v21 = 0.; v12 = v(i1,j2); v22 = 0.
  else if ( i1 .ne. nx .and. j1 .eq. ny ) then
    x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1)
    v11 = v(i1,j1); v21 = v(i2,j1); v12 = 0.; v22 = 0.
  else if ( i1 .eq. nx .and. j1 .eq. ny ) then
    x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1)
    v11 = v(i1,j1); v21 = 0.; v12 = 0.; v22 = 0.
  else
    x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2)
    v11 = v(i1,j1); v21 = v(i2,j1); v12 = v(i1,j2); v22 = v(i2,j2)
  end if

  ax = [ x2, x1, x2, x1 ]
  ay = [ y2, y2, y1, y1 ]
  av = [ v11, v21, v12, v22 ]
  N = abs( (ax - xq) * (ay - yq) ) / abs( (x2 - x1) * (y2 - y1) )
  vq = dot_product(av, N)
  return
end function interp2

function interp3(x, y, z, v, xq, yq, zq) result(vq)
    real :: vq
    real, intent(in) :: xq, yq, zq
    real, dimension(:), intent(in) :: x, y, z
    real, dimension(:,:,:), intent(in) :: v
    integer :: nx, ny, nz, i1, i2, j1, j2, k1, k2
    real :: x1, x2, y1, y2, z1, z2, v111, v211, v121, v221, &
                         v112, v212, v122, v222
    real :: N(8), ax(8), ay(8), az(8), av(8)

    nx = size(v, 1)
    ny = size(v, 2)
    nz = size(v, 3)
    i1 = minloc(xq - x, dim = 1, mask = xq .ge. x)
    j1 = minloc(yq - y, dim = 1, mask = yq .ge. y)
    k1 = minloc(zq - z, dim = 1, mask = zq .ge. z)
    i2 = i1 + 1
    j2 = j1 + 1
    k2 = k1 + 1

    x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = z(k2)
    v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = v(i1,j2,k1); v221 = v(i2,j2,k1)
    v112 = v(i1,j1,k2); v212 = v(i2,j1,k2); v122 = v(i1,j2,k2); v222 = v(i2,j2,k2)

    if ( i1 .eq. nx .and. j1 .ne. ny .and. k1 .ne. nz ) then
      x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = z(k2)
      v111 = v(i1,j1,k1); v211 = 0.; v121 = v(i1,j2,k1); v221 = 0.
      v112 = v(i1,j1,k2); v212 = 0.; v122 = v(i1,j2,k2); v222 = 0.
    else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
      x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = z(k2)
      v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = 0.; v221 = 0.
      v112 = v(i1,j1,k2); v212 = v(i2,j1,k2); v122 = 0.; v222 = 0.
    else if ( i1 .ne. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
      x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
      v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = v(i1,j2,k1); v221 = v(i2,j2,k1)
      v112 = 0.; v212 = 0.; v122 = 0.; v222 = 0.
    else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .ne. nz ) then
      x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = z(k2)
      v111 = v(i1,j1,k1); v211 = 0.; v121 = 0.; v221 = 0.
      v112 = v(i1,j1,k2); v212 = 0.; v122 = 0.; v222 = 0.
    else if ( i1 .eq. nx .and. j1 .ne. ny .and. k1 .eq. nz ) then
      x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
      v111 = v(i1,j1,k1); v211 = 0.; v121 = v(i1,j2,k1); v221 = 0.
      v112 = 0.; v212 = 0.; v122 = 0.; v222 = 0.
    else if ( i1 .ne. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
      x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
      v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = 0.; v221 = 0.
      v112 = 0.; v212 = 0.; v122 = 0.; v222 = 0.
    else if ( i1 .eq. nx .and. j1 .eq. ny .and. k1 .eq. nz ) then
      x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1); z1 = z(k1); z2 = 2.*z1 - z(nz-1)
      v111 = v(i1,j1,k1); v211 = 0.; v121 = 0.; v221 = 0.
      v112 = 0.; v212 = 0.; v122 = 0.; v222 = 0.
    else
      x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2); z1 = z(k1); z2 = z(k2)
      v111 = v(i1,j1,k1); v211 = v(i2,j1,k1); v121 = v(i1,j2,k1); v221 = v(i2,j2,k1)
      v112 = v(i1,j1,k2); v212 = v(i2,j1,k2); v122 = v(i1,j2,k2); v222 = v(i2,j2,k2)
    end if

    ax = [ x2, x1, x2, x1, x2, x1, x2, x1 ]
    ay = [ y2, y2, y1, y1, y2, y2, y1, y1 ]
    az = [ z2, z2, z2, z2, z1, z1, z1, z1 ]
    av = [ v111, v211, v121, v221, v112, v212, v212, v222 ]
    N = abs( (ax - xq) * (ay - yq) * (az - zq) ) / abs( (x2 - x1) * (y2 - y1) * (z2 -z1) )
    vq = dot_product(av, N)
    return
  end function interp3
