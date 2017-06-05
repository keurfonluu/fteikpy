function interp2(source, x, y, v, xq, yq) result(vq)
  real :: vq
  real, intent(in) :: xq, yq
  real, dimension(:), intent(in) :: source, x, y
  real, dimension(:,:), intent(in) :: v
  integer :: nx, ny, i1, i2, j1, j2
  real :: x1, x2, y1, y2, v11, v21, v12, v22, d11, d21, d12, d22
  real :: N(4), ax(4), ay(4), av(4), ad(4)

  nx = size(v, 1)
  ny = size(v, 2)
  i1 = minloc(xq - x, dim = 1, mask = xq .ge. x)
  j1 = minloc(yq - y, dim = 1, mask = yq .ge. y)
  i2 = i1 + 1
  j2 = j1 + 1

  if ( i1 .eq. nx .and. j1 .ne. ny ) then
    x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = y(j2)
    d11 = sqrt(sum((source-[x1,y1])**2)); d21 = 0.
    d12 = sqrt(sum((source-[x1,y2])**2)); d22 = 0.
    v11 = v(i1,j1); v21 = 1.; v12 = v(i1,j2); v22 = 1.
  else if ( i1 .ne. nx .and. j1 .eq. ny ) then
    x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = 2.*y1 - y(ny-1)
    d11 = sqrt(sum((source-[x1,y1])**2)); d21 = sqrt(sum((source-[x2,y1])**2))
    d12 = 0.; d22 = 0.
    v11 = v(i1,j1); v21 = v(i2,j1); v12 = 1.; v22 = 1.
  else if ( i1 .eq. nx .and. j1 .eq. ny ) then
    x1 = x(i1); x2 = 2.*x1 - x(nx-1); y1 = y(j1); y2 = 2.*y1 - y(ny-1)
    d11 = sqrt(sum((source-[x1,y1])**2)); d21 = 0.
    d12 = 0.; d22 = 0.
    v11 = v(i1,j1); v21 = 1.; v12 = 1.; v22 = 1.
  else
    x1 = x(i1); x2 = x(i2); y1 = y(j1); y2 = y(j2)
    d11 = sqrt(sum((source-[x1,y1])**2)); d21 = sqrt(sum((source-[x2,y1])**2))
    d12 = sqrt(sum((source-[x1,y2])**2)); d22 = sqrt(sum((source-[x2,y2])**2))
    v11 = v(i1,j1); v21 = v(i2,j1); v12 = v(i1,j2); v22 = v(i2,j2)
  end if

  ax = [ x2, x1, x2, x1 ]
  ay = [ y2, y2, y1, y1 ]
  av = [ v11, v21, v12, v22 ]
  ad = [ d11, d21, d12, d22 ]
  N = abs( (ax - xq) * (ay - yq) ) / abs( (x2 - x1) * (y2 - y1) )
  vq = sqrt(sum((source-[xq,yq])**2)) / dot_product(ad / av, N)
  return
end function interp2

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
