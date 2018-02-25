  ! Index of velocity nodes
  i1 = i - sgnvz
  j1 = j - sgnvx

  ! Get local times of surrounding points
  tv = tt(i-sgntz,j)
  te = tt(i,j-sgntx)
  tev = tt(i-sgntz,j-sgntx)

  ! 1D operators (refracted times)
  vref = min( slow(i1,max(j-1,1)), slow(i1,min(j,nx-1)) )
  t1d1 = tv + dz * vref                   ! First dimension (Z axis)
  vref = min( slow(max(i-1,1),j1), slow(min(i,nz-1),j1) )
  t1d2 = te + dx * vref                   ! Second dimension (X axis)

  ! 2D operators
  t2d = Big
  t1 = Big
  t2 = Big
  t3 = Big
  vref = slow(i1,j1)

  ! Choose plane wave or spherical
  ! Test for plane wave
  if ( .not. perturbation .or. ( abs(i-zsi) .gt. epsin .or. abs(j-xsi) .gt. epsin ) ) then

    ! 4 points operator if possible, otherwise do three points
    if ( ( tv .le. te + dx*vref ) .and. ( te .le. tv + dz*vref ) &
      .and. ( te - tev .ge. 0.d0 ) .and. ( tv - tev .ge. 0.d0 ) ) then
      ta = tev + te - tv
      tb = tev - te + tv
      t1 = ( ( tb * dz2i + ta * dx2i ) + sqrt( 4.d0 * vref*vref * ( dz2i + dx2i ) &
           - dz2i * dx2i * ( ta - tb ) * ( ta - tb ) ) ) / ( dz2i + dx2i )

    ! Two 3 points operators
    else if ( ( te - tev ) .le. dz*dz * vref / sqrt( dx*dx + dz*dz ) &
      .and. ( te - tev ) .gt. 0.d0 ) then
      t2 = te + dx * sqrt( vref*vref - ( ( te - tev ) / dz )**2.d0 )

    else if ( ( tv - tev ) .le. dx*dx * vref / sqrt( dx*dx + dz*dz ) &
      .and. ( tv - tev ) .gt. 0.d0 ) then
      t3 = tv + dz * sqrt( vref*vref - ( ( tv - tev ) / dx )**2.d0 )
    end if

  ! Test for spherical
  else
    ! Do spherical operator if conditions ok
    if ( ( tv .lt. te + dx*vref ) .and. ( te .lt. tv + dz*vref ) &
      .and. ( te - tev .ge. 0.d0 ) .and. ( tv - tev .ge. 0.d0 ) ) then
      t0c = t_anad(tzc, txc, i, j, dz, dx, zsa, xsa, vzero)
      tauv = tv - t_ana(i-sgntz, j, dz, dx, zsa, xsa, vzero)
      taue = te - t_ana(i, j-sgntx, dz, dx, zsa, xsa, vzero)
      tauev = tev - t_ana(i-sgntz, j-sgntx, dz, dx, zsa, xsa, vzero)
      ta = tauev + taue - tauv
      tb = tauev - taue + tauv
      apoly = dz2i + dx2i
      bpoly = 4.d0 * ( dfloat(sgntx) * txc * dxi + dfloat(sgntz) * tzc * dzi ) &
              - 2.d0 * ( ta * dx2i + tb * dz2i )
      cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
              - 4.d0 * ( dfloat(sgntx) * txc * dxi * ta + dfloat(sgntz) * tzc * dzi * tb ) &
              + 4.d0 * ( vzero*vzero - vref*vref )
      dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
      if ( dpoly .ge. 0.d0 ) t1 = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c
      if ( t1-tv .lt. 0.d0 .or. t1-te .lt. 0.d0 ) t1 = Big
    end if

  end if

  t2d = min(t1, t2, t3)

  ! Select minimum time
  time_sol = [ tt(i,j), t1d1, t1d2, t2d ]
  imin = minloc(time_sol, dim = 1)
  tt(i,j) = time_sol(imin)

  ! Compute gradient according to minimum time direction
  if ( present(ttgrad) ) then
    select case(imin)
    case(2)
      ttz(i,j) = sgntz * (tt(i,j)-tv) / dz
      ttx(i,j) = 0.d0
    case(3)
      ttz(i,j) = 0.d0
      ttx(i,j) = sgntx * (tt(i,j)-te) / dx
    case(4)
      ttz(i,j) = sgntz * (tt(i,j)-tv) / dz
      ttx(i,j) = sgntx * (tt(i,j)-te) / dx
    end select
  end if
