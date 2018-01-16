  ! Index of velocity nodes
  i1 = i - sgnvz
  j1 = j - sgnvx

  ! Get local times of surrounding points
  tv = tt(i-sgntz,j)
  te = tt(i,j-sgntx)
  tev = tt(i-sgntz,j-sgntx)

  ! Get analytical solution, if using pertubation
  t0c = t_anad(tzc, txc, i, j, dz, dx, zsa, xsa, vzero)

  ! Convert times into pertubations
  tauv = tv - t_ana(i-sgntz, j, dz, dx, zsa, xsa, vzero)
  taue = te - t_ana(i, j-sgntx, dz, dx, zsa, xsa, vzero)
  tauev = tev - t_ana(i-sgntz, j-sgntx, dz, dx, zsa, xsa, vzero)

  ! 1D operators, (refracted times),set times to BIG
  t1d = Big
  t1 = Big
  t2 = Big

  ! V plane
  vref = min( slow(i1,max(j-1,1)), slow(i1,min(j,nx-1)) )
  t1 = tv + dz * vref

  ! WE plane
  vref = min( slow(max(i-1,1),j1), slow(min(i,nz-1),j1) )
  t2= te + dx * vref

  ! End of 1D operators (just take smallest time)
  t1d = min(t1, t2)

  ! 2D operators, and diagonal operators
  t2d = Big
  t1 = Big
  t2 = Big
  t3 = Big
  tdiag = Big
  vref = slow(i1,j1)                      ! Slowness

  ! Diagonal operator
  tdiag = tev + vref * sqrt( dx*dx + dz*dz )

  ! Choose spherical or plane wave
  ! First test for Plane wave
  if ( ( abs(i-zsi) .gt. epsin .or. abs(j-xsi) .gt. epsin ) ) then

    ! 4 Point operator, if possible otherwise do three points
    if ( tv .le. te+dx*vref .and. te .le. tv+dz*vref &
      .and. te-tev .ge. 0.d0 .and. tv-tev .ge. 0.d0 ) then
      ta = tev + te - tv
      tb = tev - te + tv
      t1 = ( ( tb * dz2i + ta * dx2i ) + sqrt( 4.d0 * vref*vref * ( dz2i + dx2i ) &
           - dz2i * dx2i * ( ta - tb ) * ( ta - tb ) ) ) / ( dz2i + dx2i )

    ! Two 3 point operators
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
    if ( tv .lt. te + dx*vref .and. te .lt. tv + dz*vref &
      .and. te-tev .ge. 0.d0 .and. tv-tev .ge. 0.d0 ) then
      ta = tauev + taue - tauv   ! X
      tb = tauev - taue + tauv   ! Z
      apoly = dz2i + dx2i
      bpoly = 4.d0 * ( sgnrx * txc * dxi + sgnrz * tzc * dzi ) &
              - 2.d0 * ( ta * dx2i + tb * dz2i )
      cpoly = ( ta*ta * dx2i ) + ( tb*tb * dz2i ) &
              - 4.d0 * ( sgnrx * txc * dxi * ta + sgnrz * tzc * dzi * tb ) &
              + 4.d0 * ( vzero*vzero - vref*vref )
      dpoly = bpoly*bpoly - 4.d0 * apoly * cpoly
      if ( dpoly .ge. 0.d0 ) t1 = 0.5d0 * ( sqrt(dpoly) - bpoly ) / apoly + t0c
      if ( t1-tv .lt. 0.d0 .or. t1-te .lt. 0.d0 ) t1 = Big
    end if
  end if

  ! End of 2D operators
  t2d = min(t1, t2, t3)

  ! Shortest path of 1D and 2D and diagonal
  tt(i,j) = min(tt(i,j), t1d, t2d)
