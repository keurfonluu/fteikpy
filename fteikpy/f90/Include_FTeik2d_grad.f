  ! Index of velocity nodes
  i1 = i - sgnvz
  j1 = j - sgnvx

  ! Get local times of surrounding points
  tv = tt(i-sgntz,j)
  te = tt(i,j-sgntx)
  tev = tt(i-sgntz,j-sgntx)

  ! Get local velocity parameters
  vref = slow(i1,j1)                      ! Slowness

  ! 1D operators (refracted times)
  t1d1 = tv + dz * vref                   ! First dimension (Z axis)
  t1d2 = te + dx * vref                   ! Second dimension (X axis)
  t1d = min(t1d1, t1d2)

  ! 2D operators
  t2d = Big
  ta = tev + te - tv
  tb = tev - te + tv
  if ( ( tv .lt. te + dx*vref ) .and. ( te .lt. tv + dz*vref ) ) then
    t2d = ( ( tb * dz2i + ta * dx2i ) + sqrt( 4.d0 * vref*vref * ( dz2i + dx2i ) &
          - dz2i * dx2i * ( ta - tb ) * ( ta - tb ) ) ) / ( dz2i + dx2i )
  end if

  ! Select minimum time
  time_sol = [ tt(i,j), t1d1, t1d2, t2d ]
  imin = minloc(time_sol, dim = 1)
  tt(i,j) = time_sol(imin)

  ! Compute gradient according to minimum time direction
  if ( present(ttgrad)) then
    select case (imin)
    case (2)
      ttx(i,j) = 0.d0
      ttz(i,j) = sgntz * (tt(i,j)-tv) / dz
    case (3)
      ttx(i,j) = sgntx * (tt(i,j)-te) / dx
      ttz(i,j) = 0.d0
    case (4)
      ttx(i,j) = sgntx * (tt(i,j)-te) / dx
      ttz(i,j) = sgntz * (tt(i,j)-tv) / dz
    end select
  end if
