! Task CONV: call FLEXPART 11's own CONVECT (Emanuel 4.3c as FLEXPART sets it)
! on columns written by flexpart_convect.py, with the cloud-base mass flux
! remembered between calls as FLEXPART's convmix does (cbaseflux), and apply
! FLEXPART's calcmatrix/redist forward redistribution to a sub-cloud tracer.
!
! CONVECT and TLIFT are NOT in this file: build.sh extracts them from a local
! FLEXPART checkout (conv_mod.f90) into fp_convect.inc, which is included in
! module fp_state below. FLEXPART's compensating-subsidence displacement of
! particles that the matrix does not move (redist, wsub) is not applied: the
! tracer budget here is the matrix alone (see flexpart_convect.py).
!
! Input (stream, little-endian, native kinds): int32 ncol, nd, nl, ncalls, nspin;
! real32 delt; then per column real32 p_full(nd) [Pa], p_half(nd+1) [Pa,
! p_half(1) = surface pressure], t(nd) [K], q(nd) [kg/kg], all lowest level first.
! Output (stream): per column int32 iflag_last, nconvtop_last; real32 plcl [hPa],
! mindiag, cbmf(ncalls), tracer(nd) [kg m-2 of sub-cloud air per level after the
! calls after the first nspin (spin-up calls move no tracer)].

module par_mod
  implicit none
end module par_mod

module fp_state
  implicit none
  integer, parameter :: na = 200
  real, dimension(na, 1) :: pconv_hpa, phconv_hpa, sub, tconv, qconv, qsconv
  real, dimension(na, na, 1) :: fmass
  integer :: nconvtop
  integer, parameter :: nconvlevmax = na - 1
contains
  include 'fp_convect.inc'
end module fp_state

program fp_driver
  use fp_state
  use qvsat_mod, only: f_qvsat
  implicit none
  integer :: ncol, nd, nl, ncalls, nspin, ic, icall, k, kk, iflag, i
  real :: delt, cbmf, precip, wd, tprime, qprime, plcl, rh, chi, mindiag, rn_prev, cum, cmax
  real, allocatable :: pf(:), ph(:), tt(:), qq(:), tracer(:), tnew(:), mlev(:), cb(:), frac(:, :)
  integer :: nt

  open (10, file='in.bin', access='stream', form='unformatted', status='old')
  open (11, file='out.bin', access='stream', form='unformatted', status='replace')
  read (10) ncol, nd, nl, ncalls, nspin
  read (10) delt
  allocate (pf(nd), ph(nd + 1), tt(nd), qq(nd), tracer(nd), tnew(nd), mlev(nd), cb(ncalls))
  allocate (frac(nd, nd))
  do ic = 1, ncol
    read (10) pf, ph, tt, qq
    pconv_hpa = 0.; phconv_hpa = 0.; tconv = 0.; qconv = 0.; qsconv = 0.
    pconv_hpa(1:nd, 1) = pf/100.
    phconv_hpa(1:nd + 1, 1) = ph/100.
    tconv(1:nd, 1) = tt
    qconv(1:nd, 1) = qq
    do k = 1, nd
      qsconv(k, 1) = f_qvsat(pf(k), tt(k))
      mlev(k) = (ph(k) - ph(k + 1))/9.81
    end do
    ! FLEXPART's LCL of the lowest level (CONVECT computes it from level NK,
    ! the lowest level unless moist static energy peaks higher up).
    rh = qconv(1, 1)/qsconv(1, 1)
    chi = tconv(1, 1)/(1669.0 - 122.0*rh - tconv(1, 1))
    plcl = pconv_hpa(1, 1)*(rh**chi)
    ! Sub-cloud tracer: all of the air in the levels below the LCL.
    tracer = 0.
    do k = 1, nd
      if (pconv_hpa(k, 1) > plcl) tracer(k) = mlev(k)
    end do
    cbmf = 0.
    mindiag = 1.
    iflag = 0
    nt = 0
    do icall = 1, ncalls
      call convect(na - 1, nl, delt, iflag, precip, wd, tprime, qprime, cbmf, 1)
      cb(icall) = cbmf
      if (iflag /= 1 .and. iflag /= 4) cycle
      if (cbmf <= 0.) cycle
      if (icall <= nspin) cycle
      ! calcmatrix: fmassfrac = delt*fmass, diagonal closed to the layer mass.
      nt = nconvtop
      do k = 1, nt
        do kk = 1, nt
          frac(k, kk) = delt*fmass(k, kk, 1)
        end do
        frac(k, k) = frac(k, k) + mlev(k) - sum(frac(k, 1:nt))
        mindiag = min(mindiag, frac(k, k)/mlev(k))
      end do
      ! redist, forward: a particle in level k goes to the first level whose
      ! cumulative fraction reaches its uniform number, so the probability of
      ! level kk is the increment of the running maximum of the cumulative sum,
      ! clipped to [0, 1] (exact also when the diagonal is negative).
      tnew = tracer
      tnew(1:nt) = 0.
      do k = 1, nt
        if (tracer(k) == 0.) cycle
        cum = 0.
        cmax = 0.
        rn_prev = 0.
        do kk = 1, nt
          cum = cum + frac(k, kk)/mlev(k)
          cmax = min(1., max(cmax, cum))
          tnew(kk) = tnew(kk) + tracer(k)*(cmax - rn_prev)
          rn_prev = cmax
        end do
        ! rounding: what is left stays
        tnew(k) = tnew(k) + tracer(k)*(1. - rn_prev)
      end do
      tracer = tnew
    end do
    write (11) iflag, nt
    write (11) plcl, mindiag
    write (11) (cb(i), i=1, ncalls)
    write (11) tracer
  end do
  close (10)
  close (11)
end program fp_driver
