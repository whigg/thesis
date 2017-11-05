program hello
  real u,v
  open(52,file='ecm030101.ads60', form='unformatted',access='direct',recl=4, status='old')
  do i = 0, 2, 1
   read (52, *) u, v
   print *, u, v
  end do
  close(52)
end program hello