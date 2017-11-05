      program readcsv
      character(len=15) x(1)

      open (17, file='output.csv')
      read (17, '()')
      do i = 1, 4108
       read (17, *) x
       print *, x
      end do
      close (17)

      end program readcsv
