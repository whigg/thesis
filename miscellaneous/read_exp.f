      program read_exp
      character(len=15) x
      character(15),dimension(4108) :: data
c      character data*15

      open (17, file='output.csv')
      read (17, '()')
      do i = 1, 4108
       read (17, *) x
       print *, x
      data(i) = x
      end do
      close (17)

      end program read_exp
