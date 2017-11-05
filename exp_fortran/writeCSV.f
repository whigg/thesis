      program writeCSV
      
      implicit none
      integer, parameter :: n = 5
      integer i
      character linebuf*256
      real :: x(n) = (/ 1.2, 0.02, 0.0003, 4.2, 5.99 /)
      real :: y(n) = (/ 9.0, 0.0008, 0.37, 100000.6, 500.2 /)
      open (18, file='mydata1.csv', status='replace')
      do i = 1, n
        write (linebuf, *) x(i), ',', y(i) ! 一旦内部ファイルへ書き出す
        call del_spaces(linebuf)           ! 余分な空白を削除する
        write (18, '(a)') trim(linebuf)    ! 出力する
      end do
      close (18)
      contains

      subroutine del_spaces(s)
        character (*), intent (inout) :: s
        character (len=len(s)) tmp
        integer i, j
        j = 1
        do i = 1, len(s)
          if (s(i:i)==' ') cycle
          tmp(j:j) = s(i:i)
          j = j + 1
        end do
        s = tmp(1:j-1)
      end subroutine del_spaces

      end program writeCSV