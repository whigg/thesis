      program data2csv
        real u,v,psl
        character(15),dimension(2084) :: filename
        real ret(145*145,3)
        integer count
        character(len=13) fname
        character(len=15) x
        character(15),dimension(2084) :: data

        open (17, file='output1.csv')
        read (17, '()')
        do i = 1, 2084
         read (17, *) x
         data(i) = x
        end do
        close (17)

        do k=1,2084
         ! 本当はk=2084
         
         open(65,file=data(k),status='old',form='unformatted',
     + access='direct',recl=4)

         irec=0
         count = 1
         do 41 j=1,145
         do 41 i=1,145
           irec=irec+1
           read(65,rec=irec) u
           ret(count,1) = u
           irec=irec+1
           read(65,rec=irec) v
           ret(count,2) = v
           irec=irec+1
           read(65,rec=irec) psl
           ret(count,3) = psl
c          print *, u,v, psl
           count = count + 1
41        continue
        
         fname = data(k)(1:10) // 'csv'
         print *, k, data(k), ' --> ', fname
         call writeCSV(ret, fname)
         end do
        close(65)

      stop
      contains

      subroutine writeCSV(vec, fname)
c       読み込みのタイミングは41の直前（kのループ内）
c       入力vは(145*145,3)の行列を仮定
c       fnameは出力すべきcsvの名前

        implicit none
        real :: vec(145*145,3)
        character*(*) fname
        integer, parameter :: n = 145*145
        integer i
        character linebuf*256
        real :: x(n)
        real :: y(n)
        real :: z(n)

        open (18, file=fname, status='replace')
        do i = 1, n
          x(i) = vec(i,1)
          y(i) = vec(i,2)
          z(i) = vec(i,3)
          write (linebuf, *) x(i), ',', y(i), ',', z(i)
          ! 一旦内部ファイルへ書き出す
          call del_spaces(linebuf)
          ! 余分な空白を削除する
          write (18, '(a)') trim(linebuf)    ! 出力する
        end do
        close (18)
      end subroutine writeCSV
        
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


      end program data2csv

