program fileIO
  implicit none
  integer i, j, k
  
  open(17,file='mydata.ads60', status='replace')
  write (17,*) 100, 200, 300
  write (17,*) 400
  write (17,*) 500, 600
  close(17)
  
  ! 同じファイルを今度は読み込み用にオープンする
  open(18,file='mydata.ads60', status='old')
  read (18,*) i, j, k
  print *, i, j, k
  read (18,*) i
  print *, i
  read (18,*) i, j
  print *, i, j
  close(18)
end program fileIO