      real u,v,psl

       open(65,file="ecm030101.ads60",status='old'
     + ,form='unformatted',access='direct',recl=4)
       open(50,file="wind.csv",status='replace'
     + ,form='unformatted',access='direct',recl=4)
       irec=0
       do 41 j=1,145
       do 41 i=1,145
         irec=irec+1
         read(65,rec=irec) u
         write(50,rec=irec) u
         irec=irec+1
         read(65,rec=irec) v
         write(50,rec=irec) v
         irec=irec+1
         read(65,rec=irec) psl
         write(50,rec=irec) psl
c         irec=irec+1
c         print *, j, i
         print *, u,v, psl
 41    continue
       close(50)
       stop
       end