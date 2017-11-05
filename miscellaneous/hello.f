      real u,v

       open(65,file="ecm030101.ads60",status='old'
     + ,form='unformatted',access='direct',recl=4)

       irec=0
       do 41 j=1,145
       do 41 i=1,145
         irec=irec+1
         read(65,rec=irec) u
         irec=irec+1
         read(65,rec=irec) v
         irec=irec+1
         print *, u,v
 41    continue

       stop
       end