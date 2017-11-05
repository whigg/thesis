      subroutine loaddata(filename, ret)
      real u,v,psl
      character*(*) filename
      real ret(21025,3)
      integer count

Cf2py intent(in) filename
Cf2py intent(out) ret

      filename = "ecm030101.ads60"
      open(65,file=filename,status='old',form='unformatted',
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

        print *, u,v, psl
        count = count + 1
41    continue
      close(65)

      end