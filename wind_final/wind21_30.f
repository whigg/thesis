c 
c    Make a 37.5km grid wind data from NCEP/NCAR SLP 
c

      real u(145,145),v(145,145),psl(145,145),p(147,147)
      real u2( 145,145),v2(145,145),psl2(145,145)
      real slp(480,241),no,eo,nn(147,147),xc,yc
      integer yy,mm,yy2,ico
      character file2*64,file3*64,iii*2,mmo*2,ms*2,ys*2
      real slat,alat,alon,rsl,rnl,rwl,rel
      real re,rhu,dlon,fr,sum,dis
      real sgn,delta
      integer numy(2,3),dir
      double precision e2,PI,e,xq,yq,iq,jq
      real xa,ya,za,xb,yb,zb
      real kai1,kai2
      real deg2rad,rad2deg,r0,phi0,rate,di,dj,alpha,a2,phi,dlat
      character(13),dimension(95) :: data
      character(len=2) tmp
      character(len=2) tmp2

      re=6378.273
      e2=0.006693883
      pi=3.141592654
      e=sqrt(e2)


      dlat = 54.36
      xc = ( 900 + 1 ) * 0.5
      yc = ( 900 + 1 ) * 0.5
      dir = 1
      PI = 4.0 * atan(1.0)
      deg2rad = PI/180.0
      rad2deg = 180.0/PI
      r0 = sqrt((xc-1)*(xc-1)+(yc-1)*(yc-1))
      phi0 = dlat * deg2rad
      rate = r0 * (1.0+cos(phi0)) / sin(phi0);


      data(1) = "slp200307.dat"
      data(2) = "slp200308.dat"
      data(3) = "slp200309.dat"
      data(4) = "slp200310.dat"
      data(5) = "slp200311.dat"
      data(6) = "slp200312.dat"
      data(7) = "slp200407.dat"
      data(8) = "slp200408.dat"
      data(9) = "slp200409.dat"
      data(10) = "slp200410.dat"
      data(11) = "slp200411.dat"
      data(12) = "slp200412.dat"
      data(13) = "slp200507.dat"
      data(14) = "slp200508.dat"
      data(15) = "slp200509.dat"
      data(16) = "slp200510.dat"
      data(17) = "slp200511.dat"
      data(18) = "slp200512.dat"
      data(19) = "slp200607.dat"
      data(20) = "slp200608.dat"
      data(21) = "slp200609.dat"
      data(22) = "slp200610.dat"
      data(23) = "slp200611.dat"
      data(24) = "slp200612.dat"
      data(25) = "slp200707.dat"
      data(26) = "slp200708.dat"
      data(27) = "slp200709.dat"
      data(28) = "slp200710.dat"
      data(29) = "slp200711.dat"
      data(30) = "slp200712.dat"
      data(31) = "slp200807.dat"
      data(32) = "slp200808.dat"
      data(33) = "slp200809.dat"
      data(34) = "slp200810.dat"
      data(35) = "slp200811.dat"
      data(36) = "slp200812.dat"
      data(37) = "slp200907.dat"
      data(38) = "slp200908.dat"
      data(39) = "slp200909.dat"
      data(40) = "slp200910.dat"
      data(41) = "slp200911.dat"
      data(42) = "slp200912.dat"
      data(43) = "slp201007.dat"
      data(44) = "slp201008.dat"
      data(45) = "slp201009.dat"
      data(46) = "slp201010.dat"
      data(47) = "slp201011.dat"
      data(48) = "slp201012.dat"
      data(49) = "slp201307.dat"
      data(50) = "slp201308.dat"
      data(51) = "slp201309.dat"
      data(52) = "slp201310.dat"
      data(53) = "slp201311.dat"
      data(54) = "slp201312.dat"
      data(55) = "slp201404.dat"
      data(56) = "slp201405.dat"
      data(57) = "slp201406.dat"
      data(58) = "slp201407.dat"
      data(59) = "slp201408.dat"
      data(60) = "slp201409.dat"
      data(61) = "slp201410.dat"
      data(62) = "slp201411.dat"
      data(63) = "slp201412.dat"
      data(64) = "slp201501.dat"
      data(65) = "slp201502.dat"
      data(66) = "slp201503.dat"
      data(67) = "slp201504.dat"
      data(68) = "slp201505.dat"
      data(69) = "slp201506.dat"
      data(70) = "slp201507.dat"
      data(71) = "slp201508.dat"
      data(72) = "slp201509.dat"
      data(73) = "slp201510.dat"
      data(74) = "slp201511.dat"
      data(75) = "slp201512.dat"
      data(76) = "slp201601.dat"
      data(77) = "slp201602.dat"
      data(78) = "slp201603.dat"
      data(79) = "slp201604.dat"
      data(80) = "slp201605.dat"
      data(81) = "slp201606.dat"
      data(82) = "slp201607.dat"
      data(83) = "slp201608.dat"
      data(84) = "slp201609.dat"
      data(85) = "slp201610.dat"
      data(86) = "slp201611.dat"
      data(87) = "slp201612.dat"
      data(88) = "slp201701.dat"
      data(89) = "slp201702.dat"
      data(90) = "slp201703.dat"
      data(91) = "slp201704.dat"
      data(92) = "slp201705.dat"
      data(93) = "slp201706.dat"
      data(94) = "slp201707.dat"
      data(95) = "slp201708.dat"

      do 800 iter=21,30

      open(52,file=
     +data(iter)
     +,access='direct',form='unformatted',recl=4,status='old')

      tmp = data(iter)(6:7)
      read (tmp,*) yy

      tmp2 = data(iter)(8:9)
      read (tmp2,*) mm

c      iy = yy-2
      print *, yy, mm

c      do 800 iy=1,15

c      yy=2+iy
c      mm=1
      ico=1
      idn=0
      inum0=0

      if(yy.ge.10) write(ys,501) yy
      if(yy.lt.10) write(ys,502) 0,yy

c      do 2 jjj=1,3

        if ((mm.eq.12).or.(mm.eq.01).or.(mm.eq.03)
     +   .or.(mm.eq.05).or.(mm.eq.07).or.(mm.eq.08).or.(mm.eq.10))
     +   ed = 31
        if (mm.eq.02) then
          if ((yy.eq.4).or.(yy.eq.8).or.(yy.eq.12).or.(yy.eq.16)
     +       .or.(yy.eq.20).or.(yy.eq.24)) then
            ed = 29
          else
            ed = 28
          endif
        endif 
        if ((mm.eq.04).or.(mm.eq.06).or.(mm.eq.09).or.(mm.eq.11))
     +    ed = 30 

        ied = ed

        if(mm.ge.10) write(ms,501) mm
        if(mm.lt.10) write(ms,502) 0,mm
 501    format(i2)
 502    format(i1,i1)

        do 21 iday=1,ied
          idn=idn+1

c          print *, jjj

          do 1 k=1,241
            do 1 j=1,480
              inum0=inum0+1
              read(52,rec=inum0) rslp
              slp(j,k)=rslp/100.
1         continue    

          do 12 iw1=1,147
            do 12 jw1=1,147

              iq=21.+((iw1-2)*6.)
              jq=901-(21.+((jw1-2)*6.))
              di = xc - real(iq)
              dj = real(jq) - yc
              alpha = sqrt(di*di+dj*dj) / rate
              a2 = alpha * alpha;
              phi = acos( (1.0-a2)/(1.0+a2) )
              alon = atan2(dir*di,dj) * rad2deg
              alat = dir * ( 90.0 - phi * rad2deg )
              if( alon > 180.0 ) then
                alon = alon - 360.0
              end if
              if(alon.le.0.0) alon=alon+360.0
              if(alon.ge.360.0) alon=alon-360.0
              nn(iw1,jw1)=alat

c              if ((iw1.eq.2).and.(jw1.eq.2)) print *, alat,alon

              alat=pi*alat/180.
              alon=pi*alon/180.

              xa=cos(alat)*cos(alon)
              ya=cos(alat)*sin(alon)
              za=sin(alat)

c              if ((iw1.eq.146).and.(jw1.eq.146)) print *, alat,alon
              
              sum=0.
              p(iw1,jw1)=0.

              do 604 j=1,480
                do 604 k=1,100
                  sla=90.-(k-1)/1.3333
                  slo=(j-1)/1.3333
                  blat=pi*sla/180.
                  blon=pi*slo/180.
                  xb=cos(blat)*cos(blon)
                  yb=cos(blat)*sin(blon)
                  zb=sin(blat)

                  kai1=(xb-xa)**2.+(yb-ya)**2.+(zb-za)**2.
                  kai2=re*sqrt(kai1)
                  dis=2.*re*asin(kai2/re/2.)

                  if (dis.ge.500.) goto 604
                  fr=exp(1.5*(-dis**2./180.**2.))
                  sum=sum+fr
                  p(iw1,jw1)=p(iw1,jw1)+fr*slp(j,k)


604          continue    
              p(iw1,jw1)=p(iw1,jw1)/sum

12        continue


          do 13 k=1,145
            do 13 j=1,145

              j2=j+1
              k2=k+1

              rhu=1.301
              ref=2*(2*PI/86400)*sin(nn(j2,k2)*PI/180)
  
              u(j,k)=((p(j2,k2+1)-p(j2,k2-1))*100)
     +                   /(rhu*ref*120000)
              v(j,k)=((p(j2+1,k2)-p(j2-1,k2))*100)
     +                   /(rhu*ref*120000)
c              u(j,k)=((p(j2,k2-1)-p(j2,k2+1)))
c     +                   /(rhu*ref*6000.)
c              v(j,k)=((p(j2-1,k2)-p(j2+1,k2)))
c     +                   /(rhu*ref*6000.)
              psl(j,k)=p(j2,k2)


 13       continue

            if(iday.ge.10) write(iii,511) iday
            if(iday.lt.10) write(iii,512) 0,iday
            if(mm.ge.10) write(mmo,511) mm
            if(mm.lt.10) write(mmo,512) 0,mm
 511        format(i2)
 512        format(i1,i1)
            
            print *, ys, mmo, iii
c            print *, file3

            write(file3,505) ys,mmo,iii
505        format(3Hecm,a2,a2,a2,6H.ads60)
 
            open(51,file=file3
     +      ,form='unformatted',access='direct',recl=4)
  
            print *, file3

             do 341 k=1,145
              do 342 j=1,145 
                u2(j,k)=u(j,k)
                v2(j,k)=v(j,k)
                psl2(j,k)=psl(j,k)
 342           continue
 341         continue  

            irec=0
            do 41 k=1,145
              do 42 j=1,145 
                irec=irec+1
                write(51,rec=irec) u2(j,k)
                irec=irec+1
                write(51,rec=irec) v2(j,k)
                irec=irec+1
                write(51,rec=irec) psl2(j,k)

 42           continue
 41         continue  
  
            close(51)

  21    continue

c      if (mm.eq.3) then
c        yy=yy+1
c        mm=01
c        goto 2
c      else 
c        mm=mm+1
c        goto 2
c      endif

 2    continue

 800  continue

          close(52)

      stop
      end

c------------------------------------------------------

      subroutine mapxy(xq,yq,alat,alon,slat,e,re,e2,pi)

      real*4 alat,alon
      double precision e,e2,cdr,pi,xq,yq
      
      cdr=57.29577951
      sn=1.0                                   
      xlam=-45.0
      slat=70.0
      re=6378.273
      e2=0.006693883
      pi=3.141592654
      e=sqrt(e2)
      
      
      rho=sqrt(xq**2+yq**2)
      if(rho.gt.0.1)goto 250
      alat=135.0
      if(sn .lt.0.0)alat=-90.0
      alon=90.0
      goto 999
      
  250 cm=cos(slat*pi/180)/sqrt(1.0-e2*(sin(slat*pi/180)**2))
      t=tan((pi/4.0)-(slat/(2.0*cdr)))/((1.0-e*sin(slat*pi/180))
     *  /(1.0+e*sin(slat*pi/180)))**(e/2.0)
      t=rho*t/(re*cm)
      chi=(pi/2.0)-2.0*atan(t)
      alat=chi+((e2/2.0)+(5.0*e2**2.0/24.0)+(e2**3.0/12.0))*sin(2.0*chi)
     *+((7.0*e2**2.0/48.0)+(29.0*e2**3/240.0))*sin(4.0*chi)+
     *(7.0*e2**3.0/120.0)*sin(6.0*chi)
      alat=sn*alat*cdr
      
      alon=xlam+sn*(atan2(xq,-yq))*180/pi
      if(alon.lt.0.0)alon=alon+360.0
      if(alon.gt.0.0)alon=alon-360.0
       
  999 continue
      return
      end
