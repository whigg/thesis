      program str
      
      character(len=13) fname
      character(len=15) data

      data = 'ecm030101.ads60'
      fname = data(1:10) // 'csv'
      print *, fname

      end program str