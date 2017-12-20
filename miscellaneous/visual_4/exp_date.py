from datetime import datetime, date, timezone, timedelta

def cvt_date(dt):
	# "2013-01-01" -> "20130101"
	return str(dt)[:10].replace('-', '')

def get_date_ax(start, end):
	start_date = [start//10000, (start%10000)//100, (start%10000)%100]
	end_date = [end//10000, (end%10000)//100, (end%10000)%100]
	d1 = date(start_date[0], start_date[1], start_date[2])
	d2 = date(end_date[0], end_date[1], end_date[2])
	L = (d2-d1).days+1
	dt = d1

	date_ax = []
	date_ax_str = []
	for i in range(L):
		date_ax.append(dt)
		date_ax_str.append(cvt_date(dt))
		dt = dt + timedelta(days=1)

	return date_ax, date_ax_str



start = 20170101
end = 20170131
date_ax, date_ax_str = get_date_ax(start, end)
print(date_ax)
print(date_ax_str)

day = "20170103"

date_ax_str.remove(day)
aa = day[:4]+"-"+day[4:6]+"-"+day[6:]
print(aa)
print(type(aa))

bb = date(int(day[:4]), int(day[4:6]), int(day[6:]))
print(bb)
date_ax.remove(bb)
skipping_date_str = []
skipping_date_str.append(day)

print("\n")
print(date_ax_str)
print(date_ax)
print(skipping_date_str)









