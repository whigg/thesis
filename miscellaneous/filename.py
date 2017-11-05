import csv
import datetime as dt

date = dt.date(2003, 1, 1)
result = []
for i in range(365*9+366*2+31+28+31+1):
	a = date.strftime("%Y%m%d")
	b = str(date.strftime("%Y%m%d"))
	file = "ecm" + b[2:] + ".ads60"
	result.append(file)
	#print (file)
	date += dt.timedelta(days=1)

# ファイルオープン
f = open('output.csv', 'w')

for item in result:
	f.write("\n%s"%item)

# ファイルクローズ
f.close()