import csv
import glob, os

files = [os.path.basename(r) for r in glob.glob('erainterim-1403_60/*')]
print (len(files))

f = open('output1.csv', 'w')
for item in files:
	f.write("\n%s"%item)
f.close()
