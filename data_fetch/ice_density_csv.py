import csv, glob, os, tarfile
import numpy as np
from tqdm import tqdm
"""
def readfile(fname):
	f = "<f"
	dt = np.dtype([("u",f), ("v",f), ("slp",f)])
	fd = open(fname,"r")
	result = np.fromfile(fd, dtype=dt, count=145*145)
	return result

#unzip tar.gz file to './input_binary'
print ('Extracting tar.gz file ...')
tar = tarfile.open("erainterim-1403_60.tar.gz")
tar.extractall(path='./input_binary')
tar.close()

#read each file in 'input_binary' and save the csv file to './output_csv'
files = [os.path.basename(r) for r in glob.glob('input_binary/*')]
for f in tqdm(files):
	fname = './input_binary/' + f
	result = readfile(fname)
	sname = './output_csv/' + f[0:9] + '.csv'
	np.savetxt(sname, result, delimiter=',')
"""


fname = '../visual/data/GW1AM220120702D_IC0NP.dat'
fp = open(fname,'rb')
ary = np.fromfile(fp, '<h', -1)
fp.close()

ic0 = ary[ary>=0]
print (ic0)
print (len(ic0))


sname = "GW1AM220120702D_IC0NP.csv"
#np.savetxt(sname, result, delimiter=',')
