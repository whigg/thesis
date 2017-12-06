import numpy as np
import pandas as pd
import glob
import re

file_path = sorted(glob.glob("../getA" + '/*/*/*.dat'))

pattern = r"[0-9]-"
to_tab = r"[\s]+"
N = len(file_path)
save_data = []
for i, filename in enumerate(file_path):
	save_name = filename.replace(".dat", "") + "_fin.csv"
	data = open(filename)
	print (("{}/{}:	{} -> {}").format(i+1, N, filename, save_name))
	#print (save_name)
	for j, line in enumerate(data):
		tmp = re.findall(pattern, line)
		#print (tmp)
		if tmp != []:
			#print ("\t{}".format(j))
			iterator = re.finditer(pattern, line)
			for match in iterator:
				#print ("\t{}".format(match.start()))
				index = match.start()
				tmp_data = line
				line = tmp_data[:index] + ' ' + tmp_data[index+1:]
				#print ("\t{}".format(line))
		line = re.sub(to_tab, "\t", line)
		#print (j, line)
		line = line.split("\t")
		line = list(filter(lambda s:s != "", line))
		line = list(map(float, line))
		save_data.append(line)
		#print ("\t{}".format(save_data))
	save_data = np.array(save_data)
	#print (type(pd.DataFrame(save_data)))
	pd.DataFrame(save_data).to_csv(save_name)
	save_data = []
