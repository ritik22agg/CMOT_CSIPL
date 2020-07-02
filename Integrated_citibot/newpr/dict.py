import pandas as pd

def get_dict():
	df = pd.read_csv(r'processed_data.csv')
	d = {}
	for i in range(len(df)):
		key = df.iloc[i,1]
		val = df.iloc[i,2]
		if key in d.keys():
			if val not in d[key]:
				d[key].append(val)
		else:
			d[key] = []
			d[key].append(df.iloc[i,2])
	return d				
    