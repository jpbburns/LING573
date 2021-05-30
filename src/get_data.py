import pandas as pd
import glob, os

def get_files_as_dataframe():
	path = os.getcwd() + '/../data/subtask-2'
	all_files = glob.glob(path + "/*.csv")

	li = []

	for filename in all_files:
	    df = pd.read_csv(filename, index_col=None, header=0)
	    li.append(df)

	frame = pd.concat(li, axis=0, ignore_index=True)
	return frame