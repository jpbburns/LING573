'''
	normalize upvote scores
'''
import os
import pandas as pd
import numpy as np
import ipdb

if __name__ == '__main__':
    funny_dir = os.getcwd()+'/../data/subtask-2/humor_scores/funny'
    not_funny_dir = os.getcwd()+'/../data/subtask-2/humor_scores/not_funny'

    funny_output_dir = os.getcwd()+'/../data/subtask-2/normalized/funny'
    not_funny_output_dir = os.getcwd()+'/../data/subtask-2/normalized/not_funny'


    for data_file in os.listdir(funny_dir):
    	df=pd.read_csv(funny_dir + '/' + data_file)
    	df=df[df['idx'] >= 250]

    	high = df['score'].tolist()[0]
    	low = df['score'].tolist()[-1]
    	#print(high)
    	#print(low)

    	normalized_scores = ((np.array(df['score'].tolist()) - low) / ((high-low) / 3))
    	df['normalized_scores'] = normalized_scores

    	#ipdb.set_trace()
    	df.to_csv(funny_output_dir + '/' + data_file)
    	df.to_csv('../outputs/D4/funny/' + data_file)

    for data_file in os.listdir(not_funny_dir):
    	df=pd.read_csv(not_funny_dir + '/' + data_file)
    	df=df[df['idx'] >= 250]

    	high = df['score'].tolist()[0]
    	low = df['score'].tolist()[-1]
    	#print(high)
    	#print(low)

    	normalized_scores = ((np.array(df['score'].tolist()) - low) / ((high-low) / 3))
    	df['normalized_scores'] = normalized_scores
    	df.to_csv(not_funny_output_dir + '/' + data_file)
    	df.to_csv('../outputs/D4/not_funny/' + data_file)

