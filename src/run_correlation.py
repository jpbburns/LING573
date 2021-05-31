'''
    run_correlation
'''

from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import os
import pandas as pd

if __name__ == '__main__':
    funny_dir = os.getcwd()+'/../data/subtask-2/normalized/funny'
    not_funny_dir = os.getcwd()+'/../data/subtask-2/normalized/not_funny'

    for data_file in os.listdir(funny_dir):
        df=pd.read_csv(funny_dir + '/' + data_file)
        normalized_scores = df['normalized_scores'].tolist()
        humor_scores = df['humor_scores'].tolist()
        sp_corr = spearmanr(normalized_scores, humor_scores)
        with open('../results/D4/funny/' + data_file + '_correlation.txt', 'w') as f:
            f.write("Spearman correlation: {}".format(str(sp_corr)))

        plt.plot(normalized_scores, humor_scores)
        plt.title("Posts from {}".format(data_file))
        plt.xlabel("Normalized Upvote Scores")
        plt.ylabel("Humor Scores")
        plt.savefig('../results/D4/funny/' + data_file + '_correlation_plot.png')
        plt.clf()

    for data_file in os.listdir(not_funny_dir):
        df=pd.read_csv(not_funny_dir + '/' + data_file)
        normalized_scores = df['normalized_scores'].tolist()
        humor_scores = df['humor_scores'].tolist()
        sp_corr = spearmanr(normalized_scores, humor_scores)
        with open('../results/D4/not_funny/' + data_file + '_correlation.txt', 'w') as f:
            f.write("Spearman correlation: {}".format(str(sp_corr)))

        plt.plot(normalized_scores, humor_scores)
        plt.title("Posts from {}".format(data_file))
        plt.xlabel("Normalized Upvote Scores")
        plt.ylabel("Humor Scores")
        plt.savefig('../results/D4/not_funny/' + data_file + '_correlation_plot.png')
        plt.clf()