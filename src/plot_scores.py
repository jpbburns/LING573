'''
    plot distributions for voting scores
'''
from matplotlib import pyplot as plt
import os, sys
import pandas as pd
import numpy as np
import ipdb

if __name__ == '__main__':
    data_dir_name = str(sys.argv[1])
    data_dir_path = '../data/subtask-2/humor_scores/{}'.format(data_dir_name)
    data_files = os.listdir(data_dir_path)

    for data_file in data_files:
        data_path = data_dir_path + '/' + data_file
        df=pd.read_csv(data_path)
        scores=np.array(df['score'].tolist()[::-1])
        bottomhalf, tophalf = np.array_split(scores, 2)
        shaved = scores[0:int((3*scores.shape[0])/4)]

        # plt.plot(scores)
        # plt.title("Posts from {}".format(data_file))
        # plt.xlabel("Rank")
        # plt.ylabel("Upvote count")
        # plt.savefig('../outputs/D4/' + data_file + '_plot.png')
        # plt.clf()       

        # plt.plot(tophalf)
        # plt.title("Top half posts from {}".format(data_file))
        # plt.xlabel("Rank")
        # plt.ylabel("Upvote count")
        # plt.savefig('../outputs/D4/' + data_file + '_plot_tophalf.png')
        # plt.clf()

        # plt.plot(bottomhalf)
        # plt.title("Bottom half posts from {}".format(data_file))
        # plt.xlabel("Rank")
        # plt.ylabel("Upvote count")
        # plt.savefig('../outputs/D4/' + data_file + '_plot_bottomhalf.png')
        # plt.clf()

        plt.plot(shaved)
        plt.title("Shaved Posts from {}".format(data_file))
        plt.xlabel("Rank")
        plt.ylabel("Upvote count")
        plt.savefig('../outputs/D4/' + data_file + '_shaved.png')
        plt.clf()       