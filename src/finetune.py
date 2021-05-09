import json
import pandas as pd
import numpy as np
import time
import datetime
import pickle
import random
import copy
import torch
import zipfile
import re
import string
import math
import os
import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from main import SingleBert
import ipdb

# Feed-forward Neural Network
class LinearRegressionHead(nn.Module):
	def __init__(self, num_feature, num_class):
		super(MulticlassClassification, self).__init__()
		
		# Hidden Layers
		self.layer_1 = nn.Linear(num_feature, 512)
		self.layer_2 = nn.Linear(512, 128)
		self.layer_3 = nn.Linear(128, 64)
		self.layer_out = nn.Linear(64, num_class) 
		
		# Activation fxn, Dropout, etc
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.2)
		self.batchnorm1 = nn.BatchNorm1d(512)
		self.batchnorm2 = nn.BatchNorm1d(128)
		self.batchnorm3 = nn.BatchNorm1d(64)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Forward function for forward pass through network
	def forward(self, x):
		x = self.layer_1(x)
		x = self.batchnorm1(x)
		x = self.relu(x)
		
		x = self.layer_2(x)
		x = self.batchnorm2(x)
		x = self.relu(x)
		x = self.dropout(x)
		
		x = self.layer_3(x)
		x = self.batchnorm3(x)
		x = self.relu(x)
		x = self.dropout(x)
		
		x = self.layer_out(x)
		
		return x

class FineTune():
	def __init__(self, input_size, output_size):
		self.model = LinearRegressionHead(input_size, output_size)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def tune(self, train_data, labels, epochs=8, bs=16):
		LR = 2e-05
		EPS = 1e-08

		adam_op = AdamW(self.parameters(), lr=LR, eps=EPS)
		steps = len(load_data) * epochs
		scheduler = get_linear_schedule_with_warmup(adam_op, num_warmup_steps=0, num_training_steps=steps)

		loss_hld = []
		self.model.zero_grad()
		for i in range(epochs):
			print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
			t0 = time.time()
			loss_tot = 0
			self.model.train()

			for s, b in enumerate(train_data):
				if s % 100 == 0 and not s == 0:
					elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))

					print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(s, len(load_data), elapsed))

				outputs = self.model(b)
				loss = outputs[0]
				print("Epoch loss: {}".format(loss.item()))

				loss_tot += loss.item()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				adam_op.step()
				scheduler.step()
				self.model.zero_grad()

			training_loss = loss_tot / len(load_data)
			loss_hld.append(training_loss)

			print('Training loss: {0:.2f}'.format(training_loss))
			str(datetime.timedelta(seconds=int(round(time.time() - t0))))
			print('Training epcoh took: {:}'.format(str(datetime.timedelta(seconds=int(round(time.time() - t0))))))

		print('Training Finished.')
		torch.cuda.empty_cache()

		return training_loss


if __name__ == '__main__':
	pre_t = os.listdir(os.getcwd()+'/pretrained/single')
	#ipdb.set_trace()
	s_bert = SingleBert(max_len=96)
	for p in pre_t:	
		train_embeddings, train_labels = s_bert.get_embeddings(pretrained=p)
		num_train_instances = train_embeddings.shape[0]
		input_size = train_X.shape[1]
		output_size = 3
		ipdb.set_trace()
		model = FineTune(input_size, output_size)
		training_loss = model.tune(train_embeddings, train_labels)

