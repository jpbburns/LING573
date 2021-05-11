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

# Feed-forward Neural Network, Two hidden layers
class LinearRegressionHeadOneHidden(nn.Module):
	def __init__(self, num_feature, num_class, drop=0.2):
		super(LinearRegressionHeadOneHidden, self).__init__()
		
		self.drop = drop

		# Hidden Layers
		self.layer_1 = nn.Linear(num_feature, 64)
		self.layer_out = nn.Linear(64, num_class) 
		
		# Activation fxn, Dropout, etc
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=drop)
		self.batchnorm1 = nn.BatchNorm1d(64)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def set_dropout(self, drop):
		self.dropout = nn.Dropout(p=drop)

	# Forward function for forward pass through network
	def forward(self, x):
		x = self.layer_1(x)
		x = self.batchnorm1(x)
		x = self.relu(x)
		
		x = self.layer_out(x)

		# Add scaled sigmoid
		x = (self.sigmoid(x) * 3)
		
		return x

# Feed-forward Neural Network, Two hidden layers
class LinearRegressionHeadTwoHidden(nn.Module):
	def __init__(self, num_feature, num_class, drop=0.2):
		super(LinearRegressionHeadTwoHidden, self).__init__()
		
		self.drop = drop

		# Hidden Layers
		self.layer_1 = nn.Linear(num_feature, 128)
		self.layer_2 = nn.Linear(128, 64)
		self.layer_out = nn.Linear(64, num_class) 
		
		# Activation fxn, Dropout, etc
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=drop)
		self.batchnorm1 = nn.BatchNorm1d(128)
		self.batchnorm2 = nn.BatchNorm1d(64)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def set_dropout(self, drop):
		self.dropout = nn.Dropout(p=drop)

	# Forward function for forward pass through network
	def forward(self, x):
		x = self.layer_1(x)
		x = self.batchnorm1(x)
		x = self.relu(x)
		
		x = self.layer_2(x)
		x = self.batchnorm2(x)
		x = self.relu(x)
		x = self.dropout(x)
		
		x = self.layer_out(x)

		# Add scaled sigmoid
		x = (self.sigmoid(x) * 3)
		
		return x

# Feed-forward Neural Network, Three hidden layers
class LinearRegressionHeadThreeHidden(nn.Module):
	def __init__(self, num_feature, num_class):
		super(LinearRegressionHeadThreeHidden, self).__init__()

		# Hidden Layers
		self.layer_1 = nn.Linear(num_feature, 512)
		self.layer_2 = nn.Linear(512, 128)
		self.layer_3 = nn.Linear(128, 64)
		self.layer_out = nn.Linear(64, num_class) 
		
		# Activation fxn, Dropout, etc
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=0.2)
		self.batchnorm1 = nn.BatchNorm1d(512)
		self.batchnorm2 = nn.BatchNorm1d(128)
		self.batchnorm3 = nn.BatchNorm1d(64)

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def set_dropout(self, drop):
		self.dropout = nn.Dropout(p=drop)

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

		# Add scaled sigmoid
		x = (self.sigmoid(x) * 3)
		
		return x

class FineTune():
	def __init__(self, input_size, output_size, pre_t='Bert_single_regress_2epochs_32bs.pt', use_saved=False):
		print("Initializing Linear Regression Models...")
		self.pre_t = pre_t

		self.model = None
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.train_sentences = []
		self.train_embeddings = None
		self.train_labels = None
		
		self.dev_sentences = []
		self.dev_embeddings = None
		self.dev_labels = None

		self.test_sentences = []
		self.test_embeddings = None
		self.test_labels = None

		if use_saved:
			self.load_state()
		else:
			self.get_state()
			self.save_state()

	def get_state(self):
		print("Loading BERT Instance...")
		sbert = SingleBert()

		# Get train embeddings
		print("Extracting train embeddings from BERT...")
		train_sentences, train_embeddings, train_labels = sbert.get_embeddings(self.pre_t, data_file='train.csv')
		self.train_sentences = train_sentences
		self.train_embeddings = torch.stack(train_embeddings)
		self.train_labels = torch.stack(train_labels)

		# Get dev embeddings
		print("Extracting dev embeddings from BERT...")
		dev_sentences, dev_embeddings, dev_labels = sbert.get_embeddings(self.pre_t, data_file='dev.csv')
		self.dev_sentences = dev_sentences
		self.dev_embeddings = torch.stack(dev_embeddings)
		self.dev_labels = torch.stack(dev_labels)

		# Get test embeddings
		print("Extracting test embeddings from BERT...")
		test_sentences, test_embeddings, test_labels = sbert.get_embeddings(self.pre_t, data_file='test.csv')
		self.test_sentences = test_sentences
		self.test_embeddings = torch.stack(test_embeddings)
		self.test_labels = torch.stack(test_labels)

	def save_state(self):
		# Save train sentences in order
		with open('saved/train_sentences.txt', 'w') as f:
			for i,s in enumerate(self.train_sentences):
				f.write(s+'\n')

		# Save train embeddings in order
		torch.save(self.train_embeddings, 'saved/train_embeddings.pt')

		# Save train labels in order
		torch.save(self.train_labels, 'saved/train_labels.pt')


		# Save dev sentences in order
		with open('saved/dev_sentences.txt', 'w') as f:
			for i,s in enumerate(self.dev_sentences):
				f.write(s+'\n')

		# Save test embeddings in order
		torch.save(self.dev_embeddings, 'saved/dev_embeddings.pt')

		# Save dev labels in order
		torch.save(self.dev_labels, 'saved/dev_labels.pt')


		# Save test sentences in order
		with open('saved/test_sentences.txt', 'w') as f:
			for i,s in enumerate(self.test_sentences):
				f.write(s+'\n')

		# Save test embeddings in order
		torch.save(self.test_embeddings, 'saved/test_embeddings.pt')

		# Save test labels in order
		torch.save(self.test_labels, 'saved/test_labels.pt')


	def load_state(self):
		# Load train sentences in order
		with open('saved/train_sentences.txt', 'r') as f:
			self.train_sentences = f.readlines()

		# Load train embeddings in order
		self.train_embeddings = torch.load('saved/train_embeddings.pt')

		# Load train labels in order
		self.train_labels = torch.load('saved/train_labels.pt')


		# Load dev sentences in order
		with open('saved/dev_sentences.txt', 'r') as f:
			self.dev_sentences = f.readlines()

		# Load dev embeddings in order
		self.dev_embeddings = torch.load('saved/dev_embeddings.pt')

		# Load dev labels in order
		self.dev_labels = torch.load('saved/dev_labels.pt')


		# Load test sentences in order
		with open('saved/test_sentences.txt', 'r') as f:
			self.test_sentences = f.readlines()

		# Load test embeddings in order
		self.test_embeddings = torch.load('saved/test_embeddings.pt')

		# Load test labels in order
		self.test_labels = torch.load('saved/test_labels.pt')

	def balance(self, num):
		arr = np.linspace(0, 3, 16)
		diff, ans = (3, -1)
		for e in arr:
			if abs(e - num) < diff:
				diff = abs(e - num)
				ans = e

		return round(ans, 1)

	# Do not shuffle test data. Maintain order of orig sentence with its embedding
	# and label for output.
	def evaluate(self, eval_set='dev', epochs=8, bs=16, lr=2e-05, eps=1e-08, model_num=1, drop=0.2):
		if eval_set == 'dev':
			sentences = self.dev_sentences
			embeddings = self.dev_embeddings
			labels = self.dev_labels	
		elif eval_set == 'test':
			sentences = self.test_sentences
			embeddings = self.test_embeddings
			labels = self.test_labels
		else:
			raise ValueError("Evaluation set must be either \'dev\' or \'test\'")

		load_data = DataLoader(TensorDataset(embeddings, labels), batch_size=128)
		pred = []

		self.model.eval()
		for step, batch in enumerate(load_data):
			batch_ids, batch_lbls = tuple((t.to(self.device) for t in batch))
			with torch.no_grad():
				outputs = self.model(batch_ids).squeeze(1).tolist()
				pred.extend(outputs)

		pred = [self.balance(x) for x in pred]

		# Print output predictions along with true label and original sentence
		# to unique csv file in outputs
		output_data = pd.DataFrame({'sent':sentences, 'actual':labels,'pred':pred})
		output_name = "output_{}epochs_{}bs_{}lr_{}drop_{}hl.csv".format(epochs, bs, lr, drop, model_num)
		output_data.to_csv('../outputs/D3/{}'.format(output_name), index=False)

		# Calculate RMSE
		rmse = np.sqrt(np.mean((np.array(labels) - np.array(pred)) ** 2))

		# Print RMSE result to D3_scores.out
		result = "Epochs: {}, Batch Size: {}, Learning Rate: {}, Dropout Probability: {}, Num Hidden Layers: {}\t RMSE: {}".format(epochs, bs, lr, drop, model_num, rmse)
		print(result)
		with open("../results/D3_scores.out", "a") as score_file:
			score_file.write(result+"\n")

		return rmse

	def tune(self, epochs=8, bs=16, lr=2e-05, eps=1e-08, model_num=1, drop=0.2):
		if model_num==1:
			self.model = LinearRegressionHeadOneHidden(input_size, output_size)
		if model_num==2:
			self.model = LinearRegressionHeadTwoHidden(input_size, output_size)
		if model_num==3:
			self.model = LinearRegressionHeadThreeHidden(input_size, output_size)

		# Set up data and self.	
		#ipdb.set_trace()
		load_data = DataLoader(TensorDataset(self.train_embeddings, self.train_labels), shuffle=True, batch_size=bs)
		adam_op = AdamW(self.model.parameters(), lr=lr, eps=eps)
		steps = len(load_data) * epochs
		scheduler = get_linear_schedule_with_warmup(adam_op, num_warmup_steps=0, num_training_steps=steps)
		criterion = torch.nn.MSELoss()
		loss_hld = []
		self.model.set_dropout(drop)
		self.model.zero_grad()

		for i in range(epochs):
			print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
			t0 = time.time()
			loss_tot = 0
			self.model.train()
			for s, b in enumerate(load_data):

				# Track time
				if s % 100 == 0 and not s == 0:
					elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
					#print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(s, len(load_data), elapsed))

				# Run forward pass on batch
				batch_embeddings, batch_lbls = tuple((t.to(self.device) for t in b))
				outputs = self.model(batch_embeddings).squeeze(1)
				loss = criterion(outputs, batch_lbls)
				loss_tot += loss.item()

				# Run backward pass
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				adam_op.step()
				scheduler.step()

				# Not sure if should be doing this
				self.model.zero_grad()

			training_loss = loss_tot / len(load_data)
			loss_hld.append(training_loss)

			# Print training loss for epoch
			print('Training loss: {0:.2f}'.format(training_loss))
			str(datetime.timedelta(seconds=int(round(time.time() - t0))))
			print('Training epcoh took: {:}'.format(str(datetime.timedelta(seconds=int(round(time.time() - t0))))))

		print('Training Finished.')
		torch.cuda.empty_cache()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--extract_pretrained", default=False, action='store_true', help="Re-extract embeddings from BERT for training/testing; will be stored in the 'saved' folder.")
	args = parser.parse_args()

	with open("../results/D3_scores.out", "w") as score_file:
		score_file.write("Results on hyperparameter tuning\n\n")


	best_pre_t = 'Bert_single_regress_2epochs_32bs.pt'
	use_saved=True
	if args.extract_pretrained:
		use_saved=False
	input_size = 768 # Fixed pre-trained embedding size for BERT
	output_size = 1  # Single score output
	epochs = [2, 4, 8]
	learning_rates = [2e-04, 2e-05, 2e-06]
	dropouts = [0.2, 0.3, 0.4]
	models = [1, 2, 3]
	batch_sizes = [4, 8, 16]

	lin_reg_heads = FineTune(input_size, output_size, pre_t='Bert_single_regress_2epochs_32bs.pt', use_saved=use_saved)

	best_rmse = math.inf
	for m in models:
		for e in epochs:
			for lr in learning_rates:
				for d in dropouts:
					for bs in batch_sizes:
						lin_reg_heads.tune(epochs=e, bs=bs, lr=lr, eps=1e-08, model_num=m, drop=d)
						rmse = lin_reg_heads.evaluate(eval_set='dev', epochs=e, bs=bs, lr=lr, eps=1e-08, model_num=m, drop=d)

						if rmse < best_rmse:
							best_rmse = rmse
							best_out = "Best Model -- Epochs: {}, Batch Size: {}, Learning Rate: {}, Dropout Probability: {}, Num Hidden Layers: {}\t RMSE: {}".format(e, bs, lr, d, m, best_rmse)

	with open("../results/D3_scores.out", "a") as score_file:
		score_file.write("\n\n"+best_out+"\n")
	print(best_out)
