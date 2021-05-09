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
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import ipdb

class SingleBert:

    def __init__(self, model_name='bert-base-cased', max_len=256):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1).to(self.device)
        self.max_len = max_len

    def get_training_data(self, bs=16):
        ts, tl = DataProcessor('../data/subtask-1/train.csv').read_data()
        vs, vl = DataProcessor('../data/subtask-1/dev.csv').read_data()        
        s, l = ts + vs, tl + vl
        t_ids, t_masks, t_types = (torch.tensor(x) for x in get_ids(s, self.tokenizer, max_len=self.max_len))
        load_data = DataLoader(TensorDataset(t_ids, t_masks, t_types, torch.tensor(l)), shuffle=True, batch_size=bs)
        return load_data

    def tune(self, epochs=1, bs=16):

        load_data = self.get_training_data(bs=bs)
        adam_op = AdamW(self.model.parameters(), lr=2e-05, eps=1e-08)
        steps = len(load_data) * epochs
        scheduler = get_linear_schedule_with_warmup(adam_op, num_warmup_steps=0, num_training_steps=steps)

        loss_hld = []
        self.model.zero_grad()
        for i in range(epochs):
            print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
            t0 = time.time()
            loss_tot = 0
            self.model.train()

            for s, b in enumerate(load_data):
                if s % 100 == 0 and not s == 0:
                    elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(s, len(load_data), elapsed))

                batch_ids, batch_mask, batch_typs, batch_lbls = tuple(
                    (t.to(self.device) for t in b))
                outputs = self.model(input_ids=batch_ids, token_type_ids=batch_typs,
                                attention_mask=batch_mask, labels=batch_lbls)
                loss = outputs[0]
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

        fname = '/pretrained/{2}/Bert_{2}_regress_{0}epochs_{1}bs.pt'.format(epochs, bs, 'single')
        torch.save(self.model.state_dict(), os.getcwd() + fname)

        print(fname + ' saved.')
        torch.cuda.empty_cache()
        return self

    def predict(self, pretrained=None, bs=128):
        test_s, test_l = DataProcessor('../data/subtask-1/test.csv').read_data()
        ids, masks, types = (torch.tensor(x) for x in get_ids(test_s, self.tokenizer, max_len=self.max_len))
        dataloader = DataLoader(TensorDataset(ids, masks, types), batch_size=bs)
        
        if pretrained:
            self.model.load_state_dict(torch.load(
                    '{0}/pretrained/{1}/{2}'.format(os.getcwd(), 'single', pretrained)))
        self.model.eval()

        pred = []
        for step, batch in enumerate(dataloader):
            batch_ids, batch_mask, batch_types = tuple((t.to(self.device) for t in batch))
            with torch.no_grad():
                outputs = self.model(input_ids=batch_ids, token_type_ids=batch_types, attention_mask=batch_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            
            pred.extend(logits)
        
        return pred

    def get_embeddings(self, pretrained=None, bs=16):
        pre_t = '{0}/pretrained/{1}/{2}'.format(os.getcwd(), 'single', pretrained)
        pre_t_model = torch.load(pre_t, map_location=self.device)
        self.model.load_state_dict(pre_t_model)
        self.model.eval()
        load_data = self.get_training_data()

        embeddings = []
        labels = []
        for step, batch in enumerate(load_data):
            batch_ids, batch_mask, batch_types, batch_lbls = tuple((t.to(self.device) for t in batch))
            with torch.no_grad():
                outputs = self.model(input_ids=batch_ids, token_type_ids=batch_types, attention_mask=batch_mask)

            ipdb.set_trace()
            hidden = outputs[1]
            embeddings.extend(hidden)
            labels.extend(batch_lbls)

        return embeddings, labels

    def balance(self, num):
        arr = np.linspace(0, 3, 16)
        diff, ans = (3, -1)
        for e in arr:
            if abs(e - num) < diff:
                diff = abs(e - num)
                ans = e

        return round(ans, 1)

    def evaluate(self, pretrained=None,  bs=128):
        pred = [self.balance(x) for x in self.predict(pretrained=pretrained, bs=bs)]
        w, test_l = DataProcessor('../data/subtask-1/test.csv').read_data()

        output = pd.DataFrame({'sent':w, 'actual':test_l ,'pred':pred})
        output.to_csv('../outputs/D2/output.csv', index=False)

        out = np.sqrt(np.mean((np.array(test_l) - np.array(pred)) ** 2))
        print('{} model RMSE = {:.5f}'.format(pretrained[:-3] if pretrained else 'this model', out))
        
        return round(out, 5)


class DataProcessor:

    def __init__(self, path):
        self.path = path

    def feat_eng(self, o, e):
        so = [] 
        se = []
        for i in range(len(o)):
            so.append(re.sub('<(.+)/>', '\\g<1>', o[i]))
            se.append(re.sub('<.+/>', e[i], o[i]))

        return se

    def read_data(self):
        df = pd.read_csv(self.path)
        sents = self.feat_eng(df['original'], df['edit'])
        try:
            labels = df['meanGrade'].tolist()
            return (sents, labels)
        except KeyError:
            return sents

def get_ids(sents, tokenizer, max_len=300):
    ids = []
    masks = []
    typs = [] 
    for e in sents:
        t_e = tokenizer.encode_plus(text=e, max_length=max_len, add_special_tokens=True, pad_to_max_length='right')
        ids.append(t_e['input_ids'])
        masks.append(t_e['attention_mask'])
        typs.append(t_e['token_type_ids'])

    return (ids, masks, typs)

def run_training():
    for i in range(1, 5):
        for j in [32, 16, 8, 4]:
            SingleBert(model_name='bert-base-cased', max_len=96).tune(epochs=i, bs=j)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        run_training()

    pre_t = os.listdir(os.getcwd()+'/pretrained/single')
    s_bert = SingleBert(max_len=96)
    out = [s_bert.evaluate(pretrained=p) for p in pre_t]
    del s_bert
    hld = {'single_regress': out}
    out = {'A': hld}

    train = pd.read_csv('../data/subtask-1/train.csv')
    test = pd.read_csv('../data/subtask-1/test.csv')
    base_out = np.sqrt(np.mean((np.array(test.meanGrade) - np.array(np.mean(train.meanGrade))) ** 2))
    base_out = round(base_out, 5)

    d2_score_file = open('../results/D2_scores.out', 'w')

    print('Result:')
    d2_score_file.write('Result: \n')
    print('Baseline: {}'.format(base_out))
    d2_score_file.write('Baseline: {}\n'.format(base_out))

    final_result = out['A']['single_regress']
    hdr = ['epochs', 'batch=32', 'batch=16', 'batch=8', 'batch=4']
    col = ['1 epoch','2 epochs', '3 epochs', '4 epochs']
    row = "{:>15}" * len(hdr)
    print('Task RMSE results:')
    d2_score_file.write('Task RMSE results:\n')
    print(row.format(*hdr))
    d2_score_file.write(row.format(*hdr))
    d2_score_file.write('\n')
    cells = len(col)
    for x in range(cells):
        print(row.format(col[x], *[final_result[y] for y in range(x*cells, (x+1)*cells)]))
        d2_score_file.write(row.format(col[x]+'\n', *[final_result[y] for y in range(x*cells, (x+1)*cells)]))

    d2_score_file.close()
