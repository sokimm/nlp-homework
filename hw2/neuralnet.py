#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:16:56 2020

@author: soyeon
"""
import re
import string
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F



class nn_model(nn.Module):
    def __init__(self):
        super(nn_model ,self).__init__()
        self.fc1 = nn.Linear(1000, 200)
        self.fc2 = nn.Linear(200, 2)
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x
        
    def predict(self,x):
        pred = F.softmax(self.forward(x))
        ans = []
        for p in pred:
            if p[0]>p[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

if __name__ == "__main__":
    np.random.seed(0)

    train_x_path = 'dev_text.txt'
    train_y_path = 'dev_label.txt'
    test_x_path = 'heldout_text.txt'
    test_y_path = 'heldout_pred_nn.txt'
     
    text_list = []
    label_list = []
    stopword_list = []
    word_dict = dict()
    
    with open(train_y_path) as infile:
        for line in infile.read().splitlines():
            if line == 'pos':
                label_list.append(1)
            elif line == 'neg':
                label_list.append(0)

    # stopword list
    with open('stopword.list') as infile:
        for line in infile.read().splitlines():
            stopword_list.append(line)
    stopword_set = set(stopword_list)
    
    # pattern for remove punctuation
    regex_punc = re.compile('[%s]' % re.escape(string.punctuation.replace('\'', '')))

    start = datetime.datetime.now()
    with open(train_x_path) as infile:
        for line in infile:
            text = regex_punc.sub(' ', line.lower())
            text = text.replace('\'', '').replace('\n', ' ')
            text_list.append(text)
            # check_df = []
            for word in text.split():
                    # remove all the tokens that contain numbers
                    if word.isdigit():
                        continue
                    # remove the stop words
                    if word in stopword_set:
                        continue
                    # ctf dict
                    if word in word_dict.keys():
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
    print(datetime.datetime.now()-start)

    sorted_dict = sorted(word_dict.items(), reverse=True, key=lambda item: item[1])
#%%    
    top_1000_tokens = [sorted_dict[i][0] for i in range(1000)]
    top_1000_tokens_set = set(top_1000_tokens)

    ctf_rows = []
    ctf_cols = []
    ctf_data = []
    
    start = datetime.datetime.now()
    for i in range(len(text_list)):
        text = text_list[i]
        for word in text.split():
            if word in top_1000_tokens_set:
                ctf_rows.append(i)
                ctf_cols.append(top_1000_tokens.index(word))
                ctf_data.append(1)
    print(datetime.datetime.now()-start)

    ctf_csr = csr_matrix((ctf_data, (ctf_rows, ctf_cols)), dtype=np.int)
#%%
    X = np.array(ctf_csr.toarray())
    y = np.array(label_list)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.LongTensor)
    
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)

#%%
    test_ctf_rows = []
    test_ctf_cols = []
    test_ctf_data = []
    
    with open(test_x_path) as infile:
        i = 0
        for line in infile:
            text = regex_punc.sub(' ', line.lower())
            text = text.replace('\'', '').replace('\n', ' ')
            for word in text.split():
                if word in top_1000_tokens_set:
                    test_ctf_rows.append(i)
                    test_ctf_cols.append(top_1000_tokens.index(word))
                    test_ctf_data.append(1)
            i += 1
            
    test_ctf = csr_matrix((test_ctf_data, (test_ctf_rows, test_ctf_cols)), dtype=np.int)
    X_test = np.array(test_ctf.toarray())
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
   
#%%
    """train the model"""
    
    model = nn_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 10
    losses = []
    val_losses = []
    temp1 = 0
    temp2 = 0
    
    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_val_pred = model.forward(X_val)
        val_loss = criterion(y_val_pred, y_val)
        val_losses.append(val_loss.item())
        if val_loss.item() > temp1 and temp1 > temp2 and i >= 5:
            break
        else:
            temp2 = temp1
            temp1 = val_loss.item()
            # optimizer.zero_grad()
            # val_loss.backward()
            # optimizer.step()
        print('iter: %d' % i)
        
    print(accuracy_score(model.predict(X), y))
    plt.plot(losses)
    plt.plot(val_losses)
    
  #%%
    y_test = model.predict(X_test).detach().numpy()
  
    with open(test_y_path, 'w') as infile:
        for label in y_test:
            if label == 1:
                infile.write('pos\n')
            elif label == 0:
                infile.write('neg\n')