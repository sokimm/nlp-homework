#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:03:37 2020

@author: soyeon
"""
import sys
import re
import string
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    
    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    test_y_path = sys.argv[4]
    
    # train_x_path = 'dev_text.txt'
    # train_y_path = 'dev_label.txt'
    # test_x_path = 'heldout_text.txt'
    # test_y_path = 'heldout_pred_nb.txt'
    
    text_list = []
    label_list = []
    stopword_list = []
    word_dict = dict()
    # word_dict_df = dict()

    with open(train_y_path) as infile:
        for line in infile.read().splitlines():
            if line == 'pos':
                label_list.append(1)
            elif line == 'neg':
                label_list.append(-1)

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
                    # df dict
                    # if word not in check_df:
                    #     check_df.append(word)
                    #     if word in word_dict_df.keys():
                    #         word_dict_df[word] += 1
                    #     else:
                    #         word_dict_df[word] = 1
    print(datetime.datetime.now()-start)

    sorted_dict = sorted(word_dict.items(), reverse=True, key=lambda item: item[1])
    # sorted_dict_df = sorted(word_dict_df.items(), reverse=True, key=lambda item: item[1])
#%%    
    top_1000_tokens = [sorted_dict[i][0] for i in range(1000)]
    # top_1000_tokens_df = [sorted_dict_df[i][0] for i in range(1000)]
    top_1000_tokens_set = set(top_1000_tokens)
    # top_1000_tokens_df_set = set(top_1000_tokens_df)

    ctf_rows = []
    ctf_cols = []
    ctf_data = []
    # df_rows = []
    # df_cols = []
    # df_data = []
    
    start = datetime.datetime.now()
    for i in range(len(text_list)):
        text = text_list[i]
        for word in text.split():
            if word in top_1000_tokens_set:
                ctf_rows.append(i)
                ctf_cols.append(top_1000_tokens.index(word))
                ctf_data.append(1)
            # if word in top_1000_tokens_df_set:
            #     df_rows.append(i)
            #     df_cols.append(top_1000_tokens_df.index(word))
            #     df_data.append(1)
    print(datetime.datetime.now()-start)

    ctf_csr = csr_matrix((ctf_data, (ctf_rows, ctf_cols)), dtype=np.int)
    # df_csr = csr_matrix((df_data, (df_rows, df_cols)), dtype=np.int)
#%%
    X = np.array(ctf_csr.toarray())
    y = np.array(label_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X)
    print("Number of mislabeled points out of a total %d points : %d" \
          % (X.shape[0], (y != y_pred).sum()))
    # y_pred = gnb.fit(X_train, y_train).predict(X_test)
    # print("Number of mislabeled points out of a total %d points : %d" \
    #       % (X_test.shape[0], (y_test != y_pred).sum()))
          
#%%
    test_ctf_rows = []
    test_ctf_cols = []
    test_ctf_data = []
    # test_df_rows = []
    # test_df_cols = []
    # test_df_data = []
    
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
                # if word in top_1000_tokens_df_set:
                #     test_df_rows.append(i)
                #     test_df_cols.append(top_1000_tokens_df.index(word))
                #     test_df_data.append(1)
            i += 1
            
    test_ctf = csr_matrix((test_ctf_data, (test_ctf_rows, test_ctf_cols)), dtype=np.int)
    # test_df = csr_matrix((test_df_data, (test_df_rows, test_df_cols)), dtype=np.int)    
#%%
    X_test = np.array(test_ctf.toarray())
    y_pred = gnb.fit(X, y).predict(X_test)
#%%
    with open(test_y_path, 'w') as infile:
        for label in y_pred:
            if label == 1:
                infile.write('pos\n')
            elif label == -1:
                infile.write('neg\n')
    
        
