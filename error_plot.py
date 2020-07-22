#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:27:14 2020

@author: soyeon
"""

import numpy as np
import matplotlib.pyplot as plt

ratio_list = []
err_word_list = []
err_sen_list = []

with open('ratio.txt', 'r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        ratio, err_word, err_sen = line.split(' ')
        ratio_list.append(int(ratio)/100)
        err_word_list.append(float(err_word))
        err_sen_list.append(float(err_sen))

plt.plot(ratio_list, err_word_list)
plt.plot(ratio_list, err_sen_list)
plt.ylabel('error')
plt.xlabel('ratio')
plt.legend(['word', 'sentence'], loc='upper right')
plt.show()