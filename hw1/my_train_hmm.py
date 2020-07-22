#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ze Xuan Ong
1/21/19
Minor modifications to make it more Pythonic and to streamline piping

David Bamman
2/14/14

Python port of train_hmm.pl:

Noah A. Smith
2/21/08
Code for maximum likelihood estimation of a bigram HMM from
column-formatted training data.

Usage:  train_hmm.py tags-file text-file hmm-file

The training data should consist of one line per sequence, with
states or symbols separated by whitespace and no trailing whitespace.
The initial and final states should not be mentioned; they are
implied.
The output format is the HMM file format as described in viterbi.pl.

"""

import sys
import re
import numpy as np

from collections import defaultdict

# Files

# TAG_FILE = 'ptb.2-21.tgs'
# TOKEN_FILE = 'ptb.2-21.txt'
# OUTPUT_FILE = 'train_task2.hmm'
ratio = 1

TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
# ratio = int(sys.argv[4])/int(sys.argv[5])

# Vocabulary
vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

# Transition and emission probabilities
emissions = {}
emissions_total = defaultdict(lambda: 0)

tri_transitions = {}
tri_transitions_total = {}

bi_transitions = {}
bi_transitions_total = defaultdict(lambda: 0)

#%%
count = 0
with open(TAG_FILE, 'r') as tag_file:
    for tag_line in tag_file:
        count += 1

index_array = np.arange(count)
np.random.shuffle(index_array)
index_array = index_array[:int(count*ratio)]

#%%
with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
    count = 0
    for tag_string, token_string in zip(tag_file, token_file):
        
        if count not in index_array:
            count += 1
            continue
        
        count += 1
        tags = re.split("\s+", tag_string.rstrip())
        tokens = re.split("\s+", token_string.rstrip())
        pairs = zip(tags, tokens)

        prevtag = INIT_STATE
        prevprevtag = prevtag

        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD

            if tag not in emissions:
                emissions[tag] = defaultdict(lambda: 0)
            # trigram
            if prevprevtag not in tri_transitions:
                tri_transitions[prevprevtag] = defaultdict(lambda: 0)
                tri_transitions_total[prevprevtag] = defaultdict(lambda: 0)
            if prevtag not in tri_transitions[prevprevtag]:
                tri_transitions[prevprevtag][prevtag] = defaultdict(lambda: 0)
            # bigram
            if prevtag not in bi_transitions:
                bi_transitions[prevtag] = defaultdict(lambda: 0)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissions_total[tag] += 1
            
            tri_transitions[prevprevtag][prevtag][tag] += 1
            tri_transitions_total[prevprevtag][prevtag] += 1
            
            bi_transitions[prevtag][tag] += 1
            bi_transitions_total[prevtag] += 1
            
            prevprevtag = prevtag
            prevtag = tag

        # don't forget the stop probability for each sentence
        if prevprevtag not in tri_transitions:
            tri_transitions[prevprevtag] = defaultdict(lambda: 0)
            tri_transitions_total[prevprevtag] = defaultdict(lambda: 0)
        if prevtag not in tri_transitions[prevprevtag]:
            tri_transitions[prevprevtag][prevtag] = defaultdict(lambda: 0)
        
        if prevtag not in bi_transitions:
            bi_transitions[prevtag] = defaultdict(lambda: 0)

        bi_transitions[prevtag][FINAL_STATE] += 1
        bi_transitions_total[prevtag] += 1

        tri_transitions[prevprevtag][prevtag][FINAL_STATE] += 1
        tri_transitions_total[prevprevtag][prevtag] += 1


# Write output to output_file
with open(OUTPUT_FILE, "w") as f:
    for prevprevtag in tri_transitions:
        for prevtag in tri_transitions[prevprevtag]:
            for tag in tri_transitions[prevprevtag][prevtag]:
                f.write("tri_trans {} {} {} {}\n"
                        .format(prevprevtag, prevtag, tag, tri_transitions[prevprevtag][prevtag][tag] / tri_transitions_total[prevprevtag][prevtag]))
    
    for prevtag in bi_transitions:
        for tag in bi_transitions[prevtag]:
            f.write("bi_trans {} {} {}\n"
                .format(prevtag, tag, bi_transitions[prevtag][tag] / bi_transitions_total[prevtag]))
    
    for tag in emissions:
        f.write("uni_prob {} {}\n"
                .format(tag, emissions_total[tag]/sum(emissions_total.values())))
        
    for tag in emissions:
        for token in emissions[tag]:
            f.write("emit {} {} {}\n"
                .format(tag, token, emissions[tag][token] / emissions_total[tag]))



