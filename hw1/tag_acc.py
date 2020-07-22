"""
Tagging Accuracy Checker
Ze Xuan Ong
21 Jan 2019

Noah A. Smith
2/21/08

Adapted from Noah A. Smith's perl script from 2008, with
a significant bit more error handling. Benchmarked against
Noah's tag_acc.pl, produces identical results for this
assignment

Calculates and prints error rate by word and sentence

Usage: python tag_acc.py gold-standard-tags hypothesized-tags

Uses a word level hamming distance measure.
Produces catastrophically bad results if for some reason the
sentences have a different number of tags, or if
some lines are missing. This is intended.

"""

import sys
import re

from itertools import zip_longest

# Get files
GOLD_TAGS = sys.argv[1]
MY_TAGS = sys.argv[2]
ratio = int(sys.argv[3])

# Stats
num_sentences = 0
num_sentence_errors = 0
num_tokens = 0
num_token_errors = 0

# Iterate over both files
gold_tags = []
with open(GOLD_TAGS, "r") as gold_tags, open(MY_TAGS, "r") as my_tags:

    # zip_longest allows us to iterate over the length of the longer list
    for (gold_tag_line, my_tag_line) in zip_longest(gold_tags, my_tags):

        # Terminate loop if more lines in my_tags than gold_tags
        if not gold_tag_line:
            break

        # If missing line, add entire missing line to error num stats
        num_sentences += 1
        if not my_tag_line:
            num_sentence_errors += 1
            gold_tag = re.split("\s+", gold_tags.rstrip())            
            num_tokens += len(gold_tag)
            num_token_errors += len(gold_tag)
            continue

        # Otherwise, compare both lines token by token
        sentence_errors = 0
        for (gold_tag, my_tag) in zip_longest(re.split("\s+", gold_tag_line.rstrip()), re.split("\s+", my_tag_line.rstrip())):

            # Terminate line if my_tag_line longer than gold_tag_line
            if not gold_tag:
                break

            num_tokens += 1
            if gold_tag != my_tag:
                num_token_errors += 1
                sentence_errors += 1

        if sentence_errors > 0:
            num_sentence_errors += 1

err_word = num_token_errors / num_tokens
err_sen = num_sentence_errors / num_sentences

# Print stats
print("Error rate by word: {} ({} errors out of {})".format(err_word, num_token_errors, num_tokens))
print("Error rate by sentence: {} ({} errors out of {})".format(err_sen, num_sentence_errors, num_sentences))

#%%
if ratio == 1:
    with open('ratio.txt', 'w') as f:
        f.write("{} {} {}\n".format(ratio, err_word, err_sen))
else:
    with open('ratio.txt', 'a') as f:
        f.write("{} {} {}\n".format(ratio, err_word, err_sen))