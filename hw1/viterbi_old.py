#!/usr/bin/python

# Ze Xuan Ong
# 03/24/2019
# Quick fixes to older version to make compatible with current version of assignment

# Jocelyn Huang
# 10/27/2017
# Viterbi in Python, ported pretty directly from Noah A. Smith's viterbi.pl.
# Not responsible for comment typos due to doing this late at night.

# Usage: ./viterbi.py my.hmm ptb.22.txt > my.out

# The following is an excerpt from his comments (go look at the original 
# if you want more detail):

# Runs the Viterbi algorithm (no tricks other than logmath!), given an
# HMM, on sentences, and outputs the best state path.

import collections
import math
import sys

hmmfile = sys.argv[1]
txt = sys.argv[2]
output = sys.argv[3]

init_state = "init"
final_state = "final"
oov_symbol = "OOV"

# A -> 2-layer hash for transition probabilities
A = collections.defaultdict(dict)
# B -> Emission probabilities
B = collections.defaultdict(dict)

States = set()
Voc = set()

# Read in the HMM and store probs as log probs
with open(hmmfile, 'r') as f:
    for line in f:
        line = line.split()
        if line[0] == 'trans':
            # Read in states qq -> q, and transition prob
            qq,q,prob = line[1:4]
            # Add transition log prob and states seen
            A[qq][q] = math.log(float(prob))
            States.add(qq)
            States.add(q)
        elif line[0] == 'emit':
            # Read in state q -> word w, and emission prob
            q,w,prob = line[1:4]
            # Add emission log prob and state/vocab seen
            B[q][w] = math.log(float(prob))
            States.add(q)
            Voc.add(w)

# Read sentences line by line
output_lines = []
with open(txt, 'r') as f:
    for line in f:
        # Get list of words in the line
        # Note we add the '' to the front for 1-indexing, which makes the
        #   indexing for Viterbi a bit easier to stomach
        w = [''] + line.split()

        # Set up Viterbi for this sentence
        V = collections.defaultdict(dict)
        V[0][init_state] = 0.0 # Base case of recursive equations
        Backtrace = collections.defaultdict(dict)

        for i in range(1, len(w)):
            # If word not in vocab, rename with OOV
            if w[i] not in Voc:
                w[i] = oov_symbol

            # Iterate through possible current states
            for q in States:
                # Iterate through possible prev states
                for qq in States:
                    try: # Only consider "non-zeros"
                        v = V[i-1][qq] + A[qq][q] + B[q][w[i]]
                        # If better previous state found, take note!
                        try:
                            if v > V[i][q]:
                                V[i][q] = v # Viterbi probability
                                Backtrace[i][q] = qq # best prev state
                        except(KeyError):
                            # We get here if V[i][q] hasn't been set yet
                            V[i][q] = v
                            Backtrace[i][q] = qq
                    except(KeyError):
                        pass

        # Handles last of the Viterbi equations, that bring in the final state
        foundgoal = False
        goal = 0
        for qq in States: # For each possible state for the last word
            try:
                v = V[len(w)-1][qq] + A[qq][final_state]
                if not foundgoal or v > goal:
                    # Found a better path; remember it
                    goal = v
                    foundgoal = True
                    q = qq
            except(KeyError):
                pass
        
        # Backtracking step
        if foundgoal:
            # Backwards list of states, because appending is O(1), prepending 
            #   via insert is O(n), and I can't be bothered to figure out 
            #   something else because it's 1:00am
            t = []
            for i in range(len(w)-1, 0, -1):
                t.append(q)
                q = Backtrace[i][q]
            output_lines.append(' '.join(t[::-1]) + "\n")
        # We get a newline as consolation prize if we couldn't Viterbi it out
        else:
            output_lines.append("\n")

with open(output, "w") as f:
    f.writelines(output_lines)

