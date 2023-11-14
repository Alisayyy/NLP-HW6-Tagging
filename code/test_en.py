#!/usr/bin/env python3
# This file illustrates how you might experiment with the HMM interface at the prompt.
# You can also run it directly.

import logging
import math
import os
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus
from eval import eval_tagging, model_cross_entropy, viterbi_error_rate
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
import torch

# Set up logging
log = logging.getLogger("test_en")       # For usage, see findsim.py in earlier assignment.
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Switch working directory to the directory where the data live.  You may want to edit this line.
os.chdir("../data")
entrain = TaggedCorpus(Path("ensup"), Path("enraw"))                               # all training
ensup =   TaggedCorpus(Path("ensup"), tagset=entrain.tagset, vocab=entrain.vocab)  # supervised training
endev =   TaggedCorpus(Path("endev"), tagset=entrain.tagset, vocab=entrain.vocab)  # evaluation
log.info(f"Tagset: f{list(entrain.tagset)}")
known_vocab = TaggedCorpus(Path("ensup")).vocab    # words seen with supervised tags; used in evaluation

# Initialize an HMM
lexicon = build_lexicon(entrain, embeddings_file=Path('words-50.txt'))  # works better with more attributes!
hmm = HiddenMarkovModel(entrain.tagset, entrain.vocab, lexicon)

# Let's initialize with supervised training to approximately maximize the
# regularized log-likelihood.  If you want to speed this up, you can increase
# the tolerance of training (using the `tolerance` argument), since we don't 
# really have to train to convergence.
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=ensup)
hmm.train(corpus=ensup, loss=loss_sup, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=1) 

# Now let's throw in the unsupervised training data as well, and continue
# training to try to improve accuracy on held-out development data.
# We'll stop when accuracy is getting worse, so we can get away without regularization,
# but it would be better to search for the best `reg` and other hyperparameters in this call.
loss_dev = lambda model: viterbi_error_rate(model, eval_corpus=endev, known_vocab=known_vocab)
hmm.train(corpus=entrain, loss=loss_dev, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=0)

# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
for m, sentence in enumerate(endev):
    if m >= 10: break
    viterbi = hmm.viterbi_tagging(sentence.desupervise(), endev)
    counts = eval_tagging(predicted=viterbi, gold=sentence, 
                          known_vocab=known_vocab)
    num = counts['NUM', 'ALL']
    denom = counts['DENOM', 'ALL']
    
    log.info(f"Gold:    {sentence}")
    log.info(f"Viterbi: {viterbi}")
    log.info(f"Loss:    {denom - num}/{denom}")
    log.info(f"Prob:    {math.exp(hmm.log_prob(sentence, endev))}")
