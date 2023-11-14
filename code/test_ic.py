#!/usr/bin/env python3
# This file illustrates how you might experiment with the HMM interface at the prompt.
# You can also run it directly.

import logging, math, os
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus
from eval import model_cross_entropy, write_tagging
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
import torch

# Set up logging
log = logging.getLogger("test_ic")       # For usage, see findsim.py in earlier assignment.
logging.basicConfig(level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Switch working directory to the directory where the data live.  You may want to edit this line.
os.chdir("../data")

# Make an HMM with randomly initialized parameters.
icsup = TaggedCorpus(Path("icsup"), add_oov=False)
log.info(f"Ice cream vocabulary: {list(icsup.vocab)}")
log.info(f"Ice cream tagset: {list(icsup.tagset)}")
lexicon = build_lexicon(icsup, one_hot=True)   # one-hot lexicon: separate parameters for each word
hmm = HiddenMarkovModel(icsup.tagset, icsup.vocab, lexicon)

log.info("*** Current A, B matrices (computed by softmax from small random parameters)")
hmm.updateAB()   # compute the matrices from the initial parameters (this would normally happen during training).
                 # An alternative is to set them directly to some spreadsheet values you'd like to try.
hmm.printAB()

# While training on ice cream, we will just evaluate the cross-entropy
# on the training data itself (icsup), since we are interested in watching it improve.
log.info("*** Supervised training on icsup")
cross_entropy_loss = lambda model: model_cross_entropy(model, icsup)
hmm.train(corpus=icsup, loss=cross_entropy_loss, 
          minibatch_size=10, evalbatch_size=500, lr=0.01, tolerance=0.0001)

log.info("*** A, B matrices after training on icsup (should approximately "
         "match initial params on spreadsheet [transposed])")
hmm.printAB()

# Since we used a low tolerance, that should have gotten us about up to the
# initial parameters on the spreadsheet.  Let's tag the spreadsheet "sentence"
# (that is, the sequence of ice creams) using the Viterbi algorithm.
log.info("*** Viterbi results on icraw")
icraw = TaggedCorpus(Path("icraw"), tagset=icsup.tagset, vocab=icsup.vocab)
write_tagging(hmm, icraw, Path("icraw.output"))  # calls hmm.viterbi_tagging on each sentence
os.system("cat icraw.output")   # print the file we just created, and remove it

# Now let's use the forward algorithm to see what the model thinks about 
# the probability of the spreadsheet "sentence."
log.info("*** Forward algorithm on icraw (should approximately match iteration 0 "
             "on spreadsheet)")
for sentence in icraw:
    prob = math.exp(hmm.log_prob(sentence, icraw))
    log.info(f"{prob} = p({sentence})")

# Finally, let's reestimate on the icraw data, as the spreadsheet does.
log.info("*** Reestimating on icraw (perplexity should improve on every iteration)")
negative_log_likelihood = lambda model: model_cross_entropy(model, icraw)  # evaluate on icraw itself
hmm.train(corpus=icraw, loss=negative_log_likelihood,
          minibatch_size=10, evalbatch_size=500, lr=0.001, tolerance=0.0001)

log.info("*** A, B matrices after reestimation on icraw (SGD, not EM, but still "
         "should approximately match final params on spreadsheet [transposed])")
hmm.printAB()
