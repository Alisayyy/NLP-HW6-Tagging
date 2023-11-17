#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Implementation of Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp, sqrt
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast
import logsumexp_safe

import torch
from torch import Tensor as Tensor
from torch import tensor as tensor
from torch import optim as optim
from torch import nn as nn
from torch import cuda as cuda
from torch.nn import functional as F
from torch.nn.parameter import Parameter as Parameter
from jaxtyping import Float
from typeguard import typechecked
from tqdm import tqdm # type: ignore
import pickle

from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word)
from integerize import Integerizer

TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel(nn.Module):
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The unigram
        flag says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended
        to support higher-order HMMs: trigram HMMs used to be popular.)"""

        super().__init__() # type: ignore # pytorch nn.Module does not have type annotations

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)
        # As usual in Python, attributes starting with _ are intended as private;
        # in this case, they might go away if you changed the parametrization of the model.

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        assert vocab[-2:] == [EOS_WORD, BOS_WORD]  # make sure these are the last two

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.d = lexicon.size(1)   # dimensionality of a word's embedding in attribute space
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab
        self._E = lexicon[:-2]  # embedding matrix; omits rows for EOS_WORD and BOS_WORD

        # Useful constants that are invoked in the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        assert self.bos_t is not None    # we need this to exist
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize params


    @property
    def device(self) -> torch.device:
        """Get the GPU (or CPU) our tensors are computed and stored on."""
        return next(self.parameters()).device


    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> List[Tuple[int,Optional[int]]]:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        # Make sure that the sentence comes from a corpus that this HMM knows
        # how to handle.
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        # If so, go ahead and integerize it.
        return corpus.integerize_sentence(sentence)

    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        However, we initialize the BOS_TAG column of _WA to -inf, to ensure that
        we have 0 probability of transitioning to BOS_TAG (see "Don't guess when you know").
        See the "Parametrization" section of the reading handout."""

        # See the reading handout section "Parametrization.""

        ThetaB = 0.01*torch.rand(self.k, self.d)    
        self._ThetaB = Parameter(ThetaB)    # params used to construct emission matrix

        WA = 0.01*torch.rand(1 if self.unigram # just one row if unigram model
                             else self.k,      # but one row per tag s if bigram model
                             self.k)           # one column per tag t
        WA[:, self.bos_t] = -inf               # correct the BOS_TAG column
        self._WA = Parameter(WA)            # params used to construct transition matrix


    @typechecked
    def params_L2(self) -> TorchScalar:
        """What's the L2 norm of the current parameter vector?
        We consider only the finite parameters."""
        l2 = tensor(0.0)
        for x in self.parameters():
            x_finite = x[x.isfinite()]
            l2 = l2 + x_finite @ x_finite   # add ||x_finite||^2
        return l2


    def updateAB(self) -> None:
        """Set the transition and emission matrices A and B, based on the current parameters.
        See the "Parametrization" section of the reading handout."""
        
        A = F.softmax(self._WA, dim=1)       # run softmax on params to get transition distributions
                                             # note that the BOS_TAG column will be 0, but each row will sum to 1
        if self.unigram:
            # A is a row vector giving unigram probabilities p(t).
            # We'll just set the bigram matrix to use these as p(t | s)
            # for every row s.  This lets us simply use the bigram
            # code for unigram experiments, although unfortunately that
            # preserves the O(nk^2) runtime instead of letting us speed 
            # up to O(nk) in the unigram case.
            self.A = A.repeat(self.k, 1)
        else:
            # A is already a full matrix giving p(t | s).
            self.A = A

        WB = self._ThetaB @ self._E.t()  # inner products of tag weights and word embeddings
        B = F.softmax(WB, dim=1)         # run softmax on those inner products to get emission distributions
        self.B = B.clone()
        self.B[self.eos_t, :] = 0        # but don't guess: EOS_TAG can't emit any column's word (only EOS_WORD)
        self.B[self.bos_t, :] = 0        # same for BOS_TAG (although BOS_TAG will already be ruled out by other factors)


    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")


    @typechecked
    def log_prob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        return self.log_forward(sentence, corpus)


    @typechecked
    def log_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward 
        probability).

        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're 
        integerizing correctly."""

        # The "nice" way to construct alpha is by appending to a List[Tensor] at each
        # step.  But to better match the notation in the handout, we'll instead preallocate
        # a list of length n+2 so that we can assign directly to alpha[j].
        sent = self._integerize_sentence(sentence, corpus) 

        # self.A = self.A + 1e-45
        # self.B = self.B + 1e-45
        #alpha = [torch.empty(self.k) for _ in sent]
        n = len(sent)-2
        alpha = torch.full((len(sent), self.k), -float('inf'))

        alpha[0][self.bos_t] = 1

        for j in range(1, n+1):
            word = sent[j][0]
            tag = sent[j][1]
            if tag == None:
                pass
            else:
                alpha[j][tag] = logsumexp_safe.logsumexp_new(alpha[j-1] + torch.log(self.A[:, tag]) + torch.log(self.B[tag,word]),dim=0, keepdim=False, safe_inf=True)

        alpha[n+1][self.eos_t] =logsumexp_safe.logsumexp_new(alpha[n] + torch.log(self.A[:, self.eos_t]), dim=0, keepdim=False, safe_inf=True)
        
        Z = alpha[n+1][self.eos_t]

        return Z

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # It continues to call the vector alpha, without the ^ that is added
        # in the handout.

        # We'll start by integerizing the input Sentence.
        # But make sure you deintegerize the words and tags again when
        # constructing the return value, since the type annotation on
        # this method says that it returns a Sentence object, and
        # that's what downstream methods like eval_tagging will
        # expect.  (Running mypy on your code will check that your
        # code conforms to the type annotations ...)

        sent = self._integerize_sentence(sentence, corpus)
        print("sentence")
        print(sentence)
        print("sent")
        print(sent)
        
        n = len(sent)-2
        viterbi = torch.full((len(sent), self.k), -float('inf'))
        backpointers = torch.zeros(len(sent), self.k, dtype=torch.long)

        # viterbi[0] = torch.ones(self.k) # viterbi[0] = 0
        viterbi[0][self.tagset.index(BOS_TAG)] = 1
        print("viterbi")
        print(viterbi)
        
        for j in range(1, n+1):
            for tj in range(self.k):
                viterbi[j][tj], backpointers[j][tj] = max((viterbi[j-1][ts] * self.A[ts, tj] * self.B[tj, sent[j][0]], ts) for ts in range(self.k))
                # for ts in range(self.k):
                #     p1 = self.A[ts][tj] 
                #     p2 = self.B[tj][sent[j][0]]
                #     p = p1 * p2
                #     if viterbi[j][tj] < viterbi[j-1][ts] * p:
                #         viterbi[j][tj] = viterbi[j-1][ts] * p
                #         backpointers[j][tj] = ts

        print("viterbi after algo")
        print(viterbi)

        best_path = []
        # append EOS_TAG
        best_path.append(self.tagset.index(EOS_TAG))
        best_last_tag = viterbi[-2].argmax().item()

        #backtrace
        best_path.append(best_last_tag)
        #for j from n+1 downto 1
        for j in range(n, 0, -1):
            best_path.append(backpointers[j][best_path[-1]].item())
        best_path.reverse()

        print("backpointers")
        print(backpointers)
        print("best_path")
        print(best_path)
        print(len(best_path))

        integerized_sentence = [(self.vocab[sent[i][0]], self.tagset[best_path[i]]) for i in range(len(sent))]
        result_sentence = [tuple(map(str, tword)) for tword in integerized_sentence]

        return Sentence(result_sentence)

        # return [(self.vocab[sent[i][0]], self.tagset[best_path[i]]) for i in range(len(sent))]

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              tolerance: float =0.001,
              minibatch_size: int = 1,
              evalbatch_size: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              save_path: Path = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        The minibatch size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus.)
        The evalbatch size controls how often we evaluate (e.g., on a development corpus).
        We will stop when the relative improvement of the evaluation loss,
        since the last evalbatch, is less than the tolerance; in particular,
        we will stop when the improvement is negative, i.e., the evaluation loss 
        is getting worse (overfitting).
        lr is the learning rate, and reg is an L2 batch regularization coefficient."""

        # This is relatively generic training code.  Notice however that the
        # updateAB step before each minibatch produces A, B matrices that are
        # then shared by all sentences in the minibatch.

        # All of the sentences in a minibatch could be treated in parallel,
        # since they use the same parameters.  The code below treats them
        # in series, but if you were using a GPU, you could get speedups
        # by writing the forward algorithm using higher-dimensional tensor 
        # operations that update alpha[j-1] to alpha[j] for all the sentences
        # in the minibatch at once, and then PyTorch could actually take
        # better advantage of hardware parallelism.

        assert minibatch_size > 0
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        assert reg >= 0

        old_dev_loss: Optional[float] = None    # we'll keep track of the dev loss here

        optimizer = optim.SGD(self.parameters(), lr=lr)  # optimizer knows what the params are
        self.updateAB()                                        # compute A and B matrices from current params
        log_likelihood = tensor(0.0, device=self.device)       # accumulator for minibatch log_likelihood
        for m, sentence in tqdm(enumerate(corpus.draw_sentences_forever())):
            # Before we process the new sentence, we'll take stock of the preceding
            # examples.  (It would feel more natural to do this at the end of each
            # iteration instead of the start of the next one.  However, we'd also like
            # to do it at the start of the first time through the loop, to print out
            # the dev loss on the initial parameters before the first example.)

            # m is the number of examples we've seen so far.
            # If we're at the end of a minibatch, do an update.
            if m % minibatch_size == 0 and m > 0:
                logger.debug(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                optimizer.zero_grad()          # backward pass will add to existing gradient, so zero it
                objective = -log_likelihood + (minibatch_size/corpus.num_tokens()) * reg * self.params_L2()
                objective.backward()           # type: ignore # compute gradient of regularized negative log-likelihod
                length = sqrt(sum((x.grad*x.grad).sum().item() for x in self.parameters()))
                logger.debug(f"Size of gradient vector: {length}")  # should approach 0 for large minibatch at local min
                optimizer.step()               # SGD step
                self.updateAB()                # update A and B matrices from new params
                log_likelihood = tensor(0.0, device=self.device)    # reset accumulator for next minibatch

            # If we're at the end of an eval batch, or at the start of training, evaluate.
            if m % evalbatch_size == 0:
                with torch.no_grad():       # type: ignore # don't retain gradients during evaluation
                    dev_loss = loss(self)   # this will print its own log messages
                if old_dev_loss is not None and dev_loss >= old_dev_loss * (1-tolerance):
                    # we haven't gotten much better, so stop
                    self.save(save_path)  # Store this model, in case we'd like to restore it later.
                    break
                old_dev_loss = dev_loss            # remember for next eval batch

            # Finally, add likelihood of sentence m to the minibatch objective.
            log_likelihood = log_likelihood + self.log_prob(sentence, corpus)


    def save(self, model_path: Path) -> None:
        logger.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {model_path}")


    @classmethod
    def load(cls, model_path: Path, gpu: bool = False) -> HiddenMarkovModel:
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=('cuda' if gpu else 'cpu'))
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(result)} from saved file {model_path}.")
        logger.info(f"Loaded model from {model_path}")
        return model
