#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Module for constructing a lexicon of word attributes.

import logging
from pathlib import Path
from typing import Optional, Set

import torch

from corpus import TaggedCorpus, BOS_WORD, EOS_WORD, OOV_WORD, Word

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def build_lexicon(corpus: TaggedCorpus,
                  one_hot: bool = False,
                  embeddings_file: Optional[Path] = None,
                  log_counts: bool = False,
                  affixes: bool = False) -> torch.Tensor:
    """Returns a lexicon, implemented as a matrix Tensor
    where each row defines real-valued attributes for one of
    the words in corpus.vocab.  This is a wrapper method that
    horizontally concatenates 0 or more matrices that provide 
    different kinds of attributes."""

    matrices = [torch.empty(len(corpus.vocab), 0)]  # start with no features for each word

    if one_hot: 
        matrices.append(one_hot_lexicon(corpus))
    if embeddings_file is not None:
        matrices.append(embeddings_lexicon(corpus, embeddings_file))
    if log_counts:
        matrices.append(log_counts_lexicon(corpus))
    if affixes:
        matrices.append(affixes_lexicon(corpus))

    return torch.cat(matrices, dim=1)   # horizontally concatenate 

def one_hot_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a one-hot embedding of the corresponding word.
    This allows us to learn features that are specific to the word."""

    return torch.eye(len(corpus.vocab))  # identity matrix

def embeddings_lexicon(corpus: TaggedCorpus, file: Path) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a vector embedding of the corresponding word.
    
    The second argument is a lexicon file in the format of Homework 2 and 3, 
    which is used to look up the word embeddings.

    The lexicon entries BOS, EOS, OOV, and OOL will be treated appropriately
    if present.  In particular, any words that are not in the lexicon
    will get the embedding of OOL (or 0 if there is no such embedding).
    """

    vocab = corpus.vocab
    with open(file) as f:
        filerows, cols = [int(i) for i in next(f).split()]   # first line gives num of rows and cols
        matrix = torch.empty(len(vocab), cols)   # uninitialized matrix
        seen: Set[int] = set()                   # the words we've found embeddings for
        ool_vector = torch.zeros(cols)           # use this for other words if there is no OOL entry
        specials = {'BOS': BOS_WORD, 'EOS': EOS_WORD, 'OOV': OOV_WORD}

        # Run through the words in the lexicon, keeping those that are in the vocab.
        for line in f:
            first, *rest = line.strip().split("\t")
            word = Word(first)
            vector = torch.tensor([float(v) for v in rest])
            assert len(vector) == cols     # check that the file didn't lie about # of cols

            if word == 'OOL':
                assert word not in vocab   # make sure there's not an actual word "OOL"
                ool_vector = vector
            else:
                if word in specials:    # map the special word names that may appear in lexicon
                    word = specials[word]    
                w = vocab.index(word)   # vocab integer to use as row number
                if w is not None:
                    matrix[w] = vector  # fill the vector into that row
                    seen.add(w)

    # Fill in OOL for any other vocab entries that were not seen in the lexicon.
    for w in range(len(vocab)):
        if w not in seen:
            matrix[w] = ool_vector

    log.info(f"From {file.name}, got embeddings for {len(seen)} of {len(vocab)} word types")

    return matrix

def log_counts_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a feature matrix with as many rows as corpus.vocab, where each
    row represents a feature vector for the corresponding word w.
    There is one feature (column) for each tag in corpus.tagset.  The value of this
    feature is log(1+c) where c=count(t,w) is the number of times t emitted w in supervised
    training data.  Thus, if this feature has weight 1.0 and is the only feature,
    then p(w | t) will be proportional to 1+count(t,w), just as in add-1 smoothing."""

    raise NotImplementedError   # you fill this in!

def affixes_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a feature matrix with as many rows as corpus.vocab, where each
    row represents a feature vector for the corresponding word w.
    Each row has binary features for common suffixes and affixes that the
    word has."""

    raise NotImplementedError

# Other feature templates could be added, such as word shape.
