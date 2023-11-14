# NLP Homework 6: Tagging

## Setup and Files

As in previous homeworks, you can activate the environment anytime using 

    conda activate nlp-class

After reading the reading handout, you probably want to study the files in this order.
**Boldface** indicates parts of the code that you will write.

* `integerize.py` -- converts words and tags to ints that can be used to index PyTorch tensors (we've used this before)
* `corpus.py` -- manage access to a corpus (compare `Probs.py` on the lm homework)
* `lexicon.py` -- construct a fixed matrix of word attributes, perhaps including **your own features**
* `hmm.py` -- parameterization, **Viterbi algorithm**, **forward algorithm**, training
* `eval.py` -- measure tagging accuracy
* `test_ic.py` -- uses the above to test Viterbi tagging, supervised learning, unsupervised learning on ice cream data
* `test_en.py` -- uses the above to train on larger English data and evaluate on accuracy
* `tag.py` -- **your command-line system**
* `crf.py` -- **support for CRFs** (start by copying your finished `hmm.py`)

In the last question, you will add support for CRFs.

You can experiment with these modules at the Python prompt.
For example:

    >>> from pathlib import Path
    >>> from corpus import *
    >>> c = TaggedCorpus(Path("ictrain"))
    >>> c.tagset
    >>> list(c.tagset)
    >>> list(c.vocab)
    >>> iter = c.get_sentences()
    >>> next(iter)
    >>> next(iter)

For some of the questions in the assignment, it will be easier for you to
work from the Python prompt or within a short script that encapsulates the
same functionality. You could also try working in a Jupyter notebook while
you're familiarizing yourself with the pieces, if you're familiar with 
Jupyter and prefer that style.

Your deliverables are *written answers to the questions*, plus *Python scripts*
(not notebooks or printouts of your interpreter session).

## The HMM

### Step 0

Later we will learn HMM parameters from data.  To warm up, however,
it'll help to compare against the 
[ice cream spreadsheet](http://cs.jhu.edu/~jason/465/hw-tag/hmm.xls) 
that we covered in class.  You can hard-code the initial spreadsheet
parameters into your HMM, something like this:

    >>> hmm = HiddenMarkovModel(...)
    >>> hmm.A = torch.Tensor(...)   # transition matrix
    >>> hmm.B = torch.Tensor(...)   # emission matrix

You can read the starter code for examples of how to create a `Tensor`.
You may also benefit from [PyTorch's own tutorial](https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html).
You can set individual elements of a `Tensor` the way you'd expect:

    >>> my_tensor[3, 5] = 8.0

Think about how to get those indices, though. (Where in `hmm.B` is the
parameter for emitting `3` while in state `H`?) You may want to use the
corpus's `integerize_tag` and `integerize_word` functions or access the
integerizers directly.

### Step 1

Implement the `viterbi_tagging` method in `hmm.py` as described in the handout.
Structurally, this method is very similar to the forward algorithm.
It has a handful of differences, though:

* You're taking the max instead of the sum over possible predecessor tags.
* You must track backpointers and reconstruct the best path in a backward pass.
* This function returns a sentence tagged with the highest-probability tag sequence,
  instead of returning a (log-)probability.

Remember to handle the BOS and EOS tags appropriately.

Run your implementation on the `icraw` data, using the hard-coded parameters 
from above.  To do this, you may want to look at how `test_ic.py` calls
`viterbi_tagging`.

Check your results against the
[Viterbi version of the spreadsheet](http://cs.jhu.edu/~jason/465/hw-tag/hmm-viterbi.xls),
Do your µ values match, for each word, if you print them out along the way?
When you follow the backpointers, do you get `HHHHHHHHHHHHHCCCCCCCCCCCCCCHHHHHH` as you should?
(These are rhetorical questions. The only questions you need to turn in answers to
are in the handout.)

Try out automatic evaluation: compare your Viterbi tagging to the correct answer in `icdev`,
using an appropriate method from `eval.py`.

### Step 2

The `train` method locally maximizes the (log-)likelihood of the parameters,
starting from the current parameter values (initially random when the HMM
is constructed).

There's just one thing missing: the actual computation of the log-likelihood.
You should implement the `log_forward` method in `hmm.py`.

For this step, it's enough to implement only the special case where the
tags are fully observed.  This will be enough to handle the complete-data
log likelihood for supervised training.  This is simply the log of the 
first formula in your reading handout.

At this point, you should be able to run the initial part of `test_ic.py`,
instead of hard-coding the probabilities.  Supervised training on `icsup` 
will be pretty fast and should converge to the parameters from the 
forward-backward spreadsheet.

Notice that `test_ic.py` uses one-hot word embeddings, allowing each word
to have different emission parameters, just as on the spreadsheet.  This
is discussed in the "word embeddings" section of the reading handout.

Training by SGD is able to handle arbitrary embeddings.  In this simple 
case, however, it's just a slow way of getting to count ratios that would
have been trivial to compute directly.  (The spreadsheet uses count ratios
in later iterations.)

### Step 3

You implemented only a special case of `log_forward`.
Go ahead and make it more general, so that it can also deal with
unsupervised data (where tags are `None`).  This should not change 
your results from the previous step.

You should now be able to run all of `test_ic.py`, which
tries out the forward algorithm on `icraw` and then uses it
for training.

First, check the results of the initial forward pass against iteration 0 of the 
[forward-backward spreadsheet](http://cs.jhu.edu/~jason/465/hw-tag/hmm.xls).
Do your α values match, for each word?

### Step 4

If you continue to the last step of `test_ic.py` and train on `icraw`,
using the forward algorithm repeatedly, do you eventually get to the
same parameters as EM does?

This is a good time to stop and check for speed.  When training on 
`icraw`, our own implementation was able to process 60 to 90 unsupervised 
'training sentences' per second on a 2017 laptop with 2 1.8 GHz Intel cores.
(This is the iterations per second, or `it/s`, reported by the `tqdm` progress bar.)
It's at about 120 it/s on a 2020 laptop with 4+ cores.  If your code is markedly slower, 
you'll probably want to speed it up before we move to a larger dataset -- 
probably by making less use of loops and more use of fast tensor operations.

(It's sad that this SGD code is so much slower than the EM algorithm
implemented on the spreadsheet.  Of course, the SGD method is more
general.  The EM code is fast only when the M step is easy -- in the
special case on the spreadsheet, the M step can get the optimal new
probabilities simply by dividing (fractional) count ratios, rather
than by using a gradient.  But we'll need a gradient anyway when
we're using pretrained word embeddings.)

*Note:* Matrix multiplication is available in PyTorch (and numpy)
using the `matmul` function.  In the simplest case, it can be invoked
with the [infix operator](https://www.python.org/dev/peps/pep-0465/)
`C = A @ B`, which works the way you'd expect from your linear algebra
class.  A different syntax, `D = A * B`, performs _element-wise_
multiplication of two matrices whose entire shapes match.  (It also
works if they're "[broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html),"
like if `A` is 1x5 and `B` is 3x5.  See also [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).)

### Step 5

Now let's move on to real data!  Try out the workflow in `test_en.py`.
If you come across numerical stability problems from working with 
products of small probabilities, fix them using one of the methods
in the "Numerical Stability" section of the reading handout.

### Step 6

Now it's time to package up your work for the autograder.   Make a 
command-line script with good hyperparameters that can be run as follows:

    $ python3 tag.py <eval_file> --model <model_file> --train <training_files>

This should evaluate an HMM on the `eval file`, using the *error rate* of a Viterbi tagger.
Where does the HMM come from?  It is loaded from the `model_file` 
and then trained further on the `training_files` until the error rate metric is
no longer improving.  The improved model is saved back to the `model_file` at the 
end.

If the `model_file` doesn't exist yet or isn't provided, then the script should 
create a new randomly initialized HMM.  If no `training_files` are provided, 
then the model will not be trained further.

Thus, our autograder will be able to replicate roughly the `test_en.py`
workflow like this:

    $ python3 tag.py endev --model your_hmm.pkl --train ensup        # supervised training
    $ python3 tag.py endev --model your_hmm.pkl --train ensup enraw  # semi-supervised training

and it then will be able to evaluate the error rate of your saved model on a test file like this:
  
    $ python3 tag.py ensup  --model your_hmm.pkl  # error rate on training data
    $ python3 tag.py entest --model your_hmm.pkl  # error rate on held-out test data

Your `tag.py` should also output the Viterbi taggings of all sentences in the `eval_file`
to a text file, in the usual format.  For example, if the `eval_file` is called
`endev`, then it should create a file called `endev.output` with lines like
    
    Papa/N ate/V the/D caviar/N with/P a/D spoon/N ./.

As it works, `tag.py` is free to print additional text to the standard 
error stream, e.g., by using Python's \texttt{logging} library.
This can report other information that you may want to see,
including the tags your program picks, its perplexity and accuracy as 
it goes along, various probabilities, etc. Anything printed to standard 
error will be ignored by the autograder; use it however you'd like. 
Maybe print a few kind words to the TAs.

You're entirely welcome (and encouraged) to add other command line parameters. This will
make your hyperparameter searching much easier; you can write a script that loops over
different values to automate your search. You may also be able to parallelize this search.
Make sure that your submitted code has default values set how you want them, so that we 
run the best version.  (Don't make the autograder run your hyperparameter search.)

### Step 7

Your performance so far will be rather bad -- under 90%.  That's worse than
the simple baseline method described in the reading handout: we 
implemented a version of that method and found that it got 91.5% 
accuracy on `endev` data.

So, make some improvement to your HMM!  When `tag.py` creates a new model,
it should use the improved HMM if the `--awesome` flag is specified.
(Any model that is loaded or saved should already know whether it is awesome;
typically an awesome model will use a different class and have a different 
set of parameters.  You probably want to define `AwesomeHMM` as a
subclass of `HiddenMarkovModel`.)

Some options for improving performance are given in the handout.  Your goal is to 
beat the baseline method.  If you decide to experiment with features, check out
`lexicon.py` for some ideas.

Evaluate the accuracy of your trained model on both `ensup` and `endev`, 
for example by using `vtag.py`.  It's actually possible to do quite well 
on training data (high 90's), with decent performance on held-out data 
as well (low 90's).

Also do an ablation study -- turn something off in your `--awesome` tagger
and see whether that hurts.  The easiest option is to "turn off the transition 
probabilities" by giving the flag `unigram=True` when you construct your 
`HiddenMarkovModel`.  How does this ablation affect the different categories 
of accuracy?  Why?

### Step 8 (**extra credit**)

Try implementing posterior decoding as described in the reading handout.
This will rely on your implementation of the forward algorithm.  As
the reading handout explains, there are two options:

* Implement the backward pass yourself.  
* An awesome but tricky alternative is to let PyTorch do the work for you 
  with its backward pass, which will give you both β values (a good warmup) and
  posterior marginal probabilities.  (The tricky part is finding
  where the relevant values are stored, without ballooning the storage space
  required.)

The posterior marginal probabilities for the `icraw` data are shown on
the spreadsheet.  You can use these to check your code.

Alter your `vtag` script so that the output 
instead of Viterbi decoding. On the English data, how much better does posterior
decoding do, in terms of accuracy? Do you notice anything odd about the outputs?

## The CRF 

We constructed a lot of the `HiddenMarkovModel` class for you.
Now it's your turn to build a model class!

Implement a CRF with RNN-based features, as discussed in the reading handout.
You can use `HiddenMarkovModel` as a starting point.  

Only the model is changing.  So the supporting files like `corpus.py`, 
`lexicon.py`, and `eval.py` should not have to change.  The `train` 
method should also not have to change, although you might want to
choose different hyperparameters for its arguments when you are 
training a CRF.

### Step 0

Start by copying your `hmm.py` to `crf.py`, and rename the `HiddenMarkovModel` class in `crf.py`.

Improve `tag.py` so that if it has the `--crf` flag, it will use a CRF instead of an HMM.

For trying out your code, you may also want to modify 
`test_ic` and `test_en` so that they use the CRF instead of the HMM.

### Step 1

Edit your new class so that it maximizes the *conditional* log-likelihood.

That is, the `log_prob` function used in training should now return the log 
of the *conditional* probability p(y | x).  *Hint:* This will have to call 
`log_forward` twice ...

This is discriminative training.  Does it get better accuracy than your HMM
on `endev` when trained on `ensup`?

There is no point in training on `enraw`.  As usual, if y is not fully observed, 
then `log_prob` should marginalize over the possible values of y, as usual.  
If y is *completely* unobserved, then `log_prob` will always return the log of
sum_y p(y | x) = 1.  The gradient of this constant with respect to the
model parameters is 0.  What happens in practice when you include `enraw` 
in the training data?

### Step 2

So far, your CRF is just a discriminatively trained HMM.  The `A` and `B`
matrices still hold conditional probabilities that sum to 1.  Change the 
computation so that these matrices are computed from the parameters
using `exp` instead of `softmax`, so that their entries can be arbitrary 
positive numbers ("potentials").

Now you have a proper CRF, although it turns out that you
have not actually added any expressive power.  Try training it.
Does it work any better?

### Step 3 

Make your CRF use bidirectional RNN features as suggested in the reading 
handout (the section on CRF parameterization).  Does this improve your
tagging accuracy?

Some things you'll have to do:

* Instead of computing A and B matrices from the parameters at the start
  of every minibatch, you'll have to compute fresh A and B matrices at
  every position j of every sentence.  These are the *contextual* 
  transition and emission probabilities.

* You'll first want to compute the RNN vectors h and h' vectors 
  at all positions of the sentence, since those will help you compute 
  the necessary A and B matrices.  You might consider putting that 
  RNN code into another function, but that's up to you.

* You'll need to add parameters to the model to help you compute all
  these things (the various θ, M, and U parameters described in the
  reading handout).
  *Implementation hint:* Remember to add these to your model 
  as `nn.Parameter` objects, so that they're listed 
  in `your_model.parameters()` and passed to the SGD
  optimizer.

* You can use one-hot embeddings for the tag embeddings, or if you prefer,
  you can make the tag embeddings be learned parameters, too.

* The word embeddings come from `lexicon.py` as before.  As with the HMM,
  you could fine-tune them, or not.

## Using Kaggle

Assuming you've vectorized your code as the reading handout urged you to do,
you can optionally use a Kaggle GPU to speed up your training and tagging.  Follow
the [instructions](https://www.cs.jhu.edu/~jason/465/hw-lm/code/INSTRUCTIONS.html#using-kaggle) from Homework 3; just change `hw-lm` to `hw-tag` throughout.
In particular, the dataset for this homework is at
<https://www.kaggle.com/datasets/jhunlpclass/hw-tag-data>.
You will eventually need to _also_ add the Homework 3 dataset
<https://www.kaggle.com/datasets/jhunlpclass/hw-lm-data> to 
your notebook so that you can get a lexicon of word embeddings.

## What to submit
You should submit the following files under **Assignment 6 - Programming**:

- `tag.py`
- `hmm.py`
- `eval.py`
- `crf.py`
- `corpus.py`
- `lexicon.py`
- `ic_hmm.pkl` (ice-cream supervised)
- `ic_hmm_raw.pkl` (ice-cream semi-supervised)
- `en_hmm.pkl` (english supervised)
- `en_hmm_raw.pkl` (english semi-supervised)
- `en_hmm_awesome.pkl` (english awesome)
- `ic_crf.pkl` (ice-cream supervised with crf)
- `en_crf.pkl` (english supervised crf)
- `en_crf_raw.pkl` (english supervised crf + enraw)
- `en_crf_birnn.pkl` (english supervised birnn plus improvement)
- Any additional dependencies of your code, such as `integerize.py` and `logsumexp_safe.py`

Try your code out early as it can take a bit of time to run the autograder. Autograder should be good but please let us know if anything is broken so we can fix it ASAP.

Additional Note: Please don’t submit the output files that show up in the autograder’s feedback message. Rather, these will be produced by running your code! If you do submit them, the autograder will not grade your assignment.
