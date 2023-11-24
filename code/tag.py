#!/usr/bin/env python3
"""
Command-line interface for training and evaluating HMM and CRF taggers.
"""
import argparse
import logging
from pathlib import Path
from eval import model_cross_entropy, write_tagging
from hmm import HiddenMarkovModel
from crf import CRF
from lexicon import build_lexicon
from corpus import TaggedCorpus

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval", type=str, help="evalutation file")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="optional initial model file to load (will be trained further).  Loading a model overrides most of the other options."
    )
    parser.add_argument(
        "-l",
        "--lexicon",
        type=str,
        help="newly created model (if no model was loaded) should use this lexicon file",
    )
    parser.add_argument(
        "--crf",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should be a CRF"
    )
    parser.add_argument(
        "-u",
        "--unigram",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should only be a unigram HMM or CRF"
    )
    parser.add_argument(
        "-a",
        "--awesome",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should use extra improvements"
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        nargs="*",
        help="training files to train the model further"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=50000,
        help="maximum number of steps to train to prevent training for too long "
             "(this is an practical trick that you can choose implement in the `train` method of hmm.py and crf.py)"
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1.0,
        help="l2 regularization during further training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate during further training"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="tolerance for early stopping"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Compute tensors using a GPU (which must be available)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="tmp.model",
        help="where to save the trained model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="where to save the prediction outputs"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.model and not args.lexicon:
        parser.error("Please provide lexicon file path when no model provided")
    if not args.model and not args.train:
        parser.error("Please provide at least one training file when no model provided")
    return args

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    if args.gpu:
        # Direct all tensors to be computed on the GPU by default, 
        # This will give errors unless you are running with a GPU,
        # for example in a Kaggle Notebook where you have turned o
        # GPU acceleration.
        import torch
        torch.set_default_device('cuda')

    train = None
    model = None
    if args.model is not None:
        if args.crf:
            model = CRF.load(Path(args.model), gpu=args.gpu)
        else:
            model = HiddenMarkovModel.load(Path(args.model), gpu=args.gpu)
        assert model is not None
        tagset = model.tagset
        vocab = model.vocab
        if args.train is not None:
            train = TaggedCorpus(*[Path(t) for t in args.train], tagset=tagset, vocab=vocab)
    else:
        train = TaggedCorpus(*[Path(t) for t in args.train])
        tagset = train.tagset
        vocab = train.vocab
        if args.crf:
            lexicon = build_lexicon(train, embeddings_file=Path(args.lexicon), log_counts=args.awesome)
            model = CRF(tagset, vocab, lexicon, unigram=args.unigram)
        else:
            lexicon = build_lexicon(train, embeddings_file=Path(args.lexicon), log_counts=args.awesome)
            model = HiddenMarkovModel(tagset, vocab, lexicon, unigram=args.unigram)

    dev = TaggedCorpus(Path(args.eval), tagset=tagset, vocab=vocab)
    if args.train is not None:
        assert train is not None and model is not None
        # you can instantiate a different development loss depending on the question / which one optimizes performance
        dev_loss =  lambda x: model_cross_entropy(x, dev)
        model.train(corpus=train,
                    loss=dev_loss,
                    minibatch_size=args.train_batch_size,
                    evalbatch_size=args.eval_batch_size,
                    lr=args.lr,
                    reg=args.reg,
                    save_path=args.save_path,
                    tolerance=args.tolerance)
    write_tagging(model, dev, Path(args.eval+".output") if args.output_file is None else args.output_file)


if __name__ == "__main__":
    main()