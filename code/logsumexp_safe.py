#!/usr/bin/env python3
 
# Author: Jason Eisner <jason@cs.jhu.edu>, December 2020.

# Monkey patch for torch.logaddexp and torch.logsumexp, to correctly zero out
# nans in the backward pass provided that the gradient with respect to the
# output is zero.  Such nans currently arise when all arguments are -inf or any
# argument is inf; since they represent indeterminate *finite* quantities,
# multiplying them by zero should give zero.
#
# The patch correctly handles tensorized versions where many logsumexp
# operations are done in parallel on slices of the tensor.

# Remark: This patch is needed to make ((-inf ⊕ -inf) ⊕ 1) behave the same as
# (-inf ⊕ (-inf ⊕ 1)) or ⊕(-inf, -inf, 1) where ⊕ denotes logsumexp. This issue
# arises in the forward algorithm for HMMs or CRFs if the probabilities are
# represented in the log domain and the parameters include structural zeroes.
# 
# I reported this issue to the PyTorch project, and they may fix it soon.
#   https://github.com/pytorch/pytorch/issues/49724

# TODO: To support all the same cases as the original logaddexp and logsumexp, 
# we would need to support the `out` argument, named dims, and maybe better typing. 
# We should also inherit and modify the original docstring.

# TODO: Should additionally patch logsumexp and softmax to correctly handle the
# cases where all arguments are -inf but there is only one argument, and where
# exactly one argument is inf.  In these cases, the forward pass of softmax 
# and the backward pass of logsumexp currently return nans, but a limit 
# argument suggests that they can validly return one-hot vectors.  (logaddexp 
# already gets the latter case right, but the other functions get it wrong, and in
# different ways.  logaddexp does not encounter the former case.)

import torch
from typing import Union, Tuple
from torch.autograd.function import BackwardCFunction
import torch.autograd as autograd

Dim = Union[int, Tuple[int]]   # probably need more branches to this union to be fully general

# Custom PyTorch functions.
# See https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
# and https://pytorch.org/docs/stable/nosafeinftes/extending.html

class LogAddExp_safe_inf(torch.autograd.Function):
    """Implements a torch function that is exactly like logaddexp, 
    but is willing to zero out nans on the backward pass."""
    
    @staticmethod
    def forward(ctx, input, other): # type: ignore
        with torch.enable_grad():
            output = torch.logaddexp_old(input, other) # internal copy of output
        ctx.save_for_backward(input, other, output)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        input, other, output = ctx.saved_tensors
        grad_input, grad_other = autograd.grad(output, (input, other), grad_output, only_inputs=True)
        if grad_output == 0:
            assert grad_input == 0 or grad_input.isnan()  # everything that was multiplied by 0 should be 0 or nan
            assert grad_other == 0 or grad_other.isnan()
            return torch.tensor(0.), torch.tensor(0.)   # force to 0, since nans were obtained in this case as 0 * nan
        else: 
            return grad_input, grad_other

class LogSumExp_safe_inf(torch.autograd.Function):
    """Implements a torch function that is exactly like logsumexp, 
    but is willing to zero out nans on the backward pass."""

    @staticmethod
    def forward(ctx, input, dim, keepdim = False): # type: ignore
        with torch.enable_grad():
            output = torch.logsumexp_old(input, dim, keepdim=keepdim) # internal copy of output
        ctx.save_for_backward(input, output)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        input, output = ctx.saved_tensors
        grad_input, = autograd.grad(output, input, grad_output, only_inputs=True)
        mult_by_zero = expand_dims(grad_output==0, input.size(), ctx.dim, keepdim=ctx.keepdim)
        assert torch.logical_or(torch.logical_not(mult_by_zero),  # everything that was mult by 0 should be 0 or nan
                                torch.logical_or(grad_input==0, grad_input.isnan())).all()
        return torch.where(mult_by_zero,      # force to 0, since nans were obtained in this case as 0 * nan
                           torch.tensor(0.), grad_input), None, None

# Utility function needed above
def expand_dims(x: torch.Tensor, 
                target_size: torch.Size, 
                dim: Dim,
                keepdim: bool = False) -> torch.Tensor:
    """x is the result of reducing a tensor of target_size 
    through some operation like x.sum(dim=dim, keepdim=keepdim).
    Here we stretch it out back out to target_size, without
    copying memory."""
    if not keepdim:
        for d in (dim,) if isinstance(dim, int) else sorted(dim):
            x = x.unsqueeze(d)
    try:
        return x.expand(target_size)  # will raise exception if args weren't appropriate
    except RuntimeError as exc:
        raise RuntimeError("x doesn't have the size implied by the other arguments") from exc


# Invoke either original or custom function, according to the safe_inf argument
def logaddexp_new(input: torch.Tensor, other: torch.Tensor, safe_inf: bool = False) -> torch.Tensor:
    """Modified version of the standard torch.logaddexp.
    If `safe_inf=True` is specified, it will try to avoid nans
    in the backward pass when the result is ±∞."""
    if safe_inf:
        result = LogAddExp_safe_inf.apply(input, other)
    else:
        result = torch.logaddexp_old(input, other)
    assert isinstance(result, torch.Tensor)
    return result

def logsumexp_new(x: torch.Tensor,
                  dim: Dim, *,
                  keepdim: bool = False,
                  safe_inf: bool = False) -> torch.Tensor:
    """Modified version of the standard torch.logsumexp.
    If `safe_inf=True` is specified, it will try to avoid nans
    in the backward pass when the result is ±∞."""
    if safe_inf:
        result = LogSumExp_safe_inf.apply(x, dim, keepdim)
    else:
        result =  torch.logsumexp_old(x, dim, keepdim=keepdim)
    assert isinstance(result, torch.Tensor)
    return result


# Monkey patch: replace the old methods with our improved versions
if not hasattr(torch, 'logaddexp_old'):
    torch.logaddexp_old = torch.logaddexp  # save original def so we can call it above
    torch.logsumexp_old = torch.logsumexp  # save original def so we can call it above
torch.logaddexp = logaddexp_new
torch.Tensor.logaddexp = logaddexp_new
torch.logsumexp = logsumexp_new
torch.Tensor.logsumexp = logsumexp_new

if __name__ == "__main__":
    inf=float('inf') 

    # Some examples with logaddexp
    for a in -inf, 1., inf:
        for b in -inf, 2., inf:
            print("")
            for c in -inf, 3.:
                for safe_inf in False, True:
                    aa = torch.tensor(a, requires_grad=True)
                    bb = torch.tensor(b, requires_grad=True)
                    result = aa.logaddexp(bb, safe_inf=safe_inf).logaddexp(torch.tensor(c))
                    result.backward()
                    print(f"{'  safe' if safe_inf else 'unsafe'}: "
                          f"d=logaddexp({a}, {b}, {c})={result.item()}"
                          f"\t∂d/∂a={aa.grad.item()}\t∂d/∂b={bb.grad.item()}")
    
    # Some examples with tensorized logsumexp
    t = torch.tensor([[  2.,   3., -inf, -inf], 
                      [  5.,   7., -inf, -inf],
                      [-inf, -inf, -inf, -inf]], requires_grad=True)
    u = torch.tensor([[  1.,   0.,   1.,   0.],
                      [  1.,   0.,   1.,   0.],
                      [  1.,   0.,   1.,   0.]])
    
    for dim in 0, 1, (0,1):
        for keepdim in False, True:
            print(f"\ndim={dim}, keepdim={keepdim} -----")
            for safe_inf in False, True:
                x = t.clone()   # test that backward works when logsumexp is applied to a non-leaf
                y = x.logsumexp(dim=dim, keepdim=keepdim, safe_inf=safe_inf)
                z = u.sum(dim=dim, keepdim=keepdim)  # reduce size to match y
                (y*z).sum().backward()               # the product with z means that some elements of y's grad_output will be zero
                print(t.grad)
                t.grad.data.zero_()
