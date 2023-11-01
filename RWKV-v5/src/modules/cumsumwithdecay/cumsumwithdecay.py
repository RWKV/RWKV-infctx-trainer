import os

import torch
from torch.utils.cpp_extension import load

VERBOSE = False


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_discounted_cumsum_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_discounted_cumsum_cpu')
    torch_discounted_cumsum_cpu = load(
        name='torch_discounted_cumsum_cpu',
        sources=[
            _resolve('cumsum_cpu.cpp'),
        ],
        verbose=VERBOSE,
    )


try:
    import torch_discounted_cumsum_cuda
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_discounted_cumsum_cuda')
    torch_discounted_cumsum_cuda = None
    if torch.cuda.is_available():
        torch_discounted_cumsum_cuda = load(
            name='torch_discounted_cumsum_cuda',
            sources=[
                _resolve('cumsum_cuda.cpp'),
                _resolve('cumsum_cuda_kernel.cu'),
            ],
            verbose=VERBOSE,
        )


def _discounted_cumsum_left_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(gamma):
        raise ValueError('Gamma must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum_left_cuda(input.contiguous(), gamma.contiguous())
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum_left_cpu(input, gamma)


def _discounted_cumsum_right_dispatcher(input, gamma):
    if not torch.is_tensor(input):
        raise ValueError('Input must be a torch.Tensor')
    if not torch.is_tensor(gamma):
        raise ValueError('Gamma must be a torch.Tensor')
    if input.is_cuda:
        if torch_discounted_cumsum_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_discounted_cumsum_cuda.discounted_cumsum_right_cuda(input.contiguous(), gamma.contiguous())
    else:
        return torch_discounted_cumsum_cpu.discounted_cumsum_right_cpu(input, gamma)


class DiscountedCumSumLeftFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, gamma_requires_grad):
        output = _discounted_cumsum_left_dispatcher(input, gamma)
        ctx.save_for_backward(output if gamma_requires_grad else None, gamma)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, gamma = ctx.saved_tensors
        grad_input = _discounted_cumsum_right_dispatcher(grad_output, gamma)
        grad_gamma = None
        if output is not None:
            z = _discounted_cumsum_left_dispatcher(output, gamma)
            z = z[:, :-1]
            dLdy = grad_output[:, 1:]
            grad_gamma = (z * dLdy).sum(dim=1)
        return grad_input, grad_gamma, None


class DiscountedCumSumRightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, gamma_requires_grad):
        output = _discounted_cumsum_right_dispatcher(input, gamma)
        ctx.save_for_backward(output if gamma_requires_grad else None, gamma)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, gamma = ctx.saved_tensors
        grad_input = _discounted_cumsum_left_dispatcher(grad_output, gamma)
        grad_gamma = None
        if output is not None:
            z = _discounted_cumsum_right_dispatcher(output, gamma)
            z = z[:, 1:]
            dLdy = grad_output[:, :-1]
            grad_gamma = (z * dLdy).sum(dim=1)
        return grad_input, grad_gamma, None


def discounted_cumsum_left(input, gamma):
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma).to(input)
    if gamma.dim() == 0:
        gamma = gamma.reshape(-1)
    return DiscountedCumSumLeftFunction.apply(input, gamma, gamma.requires_grad)


def discounted_cumsum_right(input, gamma):
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma).to(input)
    if gamma.dim() == 0:
        gamma = gamma.reshape(-1)
    return DiscountedCumSumRightFunction.apply(input, gamma, gamma.requires_grad)