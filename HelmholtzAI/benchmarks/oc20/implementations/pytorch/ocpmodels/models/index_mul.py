import os

import torch
from torch.autograd import Function

ops_library_abs_path = os.path.abspath("/workspace/oc20/source/build/liboc20_customized_ops.so")
torch.ops.load_library(ops_library_abs_path)


class IndexMul(Function):
    @staticmethod
    def forward(ctx, in1: torch.Tensor, in2: torch.Tensor, idx1: torch.Tensor) -> torch.Tensor:
        if not in1.is_contiguous():
            in1 = in1.contiguous()
        if not in2.is_contiguous():
            in2 = in2.contiguous()
        if not idx1.is_contiguous():
            idx1 = idx1.contiguous()

        assert in1.is_contiguous()
        assert in2.is_contiguous()
        assert idx1.is_contiguous()

        out = torch.empty_like(in2)
        torch.ops.index_mul.index_mul_float_forward(out, in1, in2, idx1)
        ctx.for_backwards = (in1, in2, idx1)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        in1, in2, idx1 = ctx.for_backwards
        grad_in1, grad_in2 = index_mul_backward(in1, in2, idx1, grad_out)
        return grad_in1, grad_in2, None


class IndexMulBackward(Function):
    """compute Pbc distances backward."""

    @staticmethod
    def forward(ctx, in1: torch.Tensor, in2: torch.Tensor, idx1: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        if not in1.is_contiguous():
            in1 = in1.contiguous()
        if not in2.is_contiguous():
            in2 = in2.contiguous()
        if not idx1.is_contiguous():
            idx1 = idx1.contiguous()
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        assert in1.is_contiguous()
        assert in2.is_contiguous()
        assert idx1.is_contiguous()
        assert grad_out.is_contiguous()

        grad_in1 = torch.zeros_like(in1)
        grad_in2 = torch.empty_like(in2)
        torch.ops.index_mul.index_mul_float_backward(grad_in1, grad_in2, grad_out, in1, in2, idx1)

        ctx.for_backwards = (in1, in2, idx1, grad_out)
        return grad_in1, grad_in2

    @staticmethod
    def backward(ctx, grad_grad_in1, grad_grad_in2):
        if not grad_grad_in1.is_contiguous():
            grad_grad_in1 = grad_grad_in1.contiguous()
        if not grad_grad_in2.is_contiguous():
            grad_grad_in2 = grad_grad_in2.contiguous()

        assert grad_grad_in1.is_contiguous()
        assert grad_grad_in2.is_contiguous()

        in1, in2, idx1, grad_out = ctx.for_backwards

        grad_in1 = torch.zeros_like(in1)
        grad_in2 = torch.empty_like(in2)
        grad_grad_out = torch.empty_like(grad_out)

        torch.ops.index_mul.index_mul_float_backward_backward(
            grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1, grad_grad_in2, in1, in2, idx1
        )

        return grad_in1, grad_in2, None, grad_grad_out


index_mul = IndexMul.apply
index_mul_backward = IndexMulBackward.apply
