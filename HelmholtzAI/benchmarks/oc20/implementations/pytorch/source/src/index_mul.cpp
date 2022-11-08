#include "../common/pytorch_cpp_helper.hpp"
#include "../common/pytorch_device_registry.hpp"

void index_mul_float_forward_impl(    
    Tensor &out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    DISPATCH_DEVICE_IMPL(index_mul_float_forward_impl, out, in1, in2, idx1);
}

void index_mul_float_backward_impl(
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    DISPATCH_DEVICE_IMPL(index_mul_float_backward_impl, grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_float_backward_backward_impl(
    Tensor &grad_grad_out,
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &grad_grad_in1,
    const Tensor &grad_grad_in2,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    DISPATCH_DEVICE_IMPL(index_mul_float_backward_backward_impl, grad_grad_out, grad_in1, grad_in2, grad_out, 
                    grad_grad_in1, grad_grad_in2, in1, in2, idx1);
}

void index_mul_float_forward(    
    Tensor &out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    index_mul_float_forward_impl(out, in1, in2, idx1);
}

void index_mul_float_backward(    
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    index_mul_float_backward_impl(grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_float_backward_backward(    
    Tensor &grad_grad_out,
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &grad_grad_in1,
    const Tensor &grad_grad_in2,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    index_mul_float_backward_backward_impl(grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1, grad_grad_in2, in1, in2, idx1);
}