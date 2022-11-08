#include <torch/extension.h>

#include "../common/pytorch_cpp_helper.hpp"

void index_mul_float_forward(    
    Tensor &out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1);

void index_mul_float_backward(    
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1);

void index_mul_float_backward_backward(    
    Tensor &grad_grad_out,
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &grad_grad_in1,
    const Tensor &grad_grad_in2,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1);


TORCH_LIBRARY(index_mul, m) {
  m.def("index_mul_float_forward", index_mul_float_forward);
  m.def("index_mul_float_backward", index_mul_float_backward);
  m.def("index_mul_float_backward_backward", index_mul_float_backward_backward);
}