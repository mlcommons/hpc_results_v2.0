#include "../include/index_mul.cuh"
#include "../../common/pytoch_cuda_hepler.hpp"
#include <iostream>

void IndexMulFloatForwardCUDAKernelLauncher(
    Tensor &out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    const int64_t size = in2.size(0);
    constexpr int64_t fea_dim = 64;
    if (size < 0){
        return;
    }

    at::cuda::CUDAGuard device_guard(out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int BLOCK_THREADS_DIMX = 16;
    const int BLOCK_THREADS_DIMY = 16;
    const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

    index_mul_float<fea_dim><<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
        out.data_ptr<float>(), in1.data_ptr<float>(), in2.data_ptr<float>(), 
        idx1.data_ptr<int64_t>(), size);

    AT_CUDA_CHECK(cudaGetLastError());
}

void IndexMulFloatBackwardCUDAKernelLauncher(
    Tensor &grad_in1,
    Tensor &grad_in2,
    const Tensor &grad_out,
    const Tensor &in1,
    const Tensor &in2,
    const Tensor &idx1)
{
    const int64_t size = in2.size(0);
    constexpr int64_t fea_dim = 64;
    if (size < 0){
        return;
    }

    at::cuda::CUDAGuard device_guard(grad_out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int BLOCK_THREADS_DIMX = 16;
    const int BLOCK_THREADS_DIMY = 16;
    const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

    index_mul_grad_float<fea_dim><<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
        grad_in1.data_ptr<float>(), grad_in2.data_ptr<float>(), grad_out.data_ptr<float>(), 
        in1.data_ptr<float>(), in2.data_ptr<float>(), idx1.data_ptr<int64_t>(), size);

    AT_CUDA_CHECK(cudaGetLastError());
}

void IndexMulFloatBackwardBackwardCUDAKernelLauncher(
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
    const int64_t size = in2.size(0);
    constexpr int64_t fea_dim = 64;
    if (size < 0){
        return;
    }

    at::cuda::CUDAGuard device_guard(grad_grad_out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int BLOCK_THREADS_DIMX = 16;
    const int BLOCK_THREADS_DIMY = 16;
    const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

    index_mul_grad_grad_float<fea_dim><<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
        grad_grad_out.data_ptr<float>(), grad_in1.data_ptr<float>(), grad_in2.data_ptr<float>(), 
        grad_out.data_ptr<float>(), grad_grad_in1.data_ptr<float>(), grad_grad_in2.data_ptr<float>(), 
        in1.data_ptr<float>(), in2.data_ptr<float>(), idx1.data_ptr<int64_t>(), size);

    AT_CUDA_CHECK(cudaGetLastError());
}
