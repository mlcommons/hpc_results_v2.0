#ifndef INDEX_MUL_CUH
#define INDEX_MUL_CUH

#include <stdio.h>
#include <cuda.h>

template<int64_t fea_dim>
__global__ void index_mul_float(
    float *out, 
    const float *in1, 
    const float *in2, 
    const int64_t *idx1, 
    const int64_t size) 
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int start_idx = bidx * blockDim.y + tidy;

    if (start_idx < size) {
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim) / 4 + tidx;
        int64_t vec_idx2 = (start_idx * fea_dim) / 4 + tidx;
        
        float4 res, src1, src2;
        src1 = reinterpret_cast<const float4 *>(in1)[vec_idx1];
        src2 = reinterpret_cast<const float4 *>(in2)[vec_idx2];
        res.x = src1.x * src2.x;
        res.y = src1.y * src2.y;
        res.z = src1.z * src2.z;
        res.w = src1.w * src2.w;
        reinterpret_cast<float4 *>(out)[vec_idx2] = res;
    }
}

template<int64_t fea_dim>
__global__ void index_mul_grad_float(
    float *grad_in1, 
    float *grad_in2,
    const float *grad_out, 
    const float *in1,
    const float *in2,
    const int64_t *idx1, 
    const int64_t size) 
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int start_idx = bidx * blockDim.y + tidy;

    if (start_idx < size) {
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim) / 4 + tidx;
        int64_t vec_idx2 = (start_idx * fea_dim) / 4 + tidx;

        float4 src_in1, src_in2, src_grad_out, dst_grad_in2;
        src_grad_out = reinterpret_cast<const float4 *>(grad_out)[vec_idx2];
        src_in1 = reinterpret_cast<const float4 *>(in1)[vec_idx1];
        src_in2 = reinterpret_cast<const float4 *>(in2)[vec_idx2];
        int64_t grad_in1_base_idx = idx1[start_idx] * fea_dim + tidx * 4;
        atomicAdd(grad_in1 + grad_in1_base_idx + 0, src_grad_out.x * src_in2.x);
        atomicAdd(grad_in1 + grad_in1_base_idx + 1, src_grad_out.y * src_in2.y);
        atomicAdd(grad_in1 + grad_in1_base_idx + 2, src_grad_out.z * src_in2.z);
        atomicAdd(grad_in1 + grad_in1_base_idx + 3, src_grad_out.w * src_in2.w);
        dst_grad_in2.x = src_grad_out.x * src_in1.x;
        dst_grad_in2.y = src_grad_out.y * src_in1.y;
        dst_grad_in2.z = src_grad_out.z * src_in1.z;
        dst_grad_in2.w = src_grad_out.w * src_in1.w;
        reinterpret_cast<float4 *>(grad_in2)[vec_idx2] = dst_grad_in2; 
    }
}

template<int64_t fea_dim>
__global__ void index_mul_grad_grad_float(
    float *grad_grad_out,
    float *grad_in1,
    float *grad_in2,
    const float *grad_out,
    const float *grad_grad_in1,
    const float *grad_grad_in2,
    const float *in1,
    const float *in2,
    const int64_t *idx1,
    const int64_t size) 
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int start_idx = bidx * blockDim.y + tidy;

    if (start_idx < size) { 
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim) / 4 + tidx;
        int64_t vec_idx2 = (start_idx * fea_dim) / 4 + tidx;

        float4 src_grad_grad_in1, src_in1, src_grad_grad_in2, src_in2, src_grad_out;
        float4 dst_grad_grad_out, dst_grad_in2;
        src_grad_grad_in1 = reinterpret_cast<const float4 *>(grad_grad_in1)[vec_idx1];
        src_in1 = reinterpret_cast<const float4 *>(in1)[vec_idx1];
        src_grad_grad_in2 = reinterpret_cast<const float4 *>(grad_grad_in2)[vec_idx2];
        src_in2 = reinterpret_cast<const float4 *>(in2)[vec_idx2];
        dst_grad_grad_out.x = src_grad_grad_in1.x * src_in2.x + src_grad_grad_in2.x * src_in1.x;
        dst_grad_grad_out.y = src_grad_grad_in1.y * src_in2.y + src_grad_grad_in2.y * src_in1.y;
        dst_grad_grad_out.z = src_grad_grad_in1.z * src_in2.z + src_grad_grad_in2.z * src_in1.z;
        dst_grad_grad_out.w = src_grad_grad_in1.w * src_in2.w + src_grad_grad_in2.w * src_in1.w;
        reinterpret_cast<float4 *>(grad_grad_out)[vec_idx2] = dst_grad_grad_out;
        src_grad_out = reinterpret_cast<const float4 *>(grad_out)[vec_idx2];
        int64_t grad_in1_base_idx = idx1[start_idx] * fea_dim + tidx * 4;
        atomicAdd(grad_in1 + grad_in1_base_idx + 0, src_grad_grad_in2.x * src_grad_out.x);
        atomicAdd(grad_in1 + grad_in1_base_idx + 1, src_grad_grad_in2.y * src_grad_out.y);
        atomicAdd(grad_in1 + grad_in1_base_idx + 2, src_grad_grad_in2.z * src_grad_out.z);
        atomicAdd(grad_in1 + grad_in1_base_idx + 3, src_grad_grad_in2.w * src_grad_out.w);
        dst_grad_in2.x = src_grad_grad_in1.x * src_grad_out.x;
        dst_grad_in2.y = src_grad_grad_in1.y * src_grad_out.y;
        dst_grad_in2.z = src_grad_grad_in1.z * src_grad_out.z;
        dst_grad_in2.w = src_grad_grad_in1.w * src_grad_out.w;
        reinterpret_cast<float4 *>(grad_in2)[vec_idx2] = dst_grad_in2;
    }
}

#endif