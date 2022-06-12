#include <iostream>

#include "bertCommon.h"
#include "trt_tensor.hpp"
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>


int test_cublasbatched_matmul_fp32() {
    ::srand(::time(0));
    // std::cout << "test for cuda" << std::endl;
    cublasHandle_t mCublas = nullptr;

    CUBLASASSERT(cublasCreate(&mCublas));
    bert::CublasConfigHelper helper(mCublas);

    std::cout << "sm version: " << bert::getSMVersion() << std::endl;

    int n = 8;
    int k = 64;
    int m = 10;
    int b = 64;

    TRT::Tensor q_tensor(std::vector<int>{b, m, k}, TRT::DataType::Float);
    TRT::Tensor p_tensor(std::vector<int>{b, k, n}, TRT::DataType::Float);
    TRT::Tensor out_tensor(std::vector<int>{b, m, n}, TRT::DataType::Float);

    // TRT::Tensor out_tensor2(std::vector<int>{m, n});

    auto qptr_cpu = q_tensor.cpu<float>();
    auto pptr_cpu = p_tensor.cpu<float>();
    
    for(int i = 0; i < q_tensor.numel(); ++i) {
        qptr_cpu[i] = float(rand() % 100000) / 100000;
    }

    for(int i = 0; i < p_tensor.numel(); ++i) {
        pptr_cpu[i] = float(rand() % 100000) / 100000;
    }

    q_tensor.save_to_file("q_tensor.npz");
    p_tensor.save_to_file("p_tensor.npz");

    
    float* qptr_gpu = q_tensor.to_gpu(true).gpu<float>();
    float* pptr_gpu = p_tensor.to_gpu(true).gpu<float>();
    float* outptr_gpu = out_tensor.to_gpu().gpu<float>();

    // cuBLAS library uses column-major storage, and 1-based indexing.
    // A<->存储形式A.T
    // 需要计算C = A@B => 已知 A.T B.T
    // 那么C.T = B.T@A.T
    // 存储形式C.T<->C
    float alpha = 1.0;
    float beta = 0.0;

    // the LDA is used to define the distance in memory between elements of two consecutive columns which have the same row index
    
    // B [k,n] B.T [n,k]
    // A [m,k] A.T [k,m]
    // C [m,n] C.T [n,m]
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t qType = CUDA_R_32F;
    cudaDataType_t pType = CUDA_R_32F;
    cudaDataType_t oType = CUDA_R_32F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    TRT::Tensor pptr_gpu_arr_tensor(std::vector<int>{b}, TRT::DataType::Ptr);
    TRT::Tensor qptr_gpu_arr_tensor(std::vector<int>{b}, TRT::DataType::Ptr);
    TRT::Tensor outptr_gpu_arr_tensor(std::vector<int>{b}, TRT::DataType::Ptr);


    auto pptr_cpu_arr = pptr_gpu_arr_tensor.cpu<float*>();
    auto qptr_cpu_arr = qptr_gpu_arr_tensor.cpu<float*>();
    auto outptr_cpu_arr = outptr_gpu_arr_tensor.cpu<float*>();

    // 注意：这里的指针数组，空间可以不是连续的
    for(int i = 0; i < b; ++i) {
        pptr_cpu_arr[i] = pptr_gpu + i * p_tensor.size(1) * p_tensor.size(2);
    }

    for(int i = 0; i < b; ++i) {
        qptr_cpu_arr[i] = qptr_gpu + i * q_tensor.size(1) * q_tensor.size(2);
    }

    for(int i = 0; i < b; ++i) {
        outptr_cpu_arr[i] = outptr_gpu + i * out_tensor.size(1) * out_tensor.size(2);
    }
    
    // B [k,n] B.T [n,k]
    // A [m,k] A.T [k,m]
    // C [m,n] C.T [n,m]
    CUBLASASSERT(cublasGemmBatchedEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                        &alpha, 
                        (void**)pptr_gpu_arr_tensor.to_gpu(true).gpu<float*>(), pType, n,
                        (void**)qptr_gpu_arr_tensor.to_gpu(true).gpu<float*>(), qType, k,
                        &beta,
                        (void**)outptr_gpu_arr_tensor.to_gpu().gpu<float*>(), oType, n,
                        b, computeType, algo
                        ));

    out_tensor.to_cpu(true);
    out_tensor.save_to_file("out_tensor.npz");
    
    CUBLASASSERT(cublasDestroy(mCublas));
    return 0;
}

int main() {

    test_cublasbatched_matmul_fp32();

    return 0;
}