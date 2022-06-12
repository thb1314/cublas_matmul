#include <iostream>

#include "bertCommon.h"
#include "trt_tensor.hpp"
#include <ctime>
#include <cstdlib>
#include <cassert>


int test_cublasmatmul_fp32() {
    ::srand(::time(0));
    std::cout << ::rand() << std::endl;
    // std::cout << "test for cuda" << std::endl;
    cublasHandle_t mCublas;

    int n = 8;
    int k = 63;
    int m = 64;

    TRT::Tensor q_tensor(std::vector<int>{m, k});
    TRT::Tensor p_tensor(std::vector<int>{k, n});
    TRT::Tensor out_tensor(std::vector<int>{m, n});
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

    CUBLASASSERT(cublasCreate(&mCublas));
    bert::CublasConfigHelper helper(mCublas);

    auto qptr_gpu = q_tensor.to_gpu(true).gpu<float>();
    auto pptr_gpu = p_tensor.to_gpu(true).gpu<float>();
    auto outptr_gpu = out_tensor.to_gpu().gpu<float>();

    // cuBLAS library uses column-major storage, and 1-based indexing.
    // A<->存储形式A.T
    // 需要计算C = A@B => 已知A.T B.T
    // 那么C.T = B.T@A.T
    // 存储形式C.T<->A.T
    float alpha = 1.0;
    float beta = 0.0;
    
    // the LDA is used to define the distance in memory between elements of two consecutive columns which have the same row index
    
    // B [k,n] B.T [n,k]
    // A [m,k] A.T [k,m]
    // C [m,n] C.T [n,m]
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType_t qType = CUDA_R_32F;
    cudaDataType_t pType = CUDA_R_32F;
    cudaDataType_t oType = CUDA_R_32F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    cublasGemmEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, 
                pptr_gpu, pType, n,
                qptr_gpu, qType, k,
                &beta,
                outptr_gpu, oType, n,
                computeType, algo
                );

    out_tensor.to_cpu(true);
    out_tensor.save_to_file("out_tensor.npz");
    
    CUBLASASSERT(cublasDestroy(mCublas));
    return 0;
}

int main() {

    test_cublasmatmul_fp32();

    return 0;
}