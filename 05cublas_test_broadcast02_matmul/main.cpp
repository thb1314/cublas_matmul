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
    int s = 10;

    TRT::Tensor q_tensor(std::vector<int>{b, s, m, k}, TRT::DataType::Float);
    TRT::Tensor p_tensor(std::vector<int>{b, s, k, n}, TRT::DataType::Float);
    TRT::Tensor out_tensor1(std::vector<int>{b, s, m, n}, TRT::DataType::Float);
    TRT::Tensor out_tensor2(std::vector<int>{b, s, m, n}, TRT::DataType::Float);
    TRT::Tensor out_tensor3(std::vector<int>{b, s, m, n}, TRT::DataType::Float);

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
    float* outptr_gpu1 = out_tensor1.to_gpu().gpu<float>();
    float* outptr_gpu2 = out_tensor2.to_gpu().gpu<float>();
    float* outptr_gpu3 = out_tensor3.to_gpu().gpu<float>();

    // cuBLAS library uses column-major storage, and 1-based indexing.
    // A<->存储形式A.T
    // 需要计算C = A@B => 已知 A.T B.T
    // 那么C.T = B.T@A.T
    // 存储形式C.T<->C
    

    // the LDA is used to define the distance in memory between elements of two consecutive columns which have the same row index
    
    // B [k,n] B.T [n,k]
    // A [m,k] A.T [k,m]
    // C [m,n] C.T [n,m]
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t qType = CUDA_R_32F;
    cudaDataType_t pType = CUDA_R_32F;
    cudaDataType_t oType = CUDA_R_32F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    /*
    cublasGemmStridedBatchedEx(cublasHandle_t handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m,
                            int n,
                            int k,
                            const void    *alpha,
                            const void     *A,
                            cudaDataType_t Atype,
                            int lda,
                            long long int strideA,
                            const void     *B,
                            cudaDataType_t Btype,
                            int ldb,
                            long long int strideB,
                            const void    *beta,
                            void           *C,
                            cudaDataType_t Ctype,
                            int ldc,
                            long long int strideC,
                            int batchCount,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo)
    */
    // strideA: value of type long long int that gives the offset in number of elements between A[i] and A[i+1].
    // B [k,n] B.T [n,k]
    // A [m,k] A.T [k,m]
    // C [m,n] C.T [n,m]
    float alpha = 1.0;
    float beta = 0.0;
    
    CUBLASASSERT(cublasGemmStridedBatchedEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                        &alpha,
                        pptr_gpu, pType, n, p_tensor.size(2) * p_tensor.size(3),
                        qptr_gpu, qType, k, q_tensor.size(2) * q_tensor.size(3),
                        &beta,
                        outptr_gpu1, oType, n, out_tensor1.size(2) * out_tensor1.size(3),
                        b * s, computeType, algo
                        ));
    
    // Q[0:1] @ P， Q[0:1] 会广播到 P 的 batch
    // 分析： Q[0:1] 1,s,m,k
    // 分析： P b,s,k,n
    // 需要： O b,s,m,n

    // 需要转换为 多次 [1,1,m,k] 与 1,s,k,n
    // 然后 output采用不同位置

    // O[0, s, :m, :k] = Q[0, s, :m, :k] @ P[0, s, :k, :n]
    // O[1, s, :m, :k] = Q[0, s, :m, :k] @ P[1, s, :k, :n]
    for(int i = 0; i < b; ++i) {
        CUBLASASSERT(cublasGemmStridedBatchedEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                        &alpha,
                        pptr_gpu + i * p_tensor.size(1) * p_tensor.size(2) * p_tensor.size(3), pType, n, p_tensor.size(2) * p_tensor.size(3),
                        qptr_gpu, qType, k, q_tensor.size(2) * q_tensor.size(3),
                        &beta,
                        outptr_gpu2 + i * out_tensor2.size(1) * out_tensor2.size(2) * out_tensor2.size(3), oType, n, out_tensor2.size(2) * out_tensor2.size(3),
                        s, computeType, algo
                        ));
    }


    // Q[:, 0:1, :, :] @ P， Q[:, 0:1, :, :] 会广播到 P 的 s
    // 分析： Q[:, 0:1, :, :] b,1,m,k
    // 分析： P b, s, k, n
    // 需要： O b, s, m, n

    // 需要转换为 多次 [b,1,m,k] 与 b,1,k,n
    // 然后 output 采用不同位置
    // O[b, 0, :m, :k] = Q[b, 0, :m, :k] @ P[b, 0, :k, :n]
    // O[b, 1, :m, :k] = Q[b, 0, :m, :k] @ P[b, 1, :k, :n]
    // O[b, 2, :m, :k] = Q[b, 0, :m, :k] @ P[b, 2, :k, :n] 
    // 每一次的stride为后三维的维度

    for(int i = 0; i < s; ++i) {
        CUBLASASSERT(cublasGemmStridedBatchedEx(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                        &alpha,
                        pptr_gpu + (0 * p_tensor.size(1) +  i) * p_tensor.size(2) * p_tensor.size(3), pType, n, p_tensor.size(1) * p_tensor.size(2) * p_tensor.size(3),
                        qptr_gpu, qType, k, q_tensor.size(1) * q_tensor.size(2) * q_tensor.size(3),
                        &beta,
                        outptr_gpu3 + (0 * out_tensor3.size(1) +  i) * out_tensor3.size(2) * out_tensor3.size(3), oType, n, out_tensor3.size(1) * out_tensor3.size(2) * out_tensor3.size(3),
                        b, computeType, algo
                        ));
    }
    

    out_tensor1.to_cpu(true);
    out_tensor1.save_to_file("out_tensor1.npz");

    out_tensor2.to_cpu(true);
    out_tensor2.save_to_file("out_tensor2.npz");

    out_tensor3.to_cpu(true);
    out_tensor3.save_to_file("out_tensor3.npz");
    
    CUBLASASSERT(cublasDestroy(mCublas));
    return 0;
}

int main() {

    test_cublasbatched_matmul_fp32();

    return 0;
}