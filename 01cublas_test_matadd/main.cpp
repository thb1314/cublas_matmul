#include <iostream>

#include "bertCommon.h"
#include "trt_tensor.hpp"
#include <ctime>
#include <cstdlib>
#include <cassert>


int test_cublasadd_fp32() {
    ::srand(::time(0));
    std::cout << ::rand() << std::endl;
    // std::cout << "test for cuda" << std::endl;
    cublasHandle_t mCublas;

    int batch_size = 8;
    int seq_len = 63;
    int hidden_dim = 64;
    int num_head = 4;
    int mS = seq_len;
    int mHeadSize = hidden_dim;

    TRT::Tensor q_tensor(std::vector<int>{seq_len, batch_size, num_head, hidden_dim});
    TRT::Tensor p_tensor(std::vector<int>{seq_len, batch_size, num_head, hidden_dim});
    TRT::Tensor out_tensor(std::vector<int>{seq_len, batch_size, num_head, hidden_dim});

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
    auto outptr_gpu = out_tensor.to_gpu().copy_from_gpu(0, pptr_gpu, p_tensor.numel()).gpu<float>();

    int mNumHeads = num_head;
    int mLdQKV = 4 * batch_size * mNumHeads * mHeadSize;
    int mStrideQKV = 4 * mHeadSize;
    int mLdOut = batch_size * mNumHeads * mHeadSize;
    int mStrideOut = mHeadSize;
    int mOmatSize = mS * mS;
    int mNumMats = batch_size * mNumHeads;

    float alpha = 1.0;
    // output_tensor := qptr_gpu + outptr_gpu
    CUBLASASSERT(cublasSaxpy_v2(mCublas, out_tensor.numel(), &alpha, qptr_gpu, 1, outptr_gpu, 1)); //实现向量+

    out_tensor.to_cpu(true);
    out_tensor.save_to_file("out_tensor.npz");
    
    CUBLASASSERT(cublasDestroy(mCublas));
    return 0;
}

int main() {

    test_cublasadd_fp32();

    return 0;
}