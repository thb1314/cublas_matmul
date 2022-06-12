# MatMul in CUBLAS

## cublas function list

- cublasSaxpy_v2
- cublasGemmEx https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx
- cublasGemmBatchedEx https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmBatchedEx
- cublasGemmStridedBatchedEx https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmStridedBatchedEx

## implemented list

- A[m,k] + B[m,k] 01cublas_test_matadd
- A[m,k] @ B[k,n] 02cublas_test_matmul
- A[b,m,k] @ B[b,k,n] 03cublas_test_batched_matmul
- A[b,m,k] @ B[1,k,n] 04cublas_test_broadcast01_matmul
- A[b,s,m,k] @ B[1,s,k,n],A[b,s,m,k] @ B[b,1,k,n] 05cublas_test_broadcast02_matmul

## Usage

Modify the `CUDA_PATH` `SM` `CUDA_HOME` in `Makefile`, then execute `make` in bash.