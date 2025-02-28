/**
 * CUDA-Optimized Matrix Operations
 * 
 * This header file provides utility functions for matrix operations, optimized
 * for GPU performance. These functions are used across various CUDA kernels.
 * 
 * Key Features:
 * - Matrix multiplication.
 * - Matrix transpose.
 * - Element-wise operations.
 */

 #ifndef MATRIX_OPS_CUH
 #define MATRIX_OPS_CUH
 
 #include <cuda_runtime.h>
 
 // Matrix multiplication kernel
 __global__ void matmul(float* A, float* B, float* C, int m, int n, int p) {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row < m && col < p) {
         float sum = 0.0f;
         for (int k = 0; k < n; k++) {
             sum += A[row * n + k] * B[k * p + col];
         }
         C[row * p + col] = sum;
     }
 }
 
 #endif // MATRIX_OPS_CUH