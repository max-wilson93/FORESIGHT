/**
 * CUDA-Optimized Tensor Operations
 * 
 * This header file provides utility functions for tensor operations, optimized
 * for GPU performance. These functions are used across various CUDA kernels.
 * 
 * Key Features:
 * - Tensor addition, subtraction, and multiplication.
 * - Tensor reshaping and slicing.
 * - Element-wise operations.
 */

 #ifndef TENSOR_OPS_CUH
 #define TENSOR_OPS_CUH
 
 #include <cuda_runtime.h>
 
 // Tensor addition kernel
 __global__ void tensor_add(float* A, float* B, float* C, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) {
         C[idx] = A[idx] + B[idx];
     }
 }
 
 // Tensor multiplication kernel
 __global__ void tensor_mul(float* A, float* B, float* C, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) {
         C[idx] = A[idx] * B[idx];
     }
 }
 
 // Tensor reshape kernel
 __global__ void tensor_reshape(float* input, float* output, int input_size, int output_size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < output_size) {
         output[idx] = input[idx % input_size];  // Example: Simple cyclic reshape
     }
 }
 
 #endif // TENSOR_OPS_CUH