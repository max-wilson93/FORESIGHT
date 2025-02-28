/**
 * Kelly Criterion Optimization with CUDA Acceleration
 * 
 * This file implements a CUDA kernel for the Kelly Criterion, a formula used to
 * determine the optimal size of a series of bets. In finance, it is used for
 * position sizing to maximize long-term growth of capital.
 * 
 * Key Features:
 * - Parallelized computation of optimal position sizes.
 * - Integration with Python via CUDA C++ bindings.
 */

 #include <cuda_runtime.h>
 #include <iostream>
 
 // Kelly Criterion kernel
 __global__ void kelly_criterion(float* returns, float* variances, float* weights, int num_assets) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < num_assets) {
         // Kelly Criterion formula: f = (mean return) / (variance of return)
         weights[idx] = returns[idx] / variances[idx];
     }
 }
 
 // Wrapper function for Python integration
 extern "C" void run_kelly_criterion(float* returns, float* variances, float* weights, int num_assets) {
     float *d_returns, *d_variances, *d_weights;
     
     // Allocate GPU memory
     cudaMalloc((void**)&d_returns, num_assets * sizeof(float));
     cudaMalloc((void**)&d_variances, num_assets * sizeof(float));
     cudaMalloc((void**)&d_weights, num_assets * sizeof(float));
     
     // Copy input data to GPU
     cudaMemcpy(d_returns, returns, num_assets * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(d_variances, variances, num_assets * sizeof(float), cudaMemcpyHostToDevice);
     
     // Define block and grid sizes
     int threads = 256;
     int blocks = (num_assets + threads - 1) / threads;
     
     // Launch Kelly Criterion kernel
     kelly_criterion<<<blocks, threads>>>(d_returns, d_variances, d_weights, num_assets);
     
     // Copy results back to CPU
     cudaMemcpy(weights, d_weights, num_assets * sizeof(float), cudaMemcpyDeviceToHost);
     
     // Free GPU memory
     cudaFree(d_returns);
     cudaFree(d_variances);
     cudaFree(d_weights);
 }