/**
 * Kelly Criterion Optimization with CUDA Acceleration
 * 
 * Purpose:
 * This file implements the Kelly Criterion, a formula used to determine the
 * optimal size of a series of bets. In finance, it is used for position sizing
 * to maximize long-term growth of capital. The CUDA implementation ensures
 * high-performance computation, especially for large portfolios.
 * 
 * Role in FORESIGHT:
 * - Provides GPU-accelerated position sizing for portfolio optimization.
 * - Integrates with the broader portfolio management system to enhance
 *   decision-making in quantitative trading.
 * - Works alongside other optimization algorithms (e.g., Markowitz) to
 *   provide a comprehensive suite of tools for portfolio management.
 */

 #include <cuda_runtime.h>
 #include <iostream>
 
 /**
  * CUDA Kernel for Kelly Criterion Optimization
  * 
  * @param returns: Array of expected returns for each asset.
  * @param variances: Array of variances (risk) for each asset.
  * @param weights: Output array for optimal position sizes.
  * @param num_assets: Number of assets in the portfolio.
  */
 __global__ void kelly_kernel(float* returns, float* variances, float* weights, int num_assets) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < num_assets) {
         // Kelly Criterion formula: f = (mean return) / (variance of return)
         weights[idx] = returns[idx] / variances[idx];
     }
 }
 
 /**
  * Wrapper Function for Python Integration
  * 
  * @param returns: Array of expected returns (host memory).
  * @param variances: Array of variances (host memory).
  * @param weights: Output array for optimal position sizes (host memory).
  * @param num_assets: Number of assets in the portfolio.
  */
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
     kelly_kernel<<<blocks, threads>>>(d_returns, d_variances, d_weights, num_assets);
     
     // Copy results back to CPU
     cudaMemcpy(weights, d_weights, num_assets * sizeof(float), cudaMemcpyDeviceToHost);
     
     // Free GPU memory
     cudaFree(d_returns);
     cudaFree(d_variances);
     cudaFree(d_weights);
 }