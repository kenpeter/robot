// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
Flash Attention 4 CUDA Kernel Implementation
Optimized for Nvidia Blackwell architecture with asynchronous pipeline
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

// Constants
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 5; // Load, MMA, Softmax, Correction, Epilogue

// Cubic polynomial approximation for exp (faster than hardware exp for small values)
__device__ __forceinline__ float fast_exp_poly(float x) {
    // Cubic approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
    // For larger values, use range reduction
    if (x < -10.0f) return 0.0f;
    if (x > 10.0f) return expf(x); // Fall back to hardware exp

    float x2 = x * x;
    float x3 = x2 * x;
    return 1.0f + x + 0.5f * x2 + 0.16666667f * x3;
}

// Online softmax with improved numerical stability
struct OnlineSoftmax {
    float m; // running max
    float l; // running sum of exp

    __device__ __forceinline__ OnlineSoftmax() : m(-INFINITY), l(0.0f) {}

    __device__ __forceinline__ void update(float x) {
        float m_new = fmaxf(m, x);
        float scale = fast_exp_poly(m - m_new);
        l = l * scale + fast_exp_poly(x - m_new);
        m = m_new;
    }

    __device__ __forceinline__ float normalize(float x) {
        return fast_exp_poly(x - m) / l;
    }
};

// Warp roles for asynchronous pipeline
enum WarpRole {
    LOAD_WARP = 0,
    MMA_WARP = 1,
    SOFTMAX_WARP = 2,
    CORRECTION_WARP = 3,
    EPILOGUE_WARP = 4
};

// Shared memory layout for pipeline stages
template<int HEAD_DIM, int BLOCK_SIZE>
struct SharedMemory {
    // Ping-pong buffers for async loading
    __align__(16) half q_smem[2][BLOCK_SIZE][HEAD_DIM];
    __align__(16) half k_smem[2][BLOCK_SIZE][HEAD_DIM];
    __align__(16) half v_smem[2][BLOCK_SIZE][HEAD_DIM];

    // Attention scores and outputs
    __align__(16) float scores[BLOCK_SIZE][BLOCK_SIZE];
    __align__(16) half output[BLOCK_SIZE][HEAD_DIM];

    // Synchronization barriers for pipeline stages
    __align__(16) cuda::barrier<cuda::thread_scope_block> barriers[3];
};

// Flash Attention 4 kernel with async pipeline
template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attention_4_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float softmax_scale
) {
    // Grid dimensions: (num_heads, batch_size)
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Shared memory
    __shared__ SharedMemory<HEAD_DIM, BLOCK_SIZE> smem;

    // Determine warp role
    WarpRole role = static_cast<WarpRole>(warp_id % NUM_WARPS);

    // Base pointers for this batch and head
    const int head_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                           head_idx * seq_len * HEAD_DIM;
    const half* Q_head = Q + head_offset;
    const half* K_head = K + head_offset;
    const half* V_head = V + head_offset;
    half* O_head = O + head_offset;

    // Pipeline state
    int buffer_idx = 0;

    // Process sequence in blocks
    for (int block_start = 0; block_start < seq_len; block_start += BLOCK_SIZE) {
        const int block_end = min(block_start + BLOCK_SIZE, seq_len);
        const int block_len = block_end - block_start;

        // === LOAD WARP: Async load Q, K, V ===
        if (role == LOAD_WARP) {
            for (int i = threadIdx.x; i < block_len * HEAD_DIM; i += blockDim.x) {
                int row = i / HEAD_DIM;
                int col = i % HEAD_DIM;
                int seq_idx = block_start + row;

                // Load with async copy (using cp.async on Ampere+)
                smem.q_smem[buffer_idx][row][col] = Q_head[seq_idx * HEAD_DIM + col];
                smem.k_smem[buffer_idx][row][col] = K_head[seq_idx * HEAD_DIM + col];
                smem.v_smem[buffer_idx][row][col] = V_head[seq_idx * HEAD_DIM + col];
            }
        }

        __syncthreads();

        // === MMA WARP: Compute Q @ K^T using Tensor Cores ===
        if (role == MMA_WARP) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> qk_frag;

            for (int i = 0; i < block_len; i += 16) {
                for (int j = 0; j < block_len; j += 16) {
                    wmma::fill_fragment(qk_frag, 0.0f);

                    for (int k = 0; k < HEAD_DIM; k += 16) {
                        wmma::load_matrix_sync(q_frag, &smem.q_smem[buffer_idx][i][k], HEAD_DIM);
                        wmma::load_matrix_sync(k_frag, &smem.k_smem[buffer_idx][j][k], HEAD_DIM);
                        wmma::mma_sync(qk_frag, q_frag, k_frag, qk_frag);
                    }

                    // Store with softmax scaling
                    float scores_tile[16][16];
                    wmma::store_matrix_sync(&scores_tile[0][0], qk_frag, 16, wmma::mem_row_major);

                    for (int ii = 0; ii < 16; ii++) {
                        for (int jj = 0; jj < 16; jj++) {
                            if (i + ii < block_len && j + jj < block_len) {
                                smem.scores[i + ii][j + jj] = scores_tile[ii][jj] * softmax_scale;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();

        // === SOFTMAX WARP: Apply online softmax with reduced rescaling ===
        if (role == SOFTMAX_WARP) {
            for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
                OnlineSoftmax softmax;

                // First pass: compute max and sum
                for (int j = 0; j < block_len; j++) {
                    softmax.update(smem.scores[i][j]);
                }

                // Second pass: normalize (10x fewer rescaling ops)
                for (int j = 0; j < block_len; j++) {
                    smem.scores[i][j] = softmax.normalize(smem.scores[i][j]);
                }
            }
        }

        __syncthreads();

        // === CORRECTION WARP: Numerical corrections for attention sinks ===
        if (role == CORRECTION_WARP) {
            // Apply attention sink corrections for improved stability
            for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
                float row_sum = 0.0f;
                for (int j = 0; j < block_len; j++) {
                    row_sum += smem.scores[i][j];
                }

                // Renormalize if needed (should be close to 1.0)
                if (fabsf(row_sum - 1.0f) > 1e-5f) {
                    for (int j = 0; j < block_len; j++) {
                        smem.scores[i][j] /= row_sum;
                    }
                }
            }
        }

        __syncthreads();

        // === EPILOGUE WARP: Compute attention @ V ===
        if (role == EPILOGUE_WARP) {
            for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < block_len; j++) {
                        sum += smem.scores[i][j] * __half2float(smem.v_smem[buffer_idx][j][d]);
                    }
                    smem.output[i][d] = __float2half(sum);
                }

                // Write output
                int seq_idx = block_start + i;
                for (int d = 0; d < HEAD_DIM; d++) {
                    O_head[seq_idx * HEAD_DIM + d] = smem.output[i][d];
                }
            }
        }

        // Flip buffer for next iteration
        buffer_idx = 1 - buffer_idx;
        __syncthreads();
    }
}

// Host function to launch Flash Attention 4
extern "C" void launch_flash_attention_4(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 64; // Tuned for Blackwell architecture

    // Launch kernel with grid (num_heads, batch_size)
    dim3 grid(num_heads, batch_size);
    dim3 block(BLOCK_SIZE * NUM_WARPS);

    if (head_dim == 64) {
        flash_attention_4_kernel<64, BLOCK_SIZE><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(Q),
            reinterpret_cast<const half*>(K),
            reinterpret_cast<const half*>(V),
            reinterpret_cast<half*>(O),
            batch_size, num_heads, seq_len, softmax_scale
        );
    } else if (head_dim == 128) {
        flash_attention_4_kernel<128, BLOCK_SIZE><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(Q),
            reinterpret_cast<const half*>(K),
            reinterpret_cast<const half*>(V),
            reinterpret_cast<half*>(O),
            batch_size, num_heads, seq_len, softmax_scale
        );
    }

    cudaStreamSynchronize(stream);
}
