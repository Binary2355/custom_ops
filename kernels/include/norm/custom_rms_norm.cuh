#ifndef CUSTOM_CUDA_CUSTOM_RMSNORM
#define CUSTOM_CUDA_CUSTOM_RMSNORM

#include <float.h>

#include "device_utils.cuh"

#include <iostream>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include <unordered_map>

inline void swich_head_dim(int head_dim, size_t& HEAD_DIM) {
    // 使用静态数组实现查找表
    static const std::pair<int, size_t> head_dim_table[] = {
        {32, 32},
        {64, 64},
        {128, 128},
        {256, 256}
    };

    // 线性搜索
    const int table_size = sizeof(head_dim_table) / sizeof(head_dim_table[0]);
    for (int i = 0; i < table_size; ++i) {
        if (head_dim_table[i].first == head_dim) {
            HEAD_DIM = head_dim_table[i].second;
            return;
        }
    }

    throw std::invalid_argument("Unsupported head_dim: " + std::to_string(head_dim));
}

// 定义配置结构体，打包相关参数
struct NumHeadConfig {
    int threads_num;
    int warps_num;
    int loops_factor; // 用于计算loops_num
};

inline void switch_num_head(int num_heads, int& threads_num, int& warps_num, int& loops_num) {
    if(num_heads > 40) {
        throw std::invalid_argument("Unsupported num_heads: " + std::to_string(num_heads));
    }

    static constexpr NumHeadConfig configs[] = {
        {256, 8, 3}, // 对应 (num_heads & 7) == 0，右移3位
        {128, 4, 2}, // 对应 (num_heads & 3) == 0，右移2位
        {64,  2, 1}, // 对应 (num_heads & 1) == 0，右移1位
        {32,  1, 0}  // 默认情况
    };

    int config_idx = 3; // 默认使用最后一个配置
    if ((num_heads & 7) == 0) config_idx = 0;
    else if ((num_heads & 3) == 0) config_idx = 1;
    else if ((num_heads & 1) == 0) config_idx = 2;

    // 获取配置
    const NumHeadConfig& config = configs[config_idx];
    threads_num = config.threads_num;
    warps_num = config.warps_num;

    // 计算loops_num
    loops_num = (config.loops_factor > 0) ? (num_heads >> config.loops_factor) : num_heads;

    bool should_adjust = (loops_num * threads_num <= 1024) && (warps_num * loops_num <= 16);

    int original_loops_num = loops_num;
    loops_num = should_adjust ? 1 : loops_num;
    threads_num = should_adjust ? (threads_num * original_loops_num) : threads_num;
    warps_num = should_adjust ? (warps_num * original_loops_num) : warps_num;
}

template <typename T, int LOOPS>
__global__ void rmsnorm_custom_n_subwarp_reduction(T* data, T* gamma, float epsilon, const int m, const int n,
    int bdx, int bdy, int warps_num) {
    /*
        说明：
          - 子线程块：每个线程块会依据HEAD_DIM被分成多个子线程块，每个子线程块独立处理data里不同行的数据
        参数说明：
          - bdx: 每个子线程块的线程数量
          - bdy: 每个线程块的子线程块数量
          - warps_num: 每个线程块的warp数量
        模版参数说明：
          - LOOPS: 处理每一行需要的线程块数量
    */
    // 使用共享内存作为同步点
    extern __shared__ float shared_mem[];
    float* block_results = shared_mem;
    // 获取一次搬运T类型的数据量
    const int NUM_T_VECTORIZED = sizeof(float4) / sizeof(T);

    // 获取当前线程正在处理的行ID, 范围为[0, m)
    const int cur_m = blockIdx.x * bdy + threadIdx.y;

    // 设定中间变量计算平方和
    float cur_m_sum = 0.0f;
    float cur_m_rsqrt = 0.0f;

    // 获取当前线程在当前线程块中的线程id
    const int tid = threadIdx.x;
    // 获取data的指针
    T* data_ptr = data + cur_m * n;
    // 存储data读取的数据
    float4 local_val[1024][1];

#pragma unroll
    for (int cur_loop = 0; cur_loop < LOOPS; cur_loop += 1) {
        float warp_sum = 0.0f;
        float block_sum = 0.0f;
        // 获取data的指针
        T* input = data_ptr + cur_loop * bdx;
        const float4* vectorized_input = reinterpret_cast<const float4*>(input);

        // 读取数据
        local_val[cur_loop][0] = cur_m < m ? vectorized_input[tid] : local_val[cur_loop][0];
        // 解压float4数据
        T* extracted_local_val = reinterpret_cast<T*>(local_val[cur_loop]);

#pragma unroll
        for(int i = 0; i < NUM_T_VECTORIZED; i += 1) {
            warp_sum += (static_cast<float>(extracted_local_val[i])) *
                        (static_cast<float>(extracted_local_val[i]));
        }
        // 第一阶段:warp reduce
#pragma unroll
        int logical_warp_num = 32 / bdy;
        for(int i = (logical_warp_num / 2); i > 0; i >>= 1) {
            warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, i);
        }
        const int warp_id = tid / logical_warp_num;
        const int lane_id = tid & (logical_warp_num - 1);
        if (lane_id == 0) {
            block_results[warp_id] = warp_sum;
        }
        __syncthreads();
        // 第二阶段:block reduce
        if (warp_id == 0) {
            if (lane_id < warps_num) {
                block_sum = block_results[lane_id];
            } else {
                block_sum = 0.0f;
            }
#pragma unroll
            for (int i = 16; i > 0; i >>= 1) {
                block_sum += __shfl_xor_sync(0xffffffff, block_sum, i);
            }
            // 将结果写入共享内存；因此tid==0的线程会读取到block reduce的结果
            block_results[0] = block_sum;
        }
        __syncthreads();
        if (tid == 0) {
            cur_m_sum += block_results[0];
        }
    }
    // 计算最终的RMS归一化值
    if (tid == 0) {
        block_results[0] = rsqrtf(cur_m_sum / n + epsilon);
    }
    __syncthreads();
    cur_m_rsqrt = block_results[0];

    // 计算最终的输出
#pragma unroll
    for (int cur_loop = 0; cur_loop < LOOPS; cur_loop += 1) {
        // 获取input,weight和output的指针
        T* weight = gamma + cur_loop * bdx;
        T* output = data_ptr + cur_loop * bdx;
        const float4* vectorized_gamma = reinterpret_cast<const float4*>(weight);
        float4* vectorized_output = reinterpret_cast<float4*>(output);

        // 读取数据
        float4 local_gamma[1] = {vectorized_gamma[tid]};
        // 解压float4数据
        T* extracted_local_val = reinterpret_cast<T*>(local_val[cur_loop]);
        T* extracted_local_gamma = reinterpret_cast<T*>(local_gamma);
#pragma unroll
        for(int i = 0; i < NUM_T_VECTORIZED; i += 1) {
            float tmp = (static_cast<float>(extracted_local_val[i])) * cur_m_rsqrt *
                            static_cast<float>(extracted_local_gamma[i]);
            extracted_local_val[i] = static_cast<T>(tmp);
        }
        if (cur_m < m) {
            vectorized_output[tid] = local_val[cur_loop][0];
        }
    }
}


template <typename T>
void custom_rmsnorm_inplace(
                      int row,
                      int column,
                      T* data,
                      T* gamma,
                      int head_dim,
                      float epsilon,
                      cudaStream_t stream) {
    const int m = row;
    const int n = column;
    const int NUM_T_VECTORIZED = sizeof(float4) / sizeof(T);
    size_t HEAD_DIM;
    swich_head_dim(head_dim, HEAD_DIM);
    const int DIM = n;
    const int num_heads = DIM / HEAD_DIM;
    int threads_num, warps_num, loops_num;
    switch_num_head(num_heads, threads_num, warps_num, loops_num);
    const int bdy = 32 / (HEAD_DIM / NUM_T_VECTORIZED);
    const int bdx = threads_num / bdy;
    dim3 block(bdx, bdy);
    dim3 grid((m + bdy - 1) / bdy);
    if (loops_num == 1) {
        rmsnorm_custom_n_subwarp_reduction<T, 1>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 2) {
        rmsnorm_custom_n_subwarp_reduction<T, 2>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 3) {
        rmsnorm_custom_n_subwarp_reduction<T, 3>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 4) {
        rmsnorm_custom_n_subwarp_reduction<T, 4>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 5) {
        rmsnorm_custom_n_subwarp_reduction<T, 5>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 6) {
        rmsnorm_custom_n_subwarp_reduction<T, 6>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 7) {
        rmsnorm_custom_n_subwarp_reduction<T, 7>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 8) {
        rmsnorm_custom_n_subwarp_reduction<T, 8>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 9) {
        rmsnorm_custom_n_subwarp_reduction<T, 9>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else if (loops_num == 10) {
        rmsnorm_custom_n_subwarp_reduction<T, 10>
            <<<grid, block, bdy * warps_num * sizeof(float), stream>>>(data, gamma, epsilon, m, n, bdx, bdy, warps_num);
    } else {
        throw std::invalid_argument("Unsupported loops_num: " + std::to_string(loops_num));
    }
}

#endif // CUSTOM_CUDA_CUSTOM_RMSNORM