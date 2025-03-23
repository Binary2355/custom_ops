#ifndef CUSTOM_CUDA_OPS_KERNEL
#define CUSTOM_CUDA_OPS_KERNEL

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/extension.h>

#include "norm/custom_rms_norm.cuh"
#include "pytorch_extension_utils.h"

/*
    input: [m, n] Row-major; assume n is reduce dim
    output: [m, n] Row-major
    gemma: [n]
    beta: [n]
*/
void custom_rmsnorm_forward(
                     torch::Tensor input,
                     torch::Tensor gemma,
                     int head_dim,
                     float epsilon = 1e-5) {
    CHECK_INPUT(input);
    CHECK_INPUT(gemma);

    CHECK_EQ(input.dim(), 2);
    CHECK_EQ(gemma.dim(), 1);
    CHECK_EQ(input.size(1), gemma.size(0));

    bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input.scalar_type(), c_type, [&] {
        DEBUG_CUDA_CALL(
            custom_rmsnorm_inplace<c_type>(input.size(0), input.size(1),
                                static_cast<c_type*>(input.data_ptr()),
                                static_cast<c_type*>(gemma.data_ptr()),
                                head_dim,
                                epsilon,
                                at::cuda::getCurrentCUDAStream()));
        return true;
    });
    TORCH_CHECK(success, "Customized call failed");
}

#endif // CUSTOM_CUDA_OPS_KERNEL