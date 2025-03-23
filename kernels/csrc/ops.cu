#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(_kernels, m) {
    m.def("custom_rms_norm_forward", &custom_rmsnorm_forward, "Custom RMS norm with bias and learned weight.");
}