#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("voxelize_idx", &voxelize_idx_3d, "voxelize_idx");
    m.def("voxelize_fp", &voxelize_fp_feat, "voxelize_fp");
    m.def("voxelize_bp", &voxelize_bp_feat, "voxelize_bp");
}