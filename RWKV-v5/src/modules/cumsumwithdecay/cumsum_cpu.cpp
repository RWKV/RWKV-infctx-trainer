#include <torch/extension.h>


template <typename T_acc_x, typename T_acc_gamma>
inline
void discounted_sum_update(
        T_acc_x &acc_x, int batchsz, T_acc_gamma &gamma, bool gamma_scalar, int change_pos, int discounted_pos
) {
    for (int i=0; i<batchsz-3; i+=4) {
        acc_x[i+0][change_pos] += (gamma_scalar ? gamma[0] : gamma[i+0]) * acc_x[i+0][discounted_pos];
        acc_x[i+1][change_pos] += (gamma_scalar ? gamma[0] : gamma[i+1]) * acc_x[i+1][discounted_pos];
        acc_x[i+2][change_pos] += (gamma_scalar ? gamma[0] : gamma[i+2]) * acc_x[i+2][discounted_pos];
        acc_x[i+3][change_pos] += (gamma_scalar ? gamma[0] : gamma[i+3]) * acc_x[i+3][discounted_pos];
    }
    for (int i=(batchsz - (batchsz & 3)); i<batchsz; i++) {
        acc_x[i][change_pos] += (gamma_scalar ? gamma[0] : gamma[i]) * acc_x[i][discounted_pos];
    }
}


torch::Tensor discounted_cumsum_left_cpu(torch::Tensor x, torch::Tensor gamma) {
    TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(gamma.device().is_cpu(), "Gamma must be a CPU tensor");
    TORCH_CHECK(gamma.dim() == 1, "Gamma must be 1-dimensional");
    TORCH_CHECK(gamma.size(0) == 1 || gamma.size(0) == x.size(0), "Gamma dimensions must be compatible with the input");
    TORCH_CHECK(x.dtype() == gamma.dtype(), "Argument data types must match");

    bool gamma_scalar = gamma.dim() == 0;
    auto y = x.clone();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "discounted_cumsum_left_cpu_loop", ([&] {
        auto ya = y.accessor<scalar_t, 2>();
        auto ga = gamma.accessor<scalar_t, 1>();
        for (int j=0; j<y.size(1); j++) {
            int j_left = j-1;
            if (j_left == -1) {
                continue;
            }
            discounted_sum_update(ya, y.size(0), ga, gamma_scalar, j, j_left);
        }
    }));

    return y;
}


torch::Tensor discounted_cumsum_right_cpu(torch::Tensor x, torch::Tensor gamma) {
    TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(gamma.device().is_cpu(), "Gamma must be a CPU tensor");
    TORCH_CHECK(gamma.dim() == 1, "Gamma must be 1-dimensional");
    TORCH_CHECK(gamma.size(0) == 1 || gamma.size(0) == x.size(0), "Gamma dimensions must be compatible with the input");
    TORCH_CHECK(x.dtype() == gamma.dtype(), "Argument data types must match");

    bool gamma_scalar = gamma.dim() == 0;
    auto y = x.clone();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "discounted_cumsum_right_cpu_loop", ([&] {
        auto ya = y.accessor<scalar_t, 2>();
        auto ga = gamma.accessor<scalar_t, 1>();
        for (int j=y.size(1)-1; j>=0; j--) {
            int j_right = j+1;
            if (j_right == y.size(1)) {
                continue;
            }
            discounted_sum_update(ya, y.size(0), ga, gamma_scalar, j, j_right);
        }
    }));

    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("discounted_cumsum_left_cpu", &discounted_cumsum_left_cpu, "Discounted Cumulative Sums CPU (Left)");
    m.def("discounted_cumsum_right_cpu", &discounted_cumsum_right_cpu, "Discounted Cumulative Sums CPU (Right)");
}