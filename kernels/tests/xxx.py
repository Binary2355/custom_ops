import torch
import sys
sys.path.append("/root/code/zekali/sparsity/vefuser/optimization/kernels/build")
import _kernels


class TestRMSNorm(torch.nn.Module):

    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self.num_heads * self.head_dim))

    def forward(self, x):
        origin_shape = x.shape
        input = x.view(x.shape[0] * x.shape[1], self.num_heads * self.head_dim).clone()
        result = torch.nn.functional.rms_norm(input, [input.size(-1)], self.weight, self.eps)
        return result.view(origin_shape)



class CustomRMSNorm(torch.nn.Module):
    def __init__(self, dim, head_dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x)

    def _norm(self, x):
        bsz, seq_len, dim = x.shape
        x_clone = x.view(bsz * seq_len, dim).clone()
        print(f"636998 CustomRMSNorm weight: {self.weight}")
        _kernels.custom_rms_norm_forward(x_clone, self.weight, self.head_dim, self.eps)
        return x_clone.view(bsz, seq_len, dim)


num_heads=12
head_dim=128
dim=num_heads*head_dim
eps=1e-6

my_rms = CustomRMSNorm(dim=dim, head_dim=head_dim, eps=eps).cuda()
test_rms = TestRMSNorm(dim=dim, num_heads=num_heads, eps=eps).cuda()

q_ = torch.nn.Linear(dim, dim, dtype = torch.float32).cuda()


x = torch.randn(1, 16850, dim, dtype = torch.float32).cuda()
print(f"x : {x}")
b, s, n, d = *x.shape[:2], num_heads, head_dim

q_myrms = my_rms(q_(x)).view(b, s, n, d)
q_testrms = test_rms(q_(x)).view(b, s, n, d)

# print("******************* myrms ****************")
# print(q_myrms)
# print("******************* testrms ****************")
# print(q_testrms)
from compare import compare_tensors
compare_tensors(q_myrms, q_testrms, name_a="custom_rmsnorm", name_b="torch_rmsnorm", max_print=1000000, abs_threshold=1e-3, verbose=True, return_stats=True)





