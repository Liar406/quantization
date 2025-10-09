import torch

# 创建一个随机矩阵 W
W = torch.randn(3, 3)  # 例如一个 3x3 的矩阵

# 计算 Frobenius 范数，通过矩阵元素
frobenius_norm_from_elements = torch.norm(W, p='fro')

# 计算 Frobenius 范数，通过奇异值
singular_values = torch.linalg.svdvals(W)
frobenius_norm_from_singular_values = torch.sqrt(torch.sum(singular_values ** 2))

# 输出结果
print("Frobenius norm from matrix elements:", frobenius_norm_from_elements)
print("Frobenius norm from singular values:", frobenius_norm_from_singular_values)
