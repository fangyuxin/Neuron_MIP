import numpy as np
import torch
#
#
# n_class = 3
#
# label_true = np.array([0, 2, 1, 100])
#
# label_pred = np.array([0.1, 1.9, 1.3, 2.2])
#
# mask = (label_true >= 0) & (label_true < n_class)
#
# print(mask)
#
# print(label_true[mask])
#
# print(label_true[True, True, True, True])
#
# print(label_true[[True, True, True, True]])

# c = torch.tensor([[0.1, 0.1, 0.9],
#                   [0.2, 0.4, 0.7],
#                   [0.9, 0.7, 0.4]])
#
# a = torch.ones(c.size())
# b = torch.zeros(3, 3)
#
#
#
# print(torch.where(c > 0.5, a, b))


def zeros_or_ones(size):
    return torch.ones(size) * 0.5 > torch.rand(size)

print(zeros_or_ones((10, 10)))


