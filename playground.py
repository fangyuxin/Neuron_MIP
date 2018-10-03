import numpy as np

n_class = 3

label_true = np.array([0, 2, 1, 100])

label_pred = np.array([0.1, 1.9, 1.3, 2.2])

mask = (label_true >= 0) & (label_true < n_class)

print(mask)

print(label_true[mask])

print(label_true[True, True, True, True])

print(label_true[[True, True, True, True]])