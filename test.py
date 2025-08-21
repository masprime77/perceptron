import numpy as np

A = np.array([[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]])

B = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]])


C = A + B

print("A =\n", A)
print("B =\n", B)
print("A + B =\n", C)