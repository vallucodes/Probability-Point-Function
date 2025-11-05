import numpy as np

# # Example coefficients (replace with your own)
# P = [0.5, 0.2, -0.1]  # numerator coefficients
# Q = [0.3, 0.1]        # denominator coefficients
# x = 0.1               # test input

# # Compute using the same logic as in your approximation
# u = x - 0.5
# r = u * u

# # Horner’s method for P and Q
# P_val = 0
# for a in reversed(P):
#     P_val = a + P_val * r

# Q_val = 0
# for b in reversed(Q):
#     Q_val = b + Q_val * r
# Q_val = 1 + Q_val * r

# print("P:", P_val)
# print("Q:", Q_val)

# z = u * (P_val / Q_val)
# print("z =", z)


from scipy.stats import norm

x = 0.99
z = norm.ppf(x)
print(z)
