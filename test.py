r = 1.15
Q_coeffs = [1.3, 2.4, 3.1]
Q = 0
for a in reversed(Q_coeffs):
	Q = a + Q * r
Q = 1 + Q * r

A = 1 + Q_coeffs[0] * r + Q_coeffs[1] * r * r + Q_coeffs[2] * r * r * r
print(Q)
print(A)
