#
# Create pascal matrix
#

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import pascal

h = 2
n = 5

# pascal matrix
P = pascal(n, kind='lower')
#print(P)

# exponent matrix
L = np.tril(np.ones(shape=(n,n)))
R = np.tril(matrix_power(L, 2)-1)
print(R)

# H matrix, H's raised to corresponding powers
H = np.tril(np.power(h,R))
print(H)

# A
A = H * P
print(A)

# coeffcients: a0, a1, ..., an are multiplied row wise
# powers of x: x^0, x^1, ..., x^n are multiplied column wise

# ex:

# evaluate at this x
x = 3

# create random coefficients
coef = np.random.randint(-4,4, n)#np.random.randn(n,) 
print(coef)

# create powers of x: x^0, x^1, ..., x^n
xs = np.power(np.ones(shape=(n,))*x, np.linspace(0,n-1,n))
print(xs)



# calculate predicted shifted polynomial by h at x

b_prime = A @ xs
b = b_prime @ coef
print(b_prime)
print(b)

# evalutate the polynomial directly
from numpy.polynomial.polynomial import polyval
true_value = polyval(x+h, coef, tensor=True)
print(true_value)
