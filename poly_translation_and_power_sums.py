

import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank
from scipy.linalg import lu
from scipy.linalg import pascal
import sympy

def basic_D(n, h):
	# pascal matrix
	P = pascal(n, kind='lower')

	# exponent matrix
	L = np.tril(np.ones(shape=(n,n)))
	R = np.tril(matrix_power(L, 2)-1)

	# H matrix, H's raised to corresponding powers
	H = np.tril(np.power(h,R)) 

	# Create D poly shift matrix with offset h
	D = (H * P) 

	return D.T 


def solve_poly(A, q, x0, y0):

	# remove uneccisary columns
	A,q = A[:-1,1:], q[:-1]

	# solve system for coeffients a1->an
	p = np.zeros(shape=(len(q)+1,))
	p[1:] = np.linalg.solve(A,q)

	# solve system for final coefficient
	diff = polyval(x0, p, tensor=True)

	if x0 != 0:
		p[0] = (y0 - diff)/x0

	return p

# recursion: p(x +h) = p(x) + q(x+h), p(x0) = y0
def interpolate_recursion(q, h, x0, y0):
	# q is list of coefficients on [x^0, x^1, ..., x^{n-1}]

	# get degree of p, n, from degree of q, n-1
	n = len(q)

	# get shift matrix
	D_h = basic_D(n+1, h) 

	# turn q(x) into q(x+h) and add row to q with zero element
	q_h = np.zeros(shape=(n+1,))
	q_h[:-1] = q
	#q_h = D_h @ q_h


	# Find A = D_h - I, from x(D_h - I)p = x(D_h)q
	I = np.eye(n+1)
	A = D_h - I

	# solve for p
	p = solve_poly(A, q_h, x0, y0)

	#return p
	return np.poly1d(p[::-1])


def power_sum( n, d=1 ):
	""" power_sum( n ) - power_sum( n-1 ) = n**d """

	if np.isscalar(n):
		# must be an integer
		n = int(n)
		s = 0
		for i in range(1,n+1): 
			s += i**d
		return s

	return np.array([ q( ni, d=d ) for ni in n ])

def power_sum_polynomial(degree):

	# step size
	h = 1

	# any initial x value
	x0 = 4

	# known y
	y0 = power_sum(x0, d=degree)

	# create array of coefficients for p(x+h) - p(x) = (x+h)^degree
	q = np.zeros(shape=(degree+1,))
	q[-1] = 1
	D_h = basic_D(degree+1, 1)
	q = D_h @ q

	# solve for polynomial
	p = interpolate_recursion(q, h, x0, y0)

	return p


# testing with step size 2
degree = 5
h = 2

# known value to solve for final coefficient 
x0 = 20
y0 = power_sum(x0, d=degree)

# create polynomial difference function (x + 1)^degree + (x + 2)^degree
q = np.zeros(shape=(degree+2,))
q[-2] = 1
D_h1 = basic_D(degree+2, 1)
q1 = D_h1 @ q
D_h2 = basic_D(degree+2, 2)
q2 = D_h2 @ q
q = q1+q2
q = q[:-1]

# solve for interpolating polynomial
p = interpolate_recursion(q, h, x0, y0)

# display error
test_x = 100
print(power_sum(test_x, d=degree) - p(test_x))


for d in range(10):
	p = power_sum_polynomial(d)

	for n in range(6):
		print(power_sum(n, d=d) - p(n))


