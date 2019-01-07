import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank
from scipy.linalg import lu
from scipy.linalg import pascal
import sympy

# define a translation matrix Lh of size nxn
def poly_translation_matrix(n, h):

	# pascal matrix
	P = pascal(n, kind='lower')

	# exponent matrix
	L = np.tril(np.ones(shape=(n,n)))
	R = np.tril(matrix_power(L, 2)-1)

	# H matrix, H's raised to corresponding powers
	H = np.tril(np.power(h,R)) 

	# Create D poly shift matrix with offset h
	Lh = (H * P) 

	return Lh


def translate_polynomial(p, h):

	# q(x) = p(x + h)
	Lh = poly_translation_matrix(len(p)+1,h)
	coefs = Lh.T @ p.c[::-1]
	q = np.poly1d( coefs[::-1] )

	return q
