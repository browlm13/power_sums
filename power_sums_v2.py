import numpy as np
from scipy.special import comb

# power sum
def q( n, d=1 ):
	""" q( n ) - q( n-1 ) = n**d """

	if np.isscalar(n):
		# must be an integer
		n = int(n)
		s = 0
		for i in range(1,n+1): 
			s += i**d
		return s

	return np.array([ q( ni, d=d ) for ni in n ])


# method 2 - induction coef solver -- direct
def direct_solver(d, m):	

	# create b
	b = np.ones(shape=(m+1,))
	for i in range(m):
		b[i+1] = comb(d, i, exact=True)

	# create A
	A = np.zeros(shape=(m+1, m+1))
	A[0,:] = np.ones(shape=(m+1,))
	for r in range(1,m+1):
		for c in range(r,m+1):
			A[r,c] = comb(c, r-1, exact=True)

	# solve system
	x = np.linalg.solve(A,b)

	# return polynomial interpolant
	return np.poly1d(x[::-1])


#
# Testing
#


if __name__ == "__main__":

	# Settings -- Check until q_d(n), for n=1,2,...,MAX_N
	MAX_D = 10
	MAX_N = 5

	# generate test values
	ns = np.linspace(1,MAX_N,MAX_N)

	# check against known formula 
	for d in range( MAX_D+1):

		# compute actual values using known formula
		actual_vals = q(ns, d=d)


		# compute values using inductive method
		# polynomial degree
		m = d+1
		p_direct = direct_solver(d, m)
		direct_vals = p_direct(ns)

		# compute the total errors
		direct_error = abs(np.sum(np.array(actual_vals-direct_vals)))

		# display results
		print("\nd=%s, " % d)
		print("Derived poly (direct): ", p_direct)
		print("Total Error (direct): ", direct_error)
