

"""

	Finding the formula for,
		sum of first n natural numbers...

	     (*) q(n, d=1) = n^2/2 + n/2


	  	We will use...

			sum of first n squares (unkown, recursive definition)...

				  		    q(n+1, d=2) = q(n, d=2) + (n+1)^2

				  then,

				  		(1)  q(n+1, 2) - q(n, 2)  =  (n+1)^2

			(1)  q(n+1, 2) - q(n, 2)  =  (n+1)^2


			expansion of squares...

				(x + 1)^2 =  x^2  +  2x    + 1

				(2) (n + 1)^2 = n^2  + 2n +1

			(2) (n + 1)^2 = n^2  + 2n +1



		Derivation...



	          	c0               c1            c2               c3
	         ----------------------------------------------------
	 1^2  =   (0 + 1)^2     =    0^2         +  2*0              + 1
	 2^2  =   (1 + 1)^2     =    1^2         +  2*1              + 1
	 3^2  =   (2 + 1)^2     =    2^2         +  2*2              + 1
	 ...
+(n+1)^2  =   (n + 1)^2     =    n^2         +  2*n              + 1 
----------------------------------------------------------------------
			   q(n+1, 2)    =   q(n, 2)      +  2*q(n, 1)  	     + n+1
			   									2*(TARGET)

			Subtract c1 from both sides,

				q(n+1, 2) - q(n, 2) =     2*q(n, 1)  + n+1

			Substitute equation (1) for the left hand side,

				(n+1)^2 =  2*q(n, 1)  + n+1

			And (2) for the left hand side again so it looks clean,
				n^2  + 2n +1 = 2*q(n, 1)  + n+1

			We can now solve for q(n, 1),

				q(n, 1) = 1/2[n^2  + n]
				q(n, 1) = n^2/2 + n/2 


	This process can be used to derive a formula for q(n, d) for d = 2,3,4,5 ...


"""

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

# method 1 - recursive solver
def recursive_solver(max_d):

	# solve to max_d
	C = np.zeros(shape=(max_d+1, max_d+2))

	# initialize 0th row of C as formula for q_0(n) which is (1)*n^(1) as a polynomial
	C[0,1] = 1.0 # first column represents x^1 or n^1


	# include max_d, but not max_d+1
	for a in range(1,max_d+1):

		# (-1)*n**0
		C[a,0] += -1.0

		# include a-1, but not a
		for k in range(a):
			c = comb(a+1, k, exact=True)
			C[a,k] += c
			C[a,:] += -c*C[k,:]

		# (a+1 choose a)*n**a
		C[a,a] += comb(a+1, a, exact=True)

		# (a+1 choose a+1)*n**(a+1)
		C[a,a+1] += 1.0

		# (a+1 choose a)**(-1) 
		C[a,:] *= 1.0/comb(a+1, a, exact=True)


	# return coefficient matrix
	return C

def get_p(d, C):

	p = np.poly1d(C[d,:][::-1])
	return p


# method 2 - induction coef solver -- direct
def direct_solver(d):	

	# create b
	b = np.ones(shape=(d+2,))
	for i in range(d+1):
		b[i+1] = comb(d, i, exact=True)

	# create A
	A = np.zeros(shape=(d+2, d+2))
	A[0,:] = np.ones(shape=(d+2,))
	for r in range(d+2):
		# c choose r
		for c in range(r+1, d+2):
			A[r+1,c] = comb(c, r, exact=True)

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

	# compute polynomials coeffient matrix
	C = recursive_solver(MAX_D)

	# check against known formula 
	for d in range( MAX_D+1):

		# compute actual values using known formula
		actual_vals = q(ns, d=d)

		# compute values using rescursive solver
		p_recursive = get_p(d,C)
		recursive_vals = p_recursive(ns)

		# compute values using inductive method
		p_direct = direct_solver(d)
		direct_vals = p_direct(ns)

		# compute the total errors
		recursive_error = abs(np.sum(np.array(actual_vals-recursive_vals)))
		direct_error = abs(np.sum(np.array(actual_vals-direct_vals)))

		# display results
		print("\nd=%s, " % d)
		print("Derived poly (recursive): ", p_recursive)
		print("Derived poly (direct): ", p_direct)
		#print("Known forumla results: ", actual_vals)
		#print("Derived formula results (recursive): ", recursive_vals)
		#print("Derived formula results (direct): ", direct_vals)
		print("Total Error (recursive): ", recursive_error)
		print("Total Error (direct): ", direct_error)
		print("re - de = %s" % (recursive_error-direct_error))



