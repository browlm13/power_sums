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

	

	Making it more general...


	q(n+1, 2) = q(n, 2)    +  2*q(n, 1).  + n+1
	q(n+1, 3) = q(n, 3)    +  3*q(n, 2)   + 3*q(n, 1) 	  + n+1
	q(n+1, 4) = q(n, 4)    +  4*q(n, 3)   + 6*q(n, 2)     + 4*q(n, 1) 	  + n+1
	...
	q(n+1, d) = the sum k=0 to k=d of (d choose k)*q(n, k)


	Rewritting...

	q(n+1, 2) = q(n, 2)    +  2*q(n, 1).  + n+1
	(n+1)^2 = 2*q(n, 1)   + n+1
	(n+1)^2 = the sum k=0 to k=2-1 of (2 choose k)*q(n, k)
	(n+1)^2 -(n+1) = the sum k=1 to k=2-1 of (2 choose k)*q(n, k)

	(2 choose 2-1) = (n+1)^2 -(n+1) - the sum k=1 to k=2-2 of (2 choose k)*q(n, k)

	or in general...

	(n+1)^d -(n+1) = the sum k=1 to k=d-1 of (d choose k)*q(n, k)

	(d choose d-1) = (n+1)^2 -(n+1) - the sum k=1 to k=d-2 of (d choose k)*q(n, k)



"""

# [TODO]: MAke work

from scipy.special import comb

# solve to max_d
MAX_D = 4 
C = np.zeros(shape=(MAX_D+2, MAX_D+3))

# initialize 0th row of C as sum of n*1 for consistancy (last column)
C[0,:] = np.zeros(shape=(MAX_D+3,))
C[0,1] = 1.0 # first column represents x^1 or n^1

print(C)

def get_pd(d, C):

	p = np.poly1d(C[d-1,:][::-1])
	return p

for d in range(2,MAX_D+1):


	#init rhs
	rhs = np.poly1d(np.zeros(shape=(C.shape[1]))) # (all zeros so it does not matter)

	#
	# calculate (n+1)**(d+1) - n+1
	#

	# first calculate (n+1)**(d+1) 
	for k in range(0, d+1): 
		rhs[k]= comb(d, k, exact=True)


	# subtract n+1
	rhs -= np.poly1d([1.0,1.0])

	


	#
	# subtract correct multiples of q(n,d=k) according to binomial coeffs
	#

	#  do not include k=d-1 (the target)
	coeffs = [comb(d, k, exact=True) for k in range(1,d-1)]
	pds = [get_pd(k, C) for k in range(1, d-1)]

	for c,p in zip(coeffs,pds):
		rhs -= c*p

	rhs /= comb(d, d-1, exact=True)

	print(rhs)


	# set coefficient matrix
	#C[d,:] = rhs.c[::-1]
	for k in range(C.shape[1]):
		C[d-1,k] = rhs[k]


	print(C)
