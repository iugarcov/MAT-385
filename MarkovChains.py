# MAT 384/484 - Markov Chains

import numpy as np

from numpy import linalg as LA

#These are particular transition matrices: the entries in each column add up to one.
#This additional conditions assures us that the dominant eigenvalue is 1.
#The corresponding eigenvector gives the steady state.

#Example: Suppose that the matrix T below gives the loyalty probabilities of customers for three brands of cereal (or shoes, if you wish).
#Initially, 20% of buyers prefer brand 1, 20% prefer brand 2, and 60% prefer brand 3 (initial market share).
#Analyze how the market share changes with respect to time.



T=np.matrix([[.5, .6, .4], [.25, .3, .3], [.25, .1, .30]])

X0=np.array([[.2],
[.2],
[.6]])

print("Transition matrix:\n",T)

print("\nInitial probability distribution X(0):\n ",X0)

#To compute the probability vector X1 after one time period, we multiply T and X1.
#Matrix multiplication in Python is done using the .dot operator

X1=T.dot (X0)

print("Distribution X(1)=\n",X1)

#To compute the market distribution vector X10 after ten time periods, we multiply T^10 and X0.
#Power of matrices in Python can be done using the matrix_power command


X10=LA.matrix_power(T,10).dot(X0)

print("\nProbability distribution X(10)=\n",X10)

#Large powers of T show that the columns are identical. That's exactly the steady state vector.

print("\nLong term transition matrix (50 steps):\n",LA.matrix_power(T,50))

#We analyze the eigenvalues/eigenvectors of the matrix T to determine the steady-state distribution.
#The "Eigenvectors" command gives the eigenvalues (first column) and then associated eigenvectors for each eigenvalue 
#Notice that the first entry in the eigenvalues column is the dominant eigenvalue (in this case =1)


eigvals, eigvecs = LA.eig(T)

print("\nEigenvalues:\n", eigvals)

print("\nEigenvectors:\n",eigvecs)

#We calculate the steady state distribution by normalizing the dominant eigenvector.
#The indexing in Python starts from 0, hence the first column is column 0.

domeigvec=abs(eigvecs[:,0])

steady=domeigvec/LA.norm(domeigvec,1)

print("\nLong term steady state distribution:\n",steady)
