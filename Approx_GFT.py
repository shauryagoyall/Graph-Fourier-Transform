#matlab starts iteration at 1 but pytrhon does it at 0. so all the code here works like that
import numpy as np
import math

def get_sparse(laplacian, dim):
   
   lowertri = np.tril(laplacian) #convert laplacian to lower triangle to apply givens
   np.fill_diagonal(lowertri, 0) #convert all diagonal terms to 0 as we want row index > column index for givens
   lowertri = np.absolute(lowertri) #convert all terms in matrix to their absolute values
   loc = np.where(lowertri == np.max(lowertri)) #find index of the maximum absolute value in matrix
   p, q = int(loc[0]) , int(loc[1]) # assign it to p and q
   numer = (laplacian[q][q] - laplacian[p][p])  #as per figure 3 ie the optimization subproblem
   denom = 2*laplacian[p][q]
   theta = (0.5 * math.atan(numer/denom)) + ((math.pi)/4)
   S = np.zeros((dim, dim)) #form a dim x dim matrix with only zeroes
   np.fill_diagonal(S, 1) # add ones on the diagonal 
   S[p][p] = math.cos(theta) #adding the terms according to givens formula
   S[q][q] = S[p][p]
   S[p][q] = math.sin(theta)
   S[q][p] = (-1)*S[p][q]
   return S 

#np.random.seed(0)

import time 
# laplacian = np.array([[1,2,3, 4 , 6],[2,9,4, 23, 9],[0,-20,-18, 15, 1], [123,734, 63, 1, 7],[234, 85, 34, 8, 0]], float)
laplacian = np.random.rand(100,100) #makes a random matrix
n = np.shape(laplacian)[0]    #find dimension of laplacian 
order  = 2     #for J
J = order*n*math.log(n) #taking J to be of order n log n
# J = n^(a) ##   a<2   J can also be taken of this form
L = laplacian
start = time.time()  
sparse_list = [] 
for j in range(math.floor(J)): #we take the integer value of J ie. if J = 4.67 then floor(J) will be 4   range(n) starts from 0 to n-1 so a total of n iterations 
    S = get_sparse(L, n)
    S1 = S.transpose()
    L = S1@L@S # the @ sign is for matrix multiplication
    sparse_list.append(S)

diagonal_elem = np.diag(L) #take only diagonal terms of L in a single 1d array
idx = diagonal_elem.argsort()[::] 
diagonal_elem = diagonal_elem[idx]
L = np.diag(diagonal_elem) #makes L, a diagonal matrix
sparse_list.pop()
S=S[:,idx] #reorder columns of S as per idx
sparse_list.append(S)

end = time.time() #to measure time taken
print("\n\nTimes taken is", end - start) #to measure time taken 
