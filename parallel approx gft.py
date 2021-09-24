#matlab starts iteration at 1 but pytrhon does it at 0. so all the code here works like that
import numpy as np
import math

sq = 200 #no of rows & columns of square matrix
# sq = 5  #if using the matrix I have defined
def get_sparse(laplacian, dim):
   
   lowertri = np.tril(laplacian) #convert laplacian to lower triangle to apply givens
   np.fill_diagonal(lowertri, 0) #convert all diagonal terms to 0 as we want row index > column index for givens
   lowertri = np.absolute(lowertri) #convert all terms in matrix to their absolute values
   pop = np.argwhere(lowertri!=0)
   non_zero = lowertri[np.nonzero(lowertri)]
   idx = non_zero.argsort()[::-1] 
   #print(lowertri)
   #print("\n\n-------------------\n",non_zero)
   #print(pop)
   non_zero = non_zero[idx]
   pop=pop[idx]
   #print("-------------------\n",non_zero)
   #print(pop)
   nos = [0]*sq
   S = np.zeros((dim, dim)) #form a dim x dim matrix with only zeroes
   np.fill_diagonal(S, 1) # add ones on the diagonal 
   n = math.floor(dim/2)
   i = 1
   while i<=n:
       for l in range(len(pop)):
           
           p, q = (pop[l])[0], (pop[l])[1]
           if (nos[p]==0 and nos[q]==0):
               nos[p] = 1
               nos[q] = 1
               numer = (laplacian[q][q] - laplacian[p][p])  #as per figure 3 ie the optimization subproblem
               denom = 2*laplacian[p][q]
               theta = (0.5 * math.atan(numer/denom)) + ((math.pi)/4)
               S[p][p] = math.cos(theta) #adding the terms according to givens formula
               S[q][q] = S[p][p]
               S[p][q] = math.sin(theta)
               S[q][p] = (-1)*S[p][q]
               i+=1
   
   return S 

np.random.seed(0)

import time #to measure time taken
# laplacian = np.array([[1,2,-3,4,6],[2,9,4, 23, 9],[0,-20,-18, 15, 1], [123,734, -63, 1, 7],[-234, 85, 34, 8, 0]], float)
laplacian = np.random.rand(sq,sq) #makes a random 5 x 5 matrix
n = np.shape(laplacian)[0]    #find dimension of laplacian 
order  = 2     #for J
J = order*n*math.log(n) #taking J to be of order n log n
print("\nJ is currently\n\n", J)
# J = n^(a) ##   a<2  we can also take J to be of this form
L = laplacian
a = time.time()  #to measure time taken
sparse_list = [] 
j=0
k =1
while j < math.floor(J): #we take the integer value of J ie. if J = 4.67 then floor(J) will be 4   range(n) starts from 0 to n-1 so a total of n iterations 
    S = get_sparse(L, n)
    S1 = S.transpose()
    L = S1@L@S # the @ sign is for matrix multiplication
    sparse_list.append(S)
    j += n/2
    k += 1
K = k

diagonal_elem = np.diag(L) #take only diagonal terms of L in a single 1d array
#print(diagonal_elem)
idx = diagonal_elem.argsort()[::] 
diagonal_elem = diagonal_elem[idx]
#print(diagonal_elem)
L = np.diag(diagonal_elem) 
#print("\n\nApprox. diagonalized L after rearrangement\n",L)
sparse_list.pop()
#print("\n\nSk before reaarangement\n",S)
S=S[:,idx]
#print("\n\nSk after reaarangement\n",S)
sparse_list.append(S)


b = time.time() #to measure time taken
print("\n\nTimes taken is", b - a) #to measure time taken

print("j is", j)
print("K is", K)