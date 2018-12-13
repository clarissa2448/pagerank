import numpy as np
import math
chain = [[1,2,3],[3],[1,3],[1,2]]
chain1 = [[1,2,3],[3],[0,3],[0,2]]
taischain = [[1,2,3],[3],[0,3],[]]
def probMatrix(chain):
    #Probability Matrix
    #M_{ij) = 1 / len
    isDG = False
    M = [[0] * len(chain) for i in range(len(chain))]
    for j in range(len(chain)):
        if isDanglingNode(chain, j):
            isDG = True
        for i in range(len(chain)):
            if isDG:
                M[j][i] = 1/len(chain)
            elif j in chain[i]:
                M[j][i] = 1 / len(chain[i])
        isDG = False
    return M

def isDanglingNode(chain, j):
    return chain[j] == []

def isReducible(chain):
    pass


def rank(chain):
    x = [1 for i in range(len(chain))]
    M = probMatrix(chain)
    arrEig = np.linalg.eig(M)
    eigVal = list(arrEig[0]) #list of eigenvalues
    eigVec = arrEig[1] #matrix of eigenvectors = P
    min_index = -1
    min_dist = float('inf')
    for i in range(len(eigVal)):
        lambda_i = eigVal[i]
        dist = abs(lambda_i -1)
        if(dist < min_dist):
            min_index = i
            min_dist = dist
    evOne = eigVec[:,min_index]
    for i in range(len(evOne)):
        if(evOne[i] < 0):
            evOne[i] *= -1
    eigVecInv = np.linalg.inv(eigVec) #inverse of eigenvector matrix
    n = len(chain)
    C = np.zeros((n,n))
    C[:,0] = evOne
    print(evOne)
    #Compute [eigenvector of 1] * P^-1 * x_0
    prod = np.matmul(C, eigVecInv)
    prod1 = np.matmul(prod,x)
    result = [0 for i in range(n)]
    for i in range(len(prod1)):
        result[i] = np.abs(prod1[i])
    print("RES", result)
    result = [j for (i,j) in sorted([(s,t) for (t,s) in enumerate(result)])]
    result = result[::-1]
    return result



print(rank(taischain))
