import numpy as np
chain = [[1,2,3],[3],[1,3],[1,2]]

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
                M[i][j] = 1/len(chain)
            elif j in chain[i]:
                M[i][j] = 1 / len(chain[i]) 
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
    eigVal = arrEig[0].tolist() #list of eigenvalues
    eigVec = arrEig[1] #matrix of eigenvectors
    
    #fix rounding
    '''for i in range(len(eigVec)):
        for j in range(len(eigVec[i])):
            eigVec[i][j] = round(eigVec[i][j], 2)'''
    
    for i in range(len(eigVal)):
        eigVal[i] = round(eigVal[i], 1)
    
    i = eigVal.index(1) #index of eigenval 1
    evOne = eigVec[i] #eigvector for eigenval 1
    eigVecInv = np.linalg.inv(eigVec) #inverse of eigenvector matrix
    
    #Compute [eigenvector of 1] * P^-1 * x_0
    print(eigVec.tolist())
    print(evOne, eigVecInv)
    return np.multiply(np.multiply(evOne, eigVecInv), x)
    
    
print(rank(chain))
    
