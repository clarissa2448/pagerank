'''
PageRank Project by Hannah He and Clarissa Xu.
Names of Primary Functions: 1. probMatrix: input: chain, type: 2d array. Output: M, type: 2d array
                            2. isDanglingNode: input: chain, j, type: 2d array, index of the node. Output: true/false, type: boolean
                            3. rank: input: chain, type: 2d array. Output: result, type: 1d array
'''
import numpy as np
import math
chain = [[1,2,3],[3],[1,3],[1,2]]
chain1 = [[1,2,3],[3],[0,3],[0,2]]
chain2 = [[1,2,3],[3],[0,3],[]]
chain3 = [[1, 5], [2, 5], [1, 3, 5], [4], [1, 5], [2, 6], [0, 1]]
chain4 = [[1,3,4],[0,2,4],[3,6],[2,4,6],[5,8],[4,6,8],[0,7,9],[0,6,8],[2,9],[0,2,8]]

#Creates the probaility tranisiton matrix of the input
def probMatrix(chain):
    #Probability Matrix
    #M_{ij) = 1 / len
    isDG = False #is there a dangling node?
    M = [[0] * len(chain) for i in range(len(chain))]
    for j in range(len(chain)):
        if isDanglingNode(chain, j):
            isDG = True
        for i in range(len(chain)):
            if isDG:
                M[j][i] = 1/len(chain) #1/number of outgoing edges
            elif j in chain[i]:
                M[j][i] = 1 / len(chain[i]) #1/ number of nodes
        isDG = False
    return M

def isDanglingNode(chain, j):
    return chain[j] == [] #empty list for no outgoing edges

def rank(chain):
    x = [1 for i in range(len(chain))] #creates a n x 1 list of 1s
    M = probMatrix(chain)
    arrEig = np.linalg.eig(M)
    eigVal = list(arrEig[0]) #list of eigenvalues
    eigVec = arrEig[1] #matrix of eigenvectors = P
    min_index = -1
    min_dist = float('inf')
    #Compares the eigen vector array to find the eigen vector that corresponds to the eigenvalue of 1
    for i in range(len(eigVal)):
        lambda_i = eigVal[i]
        dist = abs(lambda_i -1)
        if(dist < min_dist):
            min_index = i
            min_dist = dist
    evOne = eigVec[:,min_index]
    #makes the eigenvector with the eigenvalue of 1 positive
    for i in range(len(evOne)):
        if(evOne[i] < 0):
            evOne[i] *= -1
    eigVecInv = np.linalg.inv(eigVec) #inverse of eigenvector matrix
    n = len(chain)
    C = np.zeros((n,n)) #creates matrix with the eigenvector as the first column, 0 elsewhere
    C[:,0] = evOne
    #Compute [eigenvector of 1] * P^-1 * x_0
    prod = np.matmul(C, eigVecInv)
    prod1 = np.matmul(prod,x)
    result = [0 for i in range(n)]
    for i in range(len(prod1)):
        result[i] = np.abs(prod1[i])
    result = [j for (i,j) in sorted([(s,t) for (t,s) in enumerate(result)])]
    result = result[::-1]
    return result



print(rank(chain))
