from trial_data import data_handler
import numpy as np
import math
from collections import OrderedDict
import os

data = data_handler("rating_with_timestamp.mat", "trust.mat", "epinion_trust_with_timestamp.mat")
data = data.load_matrices()
print "LOADED MATRICES"
K = data[2]

P = data[0] 
print "set p" +  str(len(P))
G = data[1]
print "set G" + str(len(G))
G_original = data[3]
_lambda = 10


def calcS(P): 
    #calculating matrix S or homophily coefficient matrix using formula in paper

    # Z = np.load(fname + '.npy')

    n = len(P)
    numerator = np.dot(P, P.transpose())

    J_norm = (np.sum(np.abs(P)**2,axis=-1)**(1./2)).astype(np.float32)
    # J_norm = J_norm.astype(np.float32)
    J_norm = J_norm.reshape(n,1)
    I_norm = J_norm.transpose().astype(np.float32)

    denominator = np.dot(J_norm,I_norm)

    # dividing numerator and denominator and handling setting all nan to zero

    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.divide(numerator,denominator).astype(np.float32)
        # Z[Z == np.inf] = 0
        Z = np.nan_to_num(Z)

    # fname = "store_Z_10"
    # np.save(fname,Z)
    
    return Z

S = calcS(P)

def hTrust(G, S, _lambda, K, maxIter, P, G_original):
    beta = 0.01
    alpha = 0.01


    # construct L using formula in paper
    # Summing up elements of each row
    d = [np.sum(x) for x in S]
    D = np.diag(d).astype(np.float32) # D = Diagonal matrix
    
    L = np.subtract(D,S).astype(np.float32) # L = Laplacian matrix
    

    [n, n] = G.shape


    # Initialize U and V randomly

    U = np.random.uniform(0.0,0.1,(n,K))
    V = np.random.uniform(0.0,0.1,(K, K))

    # U = np.ones((n,K)) * 0.05
    # V = np.ones((K,K)) * 0.05



    # U = np.random.random((n, K))
    # V = np.random.random((K, K))
        # E1 = np.absolute(np.linalg.norm(U, ord = 'fro')) 
        # E2 = np.absolute(np.linalg.norm(V, ord = 'fro'))


    iter = 0

    #% Hamid: Main loop
    while (iter < maxIter):

        UU = np.dot(U.T, U)

        A = np.dot(U, np.dot(V.T, np.dot(UU, V))) + np.dot(U,np.dot(V,np.dot(UU, V.T))) + alpha * U + _lambda * np.dot(D, U) + 1e-8
        B = np.dot(np.dot(G.T, U),V) + np.dot(np.dot(G, U),V.T) + _lambda * np.dot(S, U)
        U = U * np.sqrt(B / A)

        AV = np.dot(np.dot(UU, V), UU) + beta * V + 1e-8
        BV = np.dot(np.dot(U.T, G), U)
        V = V * np.sqrt(BV/ AV)

        # Obj = np.linalg.norm((G - np.dot(np.dot(U, V),U.T)), ord =('fro')) ** 2 + alpha * np.linalg.norm(U, ord = ('fro')) ** 2 + beta * np.linalg.norm(V, ord = ('fro')) ** 2 + _lambda * np.trace(np.dot(np.dot(U.T, L) * U))
        print (('the object is in iter %d is %f'), iter)

        iter = iter + 1


    GC = np.dot(np.dot(U, V),U.T)

    print "TP accuracy: " + str(TP_accuracy(G_original, G, GC))

    print "FINAL MATRIX CALCULATED"
    # print GC
    return GC

def TP_accuracy(G_original, G, GC):
    
    G_final = GC
    # set ratio of data to split
    ratio = int(len(G_original) *0.5)
   
    
    #creating sets A,C,D,B according to the paper
    A = G_original.tolist()

    C = A[:ratio]
    D = A[ratio:]

    D = [tuple(y) for y in D]
    D = set(D)

    B = []
    for (x,y) in np.ndenumerate(G):
        # if len(B) == 4 * len(D):
        #     break
        [i,j] = [x[0],x[1]]
        # as user numbers are one more than matrix indices
        array = [i+1,j+1]
        if G[i,j] == 0:
            B.append(array)

    B = [tuple(b) for b in B]
    B = set(B)
    DUB = D.union(B)

    # ranking DUB pairs in decreasing order of confidence
    rank_dit_1 = {}
    for pair in DUB:
        (x,y) = pair
        #getting trust stengths from predicted G
        rank_dit_1[(pair[0],pair[1])] = G_final[x-1,y-1]

    #sorting dictionary based on descending order of values (trust stengths)
    d_descending = OrderedDict(sorted(rank_dit_1.items(), key=lambda kv: kv[1], reverse=True))
    
    rank_list = d_descending.keys()
    
    # taking first D pairs from ranked list to make E (as in paper)
    E = set(rank_list[:len(D)])
    
    # calculating accuracy according to formula
    TP = (float)(len(D.intersection(E)))/len(D)
    
    return TP 


print hTrust(G,S,_lambda, K, 1000, P, G_original)
