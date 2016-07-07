import numpy as np
import math
from data_handler_hTrust import data_handler
import operator
from operator import itemgetter
from collections import OrderedDict

class Social_Status:
    def __init__(self,G,P,d,max_itr, G_original):

        self.G_original = np.array(G_original)
        self.G = G
        self.n = len(self.G)
        self.max_itr = max_itr
        self.d = d #number of user perferences (facets)
        self.alpha = 0.01
        self.beta = 0.01
        self.lambda1 = 10
        self.Z = np.zeros((self.n,self.n)) 
        self.P = P #user-rating matrix (ixk)
        # self.W = np.ones((self.n,self.n)) 
        # self.R = np.zeros((self.n,self.n)) 
        self._oldU = np.zeros((self.n, self.d))
        self.U= np.random.random((self.n, self.d))
        self._oldH = np.zeros((self.d, self.d))
        self.H = np.random.random((self.d, self.d))
        self.G_final = np.zeros((self.n,self.n))
        self.D = np.zeros((self.n,self.n))
        self.TP = -1

       

    def calcZ(self): 

        numerator = np.dot(self.P, self.P.transpose())

        J_norm = np.sum(np.abs(self.P)**2,axis=-1)**(1./2)
        J_norm = J_norm.reshape(self.n,1)
        I_norm = J_norm.transpose()
        print I_norm.shape

        denominator = np.dot(J_norm,I_norm)

        # dividing numerator and denominator and handling nan and inf values

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = np.divide(numerator,denominator)
            # Z[Z == np.inf] = 0
            Z = np.nan_to_num(Z)



        self.Z = Z
        print "FOUND Z!!"
        print Z
             

    
    def converge(self, iterNO):
        #Returns True if Converged, else return False
        # Max iterations reached
        if iterNO >= self.max_itr:
            return True

        # Convergence is reached
        # EPS = np.finfo(float).eps
        EPS = 0.000001
        E1 = np.absolute(np.linalg.norm(self.U, ord = 'fro') - np.linalg.norm(self._oldU, ord = 'fro'))
        E2 = np.absolute(np.linalg.norm(self.H, ord = 'fro') - np.linalg.norm(self._oldH, ord = 'fro'))
        if E1 < EPS and E2 < EPS:
            if iterNO != 1:   # Skip for the 1st iteration
                print("\rIteration: %d FinalError: (%f, %f) EPS:%f" %(iterNO, E1, E2, EPS))
                return True

        self._oldU = self.U[:]
        self._oldH = self.H[:]
        
        print "E1 Difference: " + str(E1-EPS) + "E2 Difference: " + str(E2-EPS)

        #print("\rIteration: %d FinalError: (%f, %f) EPS:%f" %(iterNO, E1, E2, EPS))
        return False

    def updateMatrices(self):

        term_1 = np.dot(np.dot(self.U,self.H),self.U.transpose())
        term_2 = np.dot(np.dot(self.U,self.H.transpose()),self.U.transpose())

        A_1 = np.dot(np.dot(self.G.transpose(),self.U),self.H)
        A_2 = np.dot(np.dot(self.G,self.U),self.H.transpose())
        A_3 = self.lambda1 * np.dot(self.Z,self.U)
        
        A = A_1 + A_2 + A_3 
        # print "HERE IS A"
        # print A

        B_1 = np.dot(np.dot(term_2,self.U),self.H)
        B_2 = np.dot(np.dot(term_1,self.U),self.H.transpose())
        B_3 = self.alpha * self.U
        B_4 = self.lambda1 * np.dot(self.D,self.U)
        
        B = B_1 + B_2 + B_3 + B_4 
        # print "HERE IS B"
        # print B

        
        C = np.dot(np.dot(self.U.transpose(),self.G),self.U)
        # print "HERE IS C"
        # print C

        D_1 = np.dot(np.dot(self.U.transpose(),term_1),self.U)
        D_2 = self.beta * self.H
        
        D = D_1 + D_2 
        # print "HERE IS D"
        # print D

        # print "STARTING U,H UPDATES"

        # test_B = B != 0
        # self.U = self.U * np.sqrt(A / B)
        # self.U[np.where(test_B==False)] = 0

        # test_D = D != 0
        # self.H = self.H * np.sqrt(C / D)
        # self.H[np.where(test_D==False)] = 0


        test_B = B != 0
        self.U = self.U * np.sqrt(A / B)
        indices_1 = zip(*np.where(test_B==False))
        # print "ORIGINAL LIST"
        # print np.where(test_B==False)
        for x,y in indices_1:
            self.U[x,y] = self._oldU[x,y]

        # self.U[np.where(test_B==False)] = self._oldU[np.where(test_B==False)]


        test_D = D != 0
        self.H = self.H * np.sqrt(C / D)
        indices_2 = zip(*np.where(test_D==False))
        for x,y in indices_2:
            self.H[x,y] = self._oldH[x,y]

        # print self._oldH[np.where(test_B==False)]

        # self.H[np.where(test_D==False)] = self._oldH[np.where(test_B==False)]


    def start_main(self):
        self.calcZ()
        print "DONE with first step calc Z"
        max_itr = 1000
        

        P = np.zeros(self.G.shape)
        L = np.zeros(self.G.shape)

        #calculating homophily contribution
        for i in range(0,self.n):
            total = 0
            for j in range(0,self.n):
                total = total + self.Z[j,i]
            self.D[i,i] = total

        L = self.D - self.Z
        # homo_contribution = 2 * np.trace(np.dot(np.dot(self.U.transpose(),L),self.U))
        # print "Calculated homphily contribution"

        print "DONE with generating randos"

        self.U = np.random.random((self.n,self.d))
        self.H = np.random.random((self.d,self.d))

        i = 1

    
        # print self.converge(i)
        while not i > max_itr:
            print ("Iteration: ", i)
            
            self.updateMatrices()
            #print self.U, self._oldU
            i = i + 1
        
        print "Found U and H successfully!"
        print "THEY ARE"
        print self.U, self.H
        self.calcTrust()

            

    def calcTrust(self):
        #calculate all final trust values knowing U,H

        self.G_final = np.dot(self.U,self.H)
        self.G_final = np.dot(self.G_final, self.U.transpose())

        print "Found predicted trust! Has TP accuracy: " + str(self.TP_accuracy())
        #print G_final

        print "FINAL PREDICTED"
        print self.G_final
        
        print "START OFF AS"
        print self.G

        return self.G_final


    def RMSE(self):
        return np.sqrt(np.mean((self.G_final - self.G)**2))

    def TP_accuracy(self):
        # set ratio of data to split
        ratio = int(len(self.G_original) *0.5)
        
        A = self.G_original.tolist()
    
        C = A[:ratio]
        # print C
        D = A[ratio:]
        # print D
    
        D = [tuple(y) for y in D]
        D = set(D)


         #set of no trust user pairs

        # q = 0.5
        # while q <=50:
        #     print "NOW TRYING: " + str(q)
        #     if 0.23 <= self.TP <= 0.24:
        #         break

        B = []
        for (x,y) in np.ndenumerate(self.G):
            # if len(B) == 4 * len(D):
            #     break
            [i,j] = [x[0],x[1]]
            # as user numbers are one more than matrix indices
            array = [i+1,j+1]
            if self.G[i,j] == 0:
                B.append(array)
        # print "B IS THIS BIG" + str(len(B))

        B = [tuple(b) for b in B]
        B = set(B)
        DUB = D.union(B)

        # ranking DUB pairs in decreasing order of confidence
        rank_dict_1 = {}
        for pair in DUB:
            (x,y) = pair
            rank_dict_1[(pair[0],pair[1])] = self.G_final[x-1,y-1]

        # print "BEFORE SORTING"
        # print rank_dict_1
        d_descending = OrderedDict(sorted(rank_dict_1.items(), key=lambda kv: kv[1], reverse=True))
        # print "AFTER SORTING"
        # print d_descending
        # print "BREAK BREAK"
        # print "LEN OF D IS" + str(len(D))
        rank_list = d_descending.keys()
        # print "WHERE IS [110,53]????"
        # print str(rank_list.index((110,53))) + " OF " + str(len(rank_list))
        

        # print "USER LIST"
        # print rank_list
        
        E = set(rank_list[:len(D)])
        # print "E HERE"
        # print E

        TP = (float)(len(D.intersection(E)))/len(D)
        self.TP = TP
            # q = q + 0.5

        # print "THIS IS IDEAL RATIO: " + str(q-1)

        return self.TP 
    
    
data = data_handler("rating_with_timestamp.mat", "trust.mat", "epinion_trust_with_timestamp.mat")
data = data.load_matrices()
print "LOADED MATRICES"
d = data[2]
print "set d: " + str(d)
P = data[0] # 0 for testing, 1 for training 
print "set p" +  str(len(P))
G = data[1]
print "set G" + str(len(G))
G_original = data[3]
obj = Social_Status(G,P,d,1000,G_original)
print "MADE SS OBJECT"
obj.start_main()

    


            
