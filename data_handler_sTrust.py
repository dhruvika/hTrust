import numpy as np
from scipy.io import loadmat
import collections
import math
from collections import OrderedDict



class data_handler():

    def __init__(self,rating_path,trust_path,time_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.time_path = time_path
        self.n = 0
        self.k = 0
        self.d = 0

    # def get_stats(self):
    #     return self.n_users, self.n_prod, self.n_cat


        #BRING G BACK BEFORE RUNNING MAIN--------------------

    def load_matrices(self):
        #Loading matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        f3 = open(self.time_path)
        P_initial = loadmat(f1) #user-rating matrix
        G_raw = loadmat(f2) #trust-trust matrices
        time_data = loadmat(f3)
        time_data = time_data['trust']
        print len(time_data)
        # d = {n: True for n in range(5)}
        # time_data = {(x[0],x[1]):x[2] for x in time_data}
        time_data_dict = {}
        
        for x in time_data:
            pair = (x[0],x[1])
            if pair not in time_data_dict:
                time_data_dict[pair] = x[2]
            else:
                if time_data_dict[pair] > x[2]:
                    time_data_dict[pair] = x[2]
        time_data = time_data_dict
        

        # print "TIME DATA"
        # print time_data
        #G_raw = np.array([])
        P_initial = P_initial['rating_with_timestamp']
        G_raw = G_raw['trust']
        print "SHAPE"
        print G_raw.shape
        print len(time_data)

        self.n = max(P_initial[:,0]) 
        print "N: " + str(self.n)
        self.k = max(P_initial[:,1])  
        self.d = max(P_initial[:,2]) 
        P = np.zeros((self.n,self.k))


        for row in P_initial:
            i = row[0] -1
            k = row[1] -1
            P[i,k] = row[3]

        P_size = P.shape[0]

        time_data_final = dict(time_data)
        for pair in time_data:
            if pair[0] == pair[1]:
                del time_data_final[pair]
       
        print "CALCULATING O"
        # total_relations = len(time_data_final.keys())
        # amount_TP = int(total_relations* 0.5)
        # O = time_data_final.keys()[:amount_TP]

        G_needed = np.zeros((self.n,self.n))

        for row in time_data_final:
            G_needed[row[0]-1,row[1]-1] = 1



        test_value = self.n * 1.
        print test_value
        G_needed = G_needed[:test_value]
        G_needed = np.array([x[:test_value] for x in G_needed])

        #dictionary from user pairs to time of established relation
        # time_data_final = dict(time_data)
        for pair in time_data:
            # if pair[0] == pair[1]:
            #     del time_data_final[pair]
            if pair in time_data_final:
                if pair[0] >= test_value or pair[1] >= test_value:
                    del time_data_final[pair]

        
        #sort based on time value
        time_data_final = OrderedDict(sorted(time_data_final.items(), key=lambda kv: kv[1], reverse=False))

    
        time_data_final = np.array(time_data_final.keys())



        #remove for actual run--------------

        # for (x,y) in np.ndenumerate(G_needed):
        #     [i,j] = [x[0],x[1]]
        #     if G_needed[i,j] == 1:
        #         G_raw.append([i+1,j+1])
        # G_raw = np.array(G_raw)
        

        #-----------------------------

        P = P[:test_value]
        # print "THIS IS TRUST MATRIX"
        # print G_needed
        # print len(G_needed)
        # print len(P)


        return [P, G_needed, self.d, time_data_final]

# data = data_handler("rating_with_timestamp.mat", "trust.mat", "epinion_trust_with_timestamp.mat")
# data.load_matrices()


#TODO - get P matrix from data (iXk matrix)






