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

    


    def load_matrices(self):
        #Loading matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        f3 = open(self.time_path)
        P_initial = loadmat(f1) #user-rating matrix
        G_raw = loadmat(f2) #trust-trust matrices
        time_data = loadmat(f3)
        time_data = time_data['trust']

        f1.close()
        f2.close()
        f3.close()

        time_data_dict = {}

        for x in time_data:
            pair = (x[0],x[1])
            if pair not in time_data_dict:
                time_data_dict[pair] = x[2]
            else:
                if time_data_dict[pair] > x[2]:
                    time_data_dict[pair] = x[2]

        time_data = time_data_dict
        print len(time_data)
        time_data = OrderedDict(sorted(time_data.items(), key=lambda kv: kv[1], reverse=False))
        

        P_initial = P_initial['rating_with_timestamp']
        G_raw = G_raw['trust']
    

        self.n = max(P_initial[:,0]) 
       
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
            if pair[0] == pair[1] :
                # if pair[0] in deleted_user or pair[1] in deleted_user:
                del time_data_final[pair]
            
        print "INITIAL TRUST PAIRS: "
        print len(G_raw)
        print "TOTAL TRUST PAIRS NOW CONSIDERED: " 
        print len(time_data_final)
        print ""
        print "CALCING O"
        total_relations = len(time_data_final.keys())
        amount_TP = int(total_relations* 0.5)
        O = time_data_final.keys()[:amount_TP] #old pairs

        print "CALCING G"

        #constructing G and setting everything in N to 0
        G_needed = np.zeros((self.n,self.n))
        for row in O:
            # if (row[0],row[1]) in N:
            #     continue
            G_needed[row[0]-1,row[1]-1] = 1


        #SET TEST VALUE HEREEE

        test_value = self.n * 0.05
        
        print test_value
        G_needed = G_needed[:test_value]
        G_needed = np.array([x[:test_value] for x in G_needed])

        # print "making time thing"
        #dictionary from user pairs to time of established relation
        for pair in time_data:
            if pair in time_data_final:
                if pair[0] > test_value or pair[1] > test_value:
                    del time_data_final[pair]


        time_data_final = OrderedDict(sorted(time_data_final.items(), key=lambda kv: kv[1], reverse=False))

    
        time_data_final = np.array(time_data_final.keys())
        print "ACTUAL TRUST PAIRS AFTER DELETING SELF LOOPS"
        print len(time_data_final)


        P = P[:test_value]


        return [P, G_needed, 5, time_data_final]

# data = data_handler("rating_with_timestamp.mat", "trust.mat", "epinion_trust_with_timestamp.mat")
# data.load_matrices()








