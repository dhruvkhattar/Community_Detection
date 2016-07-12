from __future__ import division
import numpy as np
from data_handler import data_handler
import pdb
import math
import copy
import networkx as nx
import random
import sys

class CD():

    def __init__(self):
        
        print "Code started"
        # Load trust and rating matrices
        data = data_handler("../data/rating.txt", "../data/trust.txt")
        self.trusts, self.ratings = data.load()
        print "Initializing"
        self.n_users, self.n_items, self.n_trusts, self.n_ratings = data.get_stats()
        self.n = self.n_users
        self.m = self.n_trusts
        self.agents = []
        self.W = np.zeros((self.n_users, self.n_users), dtype = np.float32)
        self.T = np.zeros((self.n_users, self.n_users), dtype = np.float32)
        self.R = np.zeros((self.n_users, self.n_users), dtype = np.float32)
        self.n_it = 5 * self.n
        self.communities = np.arange(self.n_users)
        self.Labels = []
        self.alpha = 0.5
        self.centers1 = []
        self.centers2 = []
        self.centers3 = []
        self.centers4 = []
        self.centers5 = []
        self.centers6 = []

        # Making agents where each agent has its own label
        for i in xrange(self.n_users):
            a = agent()
            # Each user has its own label
            a.L.append(i)
            self.Labels.append([i])
            a.trustor = sum(self.trusts[:, i])
            a.trustee = sum(self.trusts[i, :])
            a.deg = a.trustee + a.trustor
            a.neighbors = np.union1d(np.where(self.trusts[i, :] == 1)[0], np.where(self.trusts[:, i] == 1)[0])
            self.agents.append(a)

        # Calculating Wij i.e. the number of common neighbors i and j have
        for i in xrange(self.n_users):
            for j in xrange(i, self.n_users):
                self.W[j, i] = self.W[i, j] = np.intersect1d(self.agents[i].neighbors, self.agents[j].neighbors).size


    def model(self):
        print "Calculating T"
        #self.trustSimilarity()
        self.T = np.load('T.npy')
        print "Calculating R"
        #self.ratingSimilarity()
        self.R = np.load('R.npy')
        #np.save("T", self.T)
        #np.save("R", self.R)
        print "Files Saved"
        del self.W
        print "Game started"
        self.game()
        np.save("communities", self.Labels)
        print "Detecting Centers"
        print >> sys.stderr, "Detecting Centers"
        self.detectCenters()
        np.save("centers", self.centers1)
        self.x = 50
        print >> sys.stderr, "Calculating Accuracy ", self.x
        self.calculateAccuracy()


    def trustSimilarity(self):
        for i in xrange(self.n_users):
            for j in xrange(i, self.n_users):
                if self.trusts[i, j] or self.trusts[j, i]:
                    if self.W[i, j]:
                        self.T[i, j] = self.T[j, i] = (self.W[i, j] * (1 - ((self.agents[i].deg * self.agents[j].deg) / (2*self.m))))
                    else:
                        self.T[i, j] = self.T[j, i] = ((self.agents[i].deg * self.agents[j].deg) / (2*self.m))
                else:
                    if self.W[i, j]:
                        self.T[i, j] = self.T[j, i] = (self.W[i, j] / self.n)
                    else:
                        self.T[i, j] = self.T[j, i] = -((self.agents[i].deg * self.agents[j].deg) / (2*self.m))


    def ratingSimilarity(self):
        for i in xrange(self.n_users):
            for j in xrange(i, self.n_users):
                temp = np.dot(self.ratings[i, :], self.ratings[j, :])
                temp1 = math.sqrt(np.dot(self.ratings[i, :], self.ratings[i, :]))
                temp2 = math.sqrt(np.dot(self.ratings[j, :], self.ratings[j, :]))
                if temp1 and temp2:
                    self.R[i, j] = self.R[j, i] = temp / (temp1 * temp2)
                else:  
                    self.R[i, j] = self.R[j, i] = 0


    def game(self):
        for i in xrange(self.n_it):
            print >> sys.stderr, self.alpha, i
            # Choosing current agent
            current_agent = i%self.n

            # Calculating Max Utility for current agent by joining, leaving or switching communities
            join_utility = 0
            join_community = -1
            leave_utility = 0
            leave_community = -1
            switch_utility = 0
            switch_communities = (-1,-1)

            for lcommunity in self.agents[current_agent].L:
                if lcommunity != current_agent:
                    temp1 = 1
                    for j in self.Labels[lcommunity]:
                        temp1 -= ((self.alpha * self.T[current_agent,j]) + ((1 - self.alpha) * self.R[current_agent,j]))
                    temp1 /= self.m
                    if temp1 > leave_utility:
                        leave_utility = temp1
                        leave_community = lcommunity
                for jcommunity in self.communities:
                    if jcommunity not in self.agents[current_agent].L:
                        temp3 = 0
                        temp2 = -1
                        for j in self.Labels[jcommunity]:
                            temp2 += ((self.alpha * self.T[current_agent,j]) + ((1 - self.alpha) * self.R[current_agent,j]))
                        temp2 /= self.m
                        if temp2 > join_utility:
                            join_utility = temp2
                            join_community = jcommunity
                        if lcommunity != current_agent:
                            temp3 = temp1 + temp2
                            if temp3 > switch_utility:
                                switch_utility = temp3
                                switch_communities = (lcommunity, jcommunity)

            # Checking which action gives us the maximum utility
            if join_utility == 0 and leave_utility == 0 and switch_utility == 0:
                # No action
                pass
            elif join_utility >= leave_utility and join_utility >= switch_utility:
                # Join community
                self.agents[current_agent].L.append(join_community)
                self.Labels[join_community].append(current_agent)
            elif leave_utility >= join_utility and leave_utility >= switch_utility:
                # Leave community
                self.agents[current_agent].L.remove(leave_community)
                self.Labels[leave_community].remove(current_agent)
            elif switch_utility >= leave_utility and switch_utility >= leave_utility:
                # Switch communities
                self.agents[current_agent].L.remove(switch_communities[0])
                self.Labels[switch_communities[0]].remove(current_agent)
                self.agents[current_agent].L.append(switch_communities[1])
                self.Labels[switch_communities[1]].append(current_agent)


    def detectCenters(self):
        for community in self.communities:
            if len(self.Labels[community]) < 1:
                self.centers1.append(-1)
                self.centers2.append(-1)
                self.centers3.append(-1)
                self.centers4.append(-1)
                self.centers5.append(-1)
                self.centers6.append(-1)
            else:
                self.G1 = nx.MultiGraph()
                self.G1.add_nodes_from(self.Labels[community])
                for agent in self.Labels[community]:
                    r = np.where(self.trusts[agent, :] > 0)[0]
                    self.G1.add_edges_from(zip(r, [agent] * len(r)))
                    r = np.where(self.trusts[:, agent] > 0)[0]
                    self.G1.add_edges_from(zip(r, [agent] * len(r)))
                eigen = nx.eigenvector_centrality_numpy(self.G1)
                betweenness = nx.betweenness_centrality(self.G1)
                center1 = center2 = center3 = center4 = center5 = center6 = self.Labels[community][0]
                max_betweenness = betweenness[center1]
                max_degree = self.agents[center2].deg
                max_trustee = self.agents[center3].trustee
                max_trustor = self.agents[center4].trustor
                max_eigen = eigen[center6]
                for agent in self.Labels[community]:
                    if betweenness[agent] >= max_betweenness:
                        center1 = agent
                        max_betweenness = betweenness[agent]
                    if self.agents[agent].deg >= max_degree:
                        center2 = agent
                        max_degree = self.agents[agent].deg
                    if self.agents[agent].trustee >= max_trustee:
                        center3 = agent
                        max_trustee = self.agents[agent].trustee
                    if self.agents[agent].trustor >= max_trustor:
                        center4 = agent
                        max_trustor = self.agents[agent].trustor
                    if eigen[agent] >= max_eigen:
                        center6 = agent
                        max_eigen = eigen[agent]
                center5 = random.choice(self.Labels[community])
                self.centers1.append(center1)
                self.centers2.append(center2)
                self.centers3.append(center3)
                self.centers4.append(center4)
                self.centers5.append(center5)
                self.centers6.append(center6)


    def predictTrust(self, i , j):
        val1 = 0
        val2 = 0
        val3 = 0
        val4 = 0
        val5 = 0
        val6 = 0
        for l1 in self.agents[i].L:
            for l2 in self.agents[j].L:
                temp1 = (self.R[i, self.centers1[l1]] + self.R[self.centers1[l1], self.centers1[l2]] + self.R[j, self.centers1[l2]]) / 3
                temp2 = (self.R[i, self.centers2[l1]] + self.R[self.centers2[l1], self.centers2[l2]] + self.R[j, self.centers2[l2]]) / 3
                temp3 = (self.R[i, self.centers3[l1]] + self.R[self.centers3[l1], self.centers3[l2]] + self.R[j, self.centers3[l2]]) / 3
                temp4 = (self.R[i, self.centers4[l1]] + self.R[self.centers4[l1], self.centers4[l2]] + self.R[j, self.centers4[l2]]) / 3
                temp5 = (self.R[i, self.centers5[l1]] + self.R[self.centers5[l1], self.centers5[l2]] + self.R[j, self.centers5[l2]]) / 3
                temp6 = (self.R[i, self.centers6[l1]] + self.R[self.centers6[l1], self.centers6[l2]] + self.R[j, self.centers6[l2]]) / 3
                if temp1 > val1:
                    val1 = temp1
                if temp2 > val2:
                    val2 = temp2
                if temp3 > val3:
                    val3 = temp3
                if temp4 > val4:
                    val4 = temp4
                if temp5 > val5:
                    val5 = temp5
                if temp6 > val6:
                    val6 = temp6
        return [val1, val2, val3, val4, val5, val6]
                 

    def calculateAccuracy(self):
        n_N = int((((100 - self.x) / 100 ) * self.n_trusts))
        r,c = np.where(self.trusts > 0)
        N = random.sample(zip(r,c), n_N)
        r,c = np.where(self.trusts == 0)
        B = random.sample(zip(r,c), 4 * n_N)
        BUN = B + N
        T1 = []
        T2 = []
        T3 = []
        T4 = []
        T5 = []
        T6 = []
        for u, v in BUN:
            ptrust = self.predictTrust(u,v)
            T1.append((ptrust[0], u, v))
            T2.append((ptrust[1], u, v))
            T3.append((ptrust[2], u, v))
            T4.append((ptrust[3], u, v))
            T5.append((ptrust[4], u, v))
            T6.append((ptrust[5], u, v))
        T1.sort()
        T1.reverse()
        T2.sort()
        T2.reverse()
        T3.sort()
        T3.reverse()
        T4.sort()
        T4.reverse()
        T5.sort()
        T5.reverse()
        T6.sort()
        T6.reverse()
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        cnt5 = 0
        cnt6 = 0
        for i in xrange(n_N):
            if (T1[i][1],T1[i][2]) in N:
                cnt1 += 1
            if (T2[i][1],T2[i][2]) in N:
                cnt2 += 1
            if (T3[i][1],T3[i][2]) in N:
                cnt3 += 1
            if (T4[i][1],T4[i][2]) in N:
                cnt4 += 1
            if (T5[i][1],T5[i][2]) in N:
                cnt5 += 1
            if (T6[i][1],T6[i][2]) in N:
                cnt6 += 1

        PA1 = cnt1 / n_N
        PA2 = cnt2 / n_N
        PA3 = cnt3 / n_N
        PA4 = cnt4 / n_N
        PA5 = cnt5 / n_N
        PA6 = cnt6 / n_N
        del BUN
        del N
        del B
        del T1
        del T2
        del T3
        del T4
        del T5
        del T6
        print "X = ", self.x, "Prediction Accuracy (Betweenness nx)= ", PA1
        print "X = ", self.x, "Prediction Accuracy (Max Degree)= ", PA2
        print "X = ", self.x, "Prediction Accuracy (Max Trustee)= ", PA3
        print "X = ", self.x, "Prediction Accuracy (Max Trustor)= ", PA4
        print "X = ", self.x, "Prediction Accuracy (Random)= ", PA5
        print "X = ", self.x, "Prediction Accuracy (eigen nx)= ", PA6

        
class agent():

    def __init__(self):
        self.deg = 0
        self.L = []


if __name__ == "__main__":
    cd = CD()
    cd.model()
