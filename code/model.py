from __future__ import division
import numpy as np
from data_handler import data_handler
import pdb
import math
import copy

class CD():

    def __init__(self):
        
        # Load trust and rating matrices
        data = data_handler("../data/rating.txt", "../data/trust.txt")
        self.trusts, self.ratings = data.load()
        print "Initializing"
        self.n_users, self.n_items, self.n_trusts, self.n_ratings = data.get_stats()
        self.n = self.n_users
        self.m = self.n_trusts
        self.agents = []
        self.W = np.zeros((self.n_users, self.n_users))
        self.T = np.zeros((self.n_users, self.n_users))
        self.R = np.zeros((self.n_users, self.n_users))
        self.n_it = 5 * self.n
        self.communities = np.arange(self.n_users)
        self.Labels = []
        self.alpha = 0.5

        # Making agents where each agent has its own label
        for i in xrange(self.n_users):
            a = agent()
            # Each user has its own label
            a.L.append(i)
            self.Labels.append([i])
            a.deg = sum(self.trusts[:,i]) + sum(self.trusts[i,:])
            a.neighbors = np.union1d(np.where(self.trusts[i, :] == 1)[0], np.where(self.trusts[:, i] == 1)[0])
            self.agents.append(a)

        # Calculating Wij i.e. the number of common neighbors i and j have
        for i in xrange(self.n_users):
            for j in xrange(i, self.n_users):
                self.W[j, i] = self.W[i, j] = np.intersect1d(self.agents[i].neighbors, self.agents[j].neighbors).size


    def model(self):
        print "Calculating T"
        self.trustSimilarity()
        print "Calculating R"
        self.ratingSimilarity()
        print "Game started"
        self.game()
        pdb.set_trace()


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
                    self.R[i, j] = self.R[j, i] = temp/(temp1*temp2)
                else:  
                    self.R[i, j] = self.R[j, i] = 0

    def game(self):
        for i in xrange(self.n_it):
            if i%20 == 0:
                print i
            # Choosing current agent
            current_agent = i%self.n

            # Calculating Max Utility for current agent by joining communities
            join_utility = 0
            join_community = -1
            for community in self.communities:
                if community not in self.agents[current_agent].L:
                    temp = -1
                    for j in self.Labels[community]:
                        temp += ((self.alpha * T[current_agent,j]) + ((1 - self.alpha) * R[current_agent,j]))
                    temp /= self.m
                    if temp > join_utility:
                        join_utility = temp
                        join_community = community

            # Calculating Max Utility for current agent by leaving communities
            leave_utility = 0
            leave_community = -1
            for community in self.agents[current_agent].L:
                temp = 1
                for j in self.Labels[community]:
                    temp -= ((self.alpha * T[current_agent,j]) + ((1 - self.alpha) * R[current_agent,j]))
                temp /= self.m
                if temp > leave_utility:
                    leave_utility = temp
                    leave_community = community
            
            # Calculating Max Utility for current agent by switching communities
            switch_utility = 0
            switch_communities = (-1,-1)
            for lcommunity in self.agents[current_agent].L:
                for jcommunity in self.communities:
                    temp = 0
                    for j in self.Labels[jcommunity]:
                        temp += ((self.alpha * self.T[current_agent,j]) + ((1 - self.alpha) * self.R[current_agent,j]))
                    for j in self.Labels[community]:
                        temp -= ((self.alpha * self.T[current_agent,j]) + ((1 - self.alpha) * self.R[current_agent,j]))
                    temp /= self.m
                    if temp > switch_utility:
                        switch_utility = temp
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


class agent():

    def __init__(self):
        self.deg = 0
        self.L = []


if __name__ == "__main__":
    cd = CD()
    cd.model()
