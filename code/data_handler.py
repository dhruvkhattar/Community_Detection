from __future__ import division
import sys
import pdb
import numpy as np

class data_handler():

    def __init__(self, rating_path, trust_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.n_users = 0
        self.n_items = 0
        self.n_trusts = 0
        self.n_ratings = 0

    def get_stats(self):
        return self.n_users, self.n_items, self.n_trusts, self.n_ratings

    def load(self):
        users = []
        trusts = []
        with open(self.trust_path) as f:
            for line in f:
                line = line.split()
                if line[0] not in users:
                    users.append(line[0])
                if line[1] not in users:
                    users.append(line[1])
                trusts.append((line[0],line[1]))

        print "Trust Matrix completed"
        users.sort()
        
        cnt = 0
        du = {}
        for user in users:
            du[user] = cnt
            cnt += 1

        items = []
        ratings = []
        with open(self.rating_path) as f:
            for line in f:
                line = line.split()
                if line[0] not in users:
                    users.append(line[0])
                if line[1] not in items:
                    items.append(line[1])
                ratings.append((line[0],line[1],line[2]))
       
        self.n_users = len(users)
        self.n_items = len(items)
        self.n_trusts = len(trusts)
        self.n_ratings = len(ratings)

        print "Rating Matrix completed"
        print "Number of Users:", self.n_users
        print "Number of Items:", self.n_items
        print "Number of Trust Relations:", self.n_trusts
        print "Number of Ratings:", self.n_ratings
        
        items.sort()
        cnt = 0
        di = {}
        for item in items:
            di[item] = cnt
            cnt += 1
        
        trust_mat = np.zeros((len(users), len(users)))
        for u, v in trusts:
            trust_mat[du[u],du[v]] = 1
        
        rat_mat = np.zeros((len(users), len(items)))
        for u, i, r in ratings:
            rat_mat[du[u],di[i]] = r

        return trust_mat, rat_mat

if __name__ == "__main__":
    data = data_handler("../data/rating.txt", "../data/trust.txt")
    trust, rating = data.load()
