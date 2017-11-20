import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
import pandas as pd
import itertools
import random

#Retrieve instances
class TSP:
    '''Class for traveling salesman problems.

    Can load TSP data (Padberg/Rinaldi) from text files.'''

    def __init__(self, path, filename):  # initialize TSP object
        self.name = filename
        coord = self.read_coordinates(path + filename)
        self.distMatrix = self.get_distance_matrix(coord)

    def __str__(self):
        return "Instance: " + self.name + "\n" + "Distance:\n" + str(self.distMatrix) + "\n"

    def read_coordinates(self, inputfile):
        coord = []
        iFile = open(inputfile, "r")
        for i in range(6):  # skip first 6 lines
            iFile.readline()
        line = iFile.readline().strip()
        while line != "EOF":
            values = line.split()
            coord.append([float(values[1]), float(values[2])])
            line = iFile.readline().strip()
        iFile.close()
        return coord

    def get_distance_matrix(self, coord):
        distMatrix = squareform(pdist(coord, "euclidean"))  # returns an array
        return distMatrix

#Create operaters
class TSP_solution:
    """Class for representing a solution to the TSP.

    Used to manage the solution itself."""

    def __init__(self, data):
        """
        TSPdata(TSP): object holding all necessary TSP data.
        objective(float): total distance of the route
        """
        self.data = data
        self.route = []
        self.objective = 0

    def __str__(self):
        ostring = "----------------------"  # initialize empty string
        ostring += "\nSolution of:\t" + self.data.name
        ostring += "\nTotal distance:\t" + str(self.objective)
        ostring += "\nRoute:\t" + str(self.route)
        return ostring

    def nearest_neighbor(self):
        """Function solving the TSP using Nearest-Neighbor-heuristic."""
        d = self.data.distMatrix
        route = [0]  # tour starts at the depot (= node 0)
        position = 0  # starting (=current) position is depot
        numNodes = len(d)  # number of nodes to be visited
        notVisited = list(range(1, numNodes))  # creates a list of consecutive numbers from 1 to numNodes

        while len(notVisited) != 0:
            bestDistance = 1e+300  # initially set best to an arbitrarily high number
            for j in notVisited:  # check all customers not yet visited
                if (d[position][j] < bestDistance):  # find nearest customer
                    bestDistance = d[position][j]  # if true update best value & best index
                    bestPosition = j
            position = bestPosition  # update current position
            notVisited.remove(bestPosition)
            route.append(bestPosition)
        route.append(0)  # add depot at the end
        self.route = route.copy()  # save NN solution
        self.objective = self.get_total_distance(route)  # update objective
        return self.objective, self.route

    def random_route(self):
        numNodes = len(self.data.distMatrix)
        self.route = [0] + random.sample(list(range(1, numNodes)), numNodes - 1) + [
            0]  # create random route
        self.objective = self.get_total_distance(self.route)  # compute objective
        return self.objective, self.route

    def two_opt(self, route, strategy="best"):
        """finds the best/better neighbor of a route according to 2-opt which is an edge-oriented operator.

        The 2-opt removes two edges and reinserts them crosswise. The function returns the best/better neighbor."""
        bestdist = self.get_total_distance(
            route)  # bestdist = incumbent_obj --> decrease in solution quality is NOT possible
        # bestdist = 1e+300   --> assigning a very high value for initial best --> decrease in solution quality is possible
        bestr = route.copy()
        for i in range(0, len(route) - 3):
            for j in range(i + 2, len(route) - 1):
                newr = route[:i + 1] + route[j:i:-1] + route[j + 1:]
                newdist = self.get_total_distance(newr)
                if newdist < bestdist:
                    bestr = newr.copy()
                    bestdist = newdist
                    if strategy == "first":
                        return bestdist, bestr
        return bestdist, bestr

    def local_search(self):
        incumbent_obj, incumbent_route = self.nearest_neighbor()  # create initial solution = incumbent solution
        # print(mysol)
        # print("-----------------------------")
        improvement = True  # loop while there is an improvement
        while improvement:
            new_obj, new_route = self.two_opt(incumbent_route, "best")
            if new_obj < incumbent_obj:
                incumbent_obj = new_obj
                incumbent_route = new_route.copy()
                print(incumbent_obj)
            else:
                improvement = False
        self.objective = incumbent_obj
        self.route = incumbent_route.copy()
        return (self.objective, self.route)

    def get_total_distance(self, route):
        """Function calculating the total travel distance of a TSP for a given sequence"""
        d = self.data.distMatrix
        total = 0
        for i in range(len(route) - 1):
            total += d[route[i]][route[i + 1]]
        return round(total, 2)

    def random_ksubset(self, k):
        d = self.data.distMatrix
        inc_sol_objective, inc_sol_route = self.local_search()
        numNodes = len(inc_sol_route)
        ls = list(range(1, numNodes))
        if k < 1 or k > numNodes: # sanity check
            return [] # Create a list of length ls, where each element is the index of the subset that the corresponding member of ls will be assigned to.
        indices = list(range(k)) # We require that this list contains k different values, so we start by adding each possible different value.
        indices.extend([random.choice(list(range(k))) for _ in range(numNodes - k)]) # add random values from range(k) to indices to fill it up to the length of ls
        random.shuffle(indices)  # shuffle the indices into a random order
        N = [{x[1] for x in xs} for (_, xs) in itertools.groupby(sorted(zip(indices, ls)), lambda x: x[0])]# construct and return the random subset: sort the elements by which subset they will be assigned to, and group them into sets
        return N

    def random_x(self, k):
        N = self.random_ksubset(k)
        route = [0]
        for i in N:
            i = list(i)
            #print('1 subset',i)
            for j in i:
                x = random.shuffle(i)
                x_dist = self.get_total_distance(x)
                y = self.local_search(x)
                y_dist = self.get_total_distance(y)
                if y_dist < x_dist:
                    y == x
                    route.append(y)
                else:
                    i.extend(i)
        route.append(0)
        best_route = route.copy()
        best_distance = self.get_total_distance(best_route)
        return (best_route, best_distance)

#Create TSP data object
mytsp = TSP("TSP_instances/", "pr76.tsp.txt")  # create TSP data object providing the TSP data
mysol = TSP_solution(mytsp)                # create TSP_solution object providing functions for solving TSPs

#VNS Alg
neighbors = mysol.random_x(10)
timelim = 60
cpu_time = 0
start_time = time.time()
list_incumb_obj = []
iteration = 0
while cpu_time < timelim:
    mysol.random_x(10)
    cpu_time = (time.time() - start_time)
    iteration += 1

