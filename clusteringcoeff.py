from __future__ import division
from collections import defaultdict
import math
import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab

testingFile = "networkDatasets//karate.txt"
Xtrain = numpy.loadtxt(testingFile, dtype=int)

print(Xtrain.shape[0])
print(Xtrain.shape[1]-1)

lastEl = Xtrain[len(Xtrain)-1][0]
maxEl = (max(Xtrain.max(axis=0)[0], Xtrain.max(axis=0)[1]))

print("Number of Nodes in graph: {}".format(maxEl))

print("Number of Elements in file: {}".format(len(Xtrain)))

adjMatrix = [[0 for x in range(maxEl+1)] for y in range(maxEl+1)]

for x in range(maxEl):
    for y in range(maxEl):
        adjMatrix[y][x] = 0

count = 0

degreeDict = defaultdict(list)

for x in range(0, maxEl):
    if count < len(Xtrain)-1:
        while(Xtrain[count][0]==x):
            adjMatrix[Xtrain[count][0]][Xtrain[count][1]] = 1
            adjMatrix[Xtrain[count][1]][Xtrain[count][0]] = 1
            count+=1

totalCoeff = 0.0

for node in range(1, maxEl):
        localCC = 0.0
        degree = 0
        triangles = 0
        neighbors = []

        for friends in range(maxEl):
            if(adjMatrix[node][friends] == 1):
              neighbors.append(friends)
              degree+=1

        if (degree > 1):
            degreeDict[degree].append(node)
            for y in range(len(neighbors)):
                pal1 = neighbors[y]
                for x in range(y+1, len(neighbors)):
                    pal2 = neighbors[x]
                    if adjMatrix[pal1][pal2] == 1:
                        triangles+=1

            localCC = (2 * triangles) / (degree * (degree - 1))
            totalCoeff += localCC

totalCC = totalCoeff/maxEl
print("Total Coefficient: {}".format(totalCC))

lengthOfDegree = dict()

for x in degreeDict.keys():
    lengthOfDegree[x] = len(degreeDict[x])

print(lengthOfDegree)
probDict = dict()

for x in degreeDict.keys():
    sum = len(degreeDict[x])
    prob = sum/maxEl
    probDict[x] = prob
    print("Degree of {} has {} member(s) and a Probability of {}".format(x, sum, prob))

plt.loglog(sorted(probDict.keys()), probDict.values(), basex=2, basey=2)
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree Probability Plot")
plt.show()

plt.scatter(sorted(lengthOfDegree.keys()), lengthOfDegree.values())
plt.xlabel("Degree")
plt.ylabel("Number of Nodes w/ Degree")
plt.title("Degree Distribution")
plt.show()