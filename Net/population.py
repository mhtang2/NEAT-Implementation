from .counter import Counter
from .node import Node
from .edge import Edge
from .network import Network, tanh
import numpy as np
import numpy.random as random
import math

EDGE_MUTATION_RATE = 0.05
ADD_EDGE_MUTATION_RATE = 0.05
ADD_NODE_MUTATION_RATE = 0.03
MUTATION_STRENGTH = 1

ADD_NODE_MUTATION_NUMBER = 5
ADD_EDGE_MUTATION_NUMBER = 5

NODE_ABLENESS_MUTATION_RATE = 0  # 0.01
EDGE_ABLENESS_MUTATION_RATE = 0  # 0.05

MAX_SPECIES_DIFF = 0.8
DIST_C1 = 1
DIST_C2 = 1
DIST_C3 = 1


ELITE_PERCENTAGE = 0.20
MAX_POPULATION = 100
GRACE_PERIOD = 10

PERFECT_FITNESS = 250


def crossover(net1: "Network", net2: "Network") -> "Network":
    newNet = Network(net1.numInputs, net1.numOutputs,
                     net1.numRNN, empty=True)

    # Add Nodes to new net
    added = {}
    for node in net1.nodes:
        newNode = node.copyConstructor()
        newNet.nodes.append(newNode)
        added[node.innv] = newNode

    for node in net2.nodes:
        if node.innv not in added:
            newNode = node.copyConstructor()
            newNet.nodes.append(newNode)
            added[node.innv] = newNode

    # Add Edges to new net
    edgeNum1 = 0
    edgeNum2 = 0
    while edgeNum1 < len(net1.edges) or edgeNum2 < len(net2.edges):
        if edgeNum1 == len(net1.edges):
            # Helper copy net2[edge2Num]
            newEdge = net2.edges[edgeNum2].copyEdge(added)
            edgeNum2 += 1
        elif edgeNum2 == len(net2.edges):
            # Helper copy net1[edge1Num]
            newEdge = net1.edges[edgeNum1].copyEdge(added)
            edgeNum1 += 1
        else:
            if net1.edges[edgeNum1].innv < net2.edges[edgeNum2].innv:
                newEdge = net1.edges[edgeNum1].copyEdge(added)
                edgeNum1 += 1
            elif net1.edges[edgeNum1].innv > net2.edges[edgeNum2].innv:
                newEdge = net2.edges[edgeNum2].copyEdge(added)
                edgeNum2 += 1
            else:
                newEdge = net2.edges[edgeNum2].copyEdge(added)
                edgeNum1 += 1
                edgeNum2 += 1

        if random.random() < EDGE_MUTATION_RATE:
            newEdge.weight = newEdge.weight + \
                (random.normal() * MUTATION_STRENGTH)

        newNet.edges.append(newEdge)
    for i in range(random.binomial(ADD_NODE_MUTATION_NUMBER, NODE_ABLENESS_MUTATION_RATE)):
        newNet.mutate_node_ableness()

    for i in range(random.binomial(ADD_EDGE_MUTATION_NUMBER, EDGE_ABLENESS_MUTATION_RATE)):
        newNet.mutate_edge_ableness()

    # Pick number of new nodes to muate using a binomial distribution
    for i in range(random.binomial(ADD_NODE_MUTATION_NUMBER, ADD_NODE_MUTATION_RATE)):
        newNet.mutate_add_node()

    # Pick number of new edges to muate using a binomial distribution
    for i in range(random.binomial(ADD_EDGE_MUTATION_NUMBER, ADD_EDGE_MUTATION_RATE)):
        newNet.mutate_add_edge()

    return newNet


class Species:
    def __init__(self, initialSpecies=[]):
        self.nets = initialSpecies
        self.fitnessList = [0]*len(initialSpecies)
        self.age = 0
        self.fitnessSum = 0

    def size(self):
        return len(self.nets)

    def updateFitnessSum(self):
        self.fitnessSum = sum(self.fitnessList)
        return self.fitnessSum


class Population:
    def __init__(self, initialPopulation, numInputs, numOutputs, numRNN, environment):
        self.population = [Species([Network(numInputs, numOutputs, numRNN)
                                    for _ in range(initialPopulation)])]
        self.environment = environment
        print(self.population)

    def getCurrentPop(self):
        x = 0
        for species in self.population:
            x += species.size()
        return x

    def compatibilityDistance(self, net1: "Network", net2: "Network"):
        E, D, W, W_n = 0, 0, 0, 0

        edgeNum1 = 0
        edgeNum2 = 0
        while edgeNum1 < len(net1.edges) or edgeNum2 < len(net2.edges):
            if edgeNum1 == len(net1.edges):
                E += 1
                edgeNum2 += 1
            elif edgeNum2 == len(net2.edges):
                E += 1
                edgeNum1 += 1
            else:
                if net1.edges[edgeNum1].innv < net2.edges[edgeNum2].innv:
                    D += 1
                    edgeNum1 += 1
                elif net1.edges[edgeNum1].innv > net2.edges[edgeNum2].innv:
                    D += 1
                    edgeNum2 += 1
                else:
                    W += np.abs(net1.edges[edgeNum1].weight -
                                net2.edges[edgeNum2].weight)
                    W_n += 1
                    edgeNum1 += 1
                    edgeNum2 += 1
        W = W/W_n if W_n != 0 else 0
        N = max(len(net1.edges), len(net2.edges))
        return (DIST_C3 * W) + ((DIST_C1 * E + DIST_C2 * D)/N if N != 0 else 0)

    def fitInSpecies(self, net, species):
        """Checks if a network belongs in a species"""
        avgGeneticDistance = 0

        for netToCompare in species.nets:
            avgGeneticDistance += self.compatibilityDistance(net, netToCompare)
        avgGeneticDistance = avgGeneticDistance / \
            species.size() if species.size() != 0 else 0
        if (avgGeneticDistance) < MAX_SPECIES_DIFF:
            return True
        else:
            return False

    def addToPopulation(self, net):
        """Add a network to the population in the correct species"""
        for species in self.population:
            if self.fitInSpecies(net, species):
                species.nets.append(net)
                species.fitnessList.append(0)
                return 0
        self.population.append(Species([net]))
        return 1

    def eliminateWorstPerforming(self, species: Species, numEliminate):
        numEliminate = math.floor(min(species.size()-2, numEliminate))
        if numEliminate <= 0:
            # TODO:  FIGURE OUT WHY NUMELIMINATE IS NEGATIVE
            return
        idxToAdd = np.argpartition(np.array(species.fitnessList), numEliminate)
        idxToAdd = idxToAdd[numEliminate:]
        # print("Left after elim ", len(idxToAdd))

        # Calculate elite fitness: Can delete
        newFitnessList = [species.fitnessList[idx] for idx in idxToAdd]
        pctOfPop = len(newFitnessList)/(numEliminate+len(newFitnessList))
        print("Elite Fitness ", sum(newFitnessList)/pctOfPop)

        newList = [species.nets[idx] for idx in idxToAdd]
        species.nets = newList
        species.fitnessList = [0]*len(newList)

    def test(self, topN=5):
        networkList = []
        fitnessList = []
        for species in self.population:
            for net in species.nets:
                fitnessList.append(self.environment.eval_train(net))
                networkList.append(net)
        idxToAdd = np.argpartition(
            np.array(fitnessList), len(networkList)-topN)
        idxToAdd = idxToAdd[len(networkList)-topN:]

        fitnessList = [fitnessList[idx] for idx in idxToAdd]
        print(fitnessList)
        networkList = [networkList[idx] for idx in idxToAdd]

        for net in networkList:
            print("Elite Network Test: ", self.environment.eval_test(net))

    def run(self):
        for species in self.population:
            numPerfect = 0
            for netNum in range(species.size()):
                fitness = self.environment.eval_train(species.nets[netNum])
                if fitness >= PERFECT_FITNESS:
                    numPerfect += 1
                species.fitnessList[netNum] = fitness / species.size()
            # print(f"Found {numPerfect} perfect members")
        totalFitness = 0  # Total fitness of population
        for species in self.population:
            totalFitness += species.updateFitnessSum()

        # Kill lowest performing members of species
        currentPop = self.getCurrentPop()
        totalEliminate = currentPop - (ELITE_PERCENTAGE * MAX_POPULATION)
        for species in self.population:
            # numToEliminate: function of current pop, and max pop
            # eliminate such that 5 of max pop remains
            # totalEliminate = currentPop - 5% * max pop
            # speciesEliminate = (len(species)/currentPop) * totalEliminate
            numToEliminate = (species.size()/currentPop) * totalEliminate
            self.eliminateWorstPerforming(species, numToEliminate)
        # Reproduce
        numDied = 0
        numNew = 0
        for speciesNum in range(len(self.population)-1, -1, -1):
            species = self.population[speciesNum]
            numChildren = int((species.fitnessSum /
                               totalFitness) * MAX_POPULATION)
            if species.age < GRACE_PERIOD:
                numChildren = max(numChildren, 2)
            if numChildren < 2:
                numChildren = 0
                del self.population[speciesNum]
                numDied += 1
            for childNum in range(numChildren):
                # Pair two random surviving parents
                parent1Num = random.randint(species.size())
                parent2Num = random.randint(species.size())
                child = crossover(species.nets[parent1Num],
                                  species.nets[parent2Num])
                numNew += self.addToPopulation(child)
            species.age += 1
