from .counter import Counter
from .node import Node
from .edge import Edge
from .network import Network
import numpy as np
import numpy.random as random

MAX_SPECIES_DIFF = 1
DIST_C1 = 1
DIST_C2 = 1
DIST_C3 = 1

# INPUTS -> OUTPUTS


class Population:
    def __init__(self, initialPopulation, numInputs, numOutputs, numRNN, environment, activation=None):
        self.networks = [[Network(numInputs, numOutputs, numRNN, activation)
                         for _ in range(initialPopulation)]]
        self.environment = environment
        print(self.networks)

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
        avgGeneticDistance = 0

        for netToCompare in species:
            avgGeneticDistance += self.compatibilityDistance(net, netToCompare)
        avgGeneticDistance = avgGeneticDistance / \
            len(species) if len(species) != 0 else 0
        if (avgGeneticDistance) < MAX_SPECIES_DIFF:
            return True
        else:
            return False

    def addToPopulation(self, net):

        for species in self.population:
            if self.fitInSpecies(net, species):
                species.append(net)
                return

        self.population.append([net])

    def elimnateWorstPerforming(self, speciesList, speciesFitnessList, numEliminate):
        # TODO: Write test cases!!!
        numEliminate = min(len(speciesList)-2, numEliminate)
        idxToRemove = np.partition(np.array(speciesFitnessList), numEliminate)[
            :numEliminate]
        for idx in idxToRemove:
            del speciesList[idx]

    def run(self, epochs):
        fitnessList = []
        for species in self.networks:
            speciesFitnessList = []
            for net in species:
                fitness = self.environment.evaluate(net) / max(1, len(species))
                speciesFitnessList.append(fitness)
            fitnessList.append(speciesFitnessList)
        # Kill lowest performing members of species
        # Assign babies based on sum of adjusted fitness
        # Reproduce
