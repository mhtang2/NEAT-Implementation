from Net import Network
import Net
import numpy as np
from numpy import random


class Environment():
    def evaluate(network):
        pass


class XOR_Env(Environment):
    def evaluate(network):
        err = 0
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                y = network.feedforward([x1, x2])[0]
                err += np.abs((x1 ^ x2) - y)
        return 4.0 - err


class MEMORY_env(Environment):
    def evaluate(network):
        err = 0
        for i in range(10):
            y = network.feedforward([i])[0]
            if i > 0:
                err += np.abs((i-1)-y)
        return max(0.01, 10-err)
