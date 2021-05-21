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
        x_prev = 0
        for i in range(10):
            x = random.rand()
            if (i >= 1):
                y = network.feedforward([x])[0]
                err += np.abs(x_prev-y)
            x_prev = x
        return 0-err
