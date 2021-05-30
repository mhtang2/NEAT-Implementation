from Net import Network
import Net
import numpy as np
from numpy import random


class Environment():
    def eval_train(network):
        pass

    def eval_test(network):
        pass


class XOR_Env(Environment):

    def eval_train(network):
        err = 0
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                y = network.feedforward([x1, x2])[0]
                err += np.abs((x1 ^ x2) - y)
        return 4.0 - err

    eval_test = eval_train


class MEMORY_env(Environment):
    dat = [14, 3, 12, 35, 2, 31, 6, 2, 9, 19]

    def eval_train(network):
        err = 0
        for i in range(len(MEMORY_env.dat)):
            y = network.feedforward([MEMORY_env.dat[i]])[0]
            if i > 0:
                err += np.abs(MEMORY_env.dat[i-1]-y)
        return max(0.01, 10-err)

    def eval_test(network):
        err = 0
        x_prev = 0
        for i in range(10):
            x = random.random()
            y = network.feedforward([x])[0]
            if i > 0:
                err += np.abs((x_prev)-y)
            x_prev = x
        return max(0.01, 10-err)
