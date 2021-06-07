from Net import Network
import Net
import numpy as np
from numpy import random
import pandas as pd
from sklearn import preprocessing


class Environment():
    def eval_train(network):
        pass

    def eval_test(network):
        pass


class Stock_env(Environment):

    def loadData():
        df = pd.read_csv(
            "data/aapl1y")[["Close/Last", "Volume", "Open", "High", "Low"]]
        df = df.iloc[::-1]
        print(df.head)
        data = df.values
        print(data)
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        training = data[:-30]
        testing = data[-30:]
        return training, testing, scaler

    trainingDat, testingDat, scaler = loadData()

    # Predict next day close given previous day open, volume, high, low, and close
    def eval_train(network) -> float:
        err = 0
        for i in range(len(Stock_env.trainingDat)-1):
            xi = Stock_env.trainingDat[i].tolist()
            y = network.feedforward(xi)[0]
            err += np.abs(Stock_env.trainingDat[i+1][0]-y)
        return max(0.01, 260-err)

    def eval_test(network) -> float:
        err = 0
        for i in range(len(Stock_env.testingDat)-1):
            xi = Stock_env.testingDat[i].tolist()
            y = network.feedforward(xi)[0]
            err += np.abs(Stock_env.testingDat[i+1][0]-y)
        return max(0.01, 30-err)


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
