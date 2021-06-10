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
        training = data[:-41]
        testing = data[-41:]
        return training, testing, scaler

    trainingDat, testingDat, scaler = loadData()

    random_chunk_start = -1

    def setRandomStart():
        Stock_env.random_chunk_start = random.randint(
            len(Stock_env.trainingDat) - 40)
    
    def theoretical_max_profit():
        availible_cash = STARTING_CASH
        shares_held = 0
        for day in range(Stock_env.random_chunk_start, Stock_env.random_chunk_start + 40):
            if Stock_env.trainingDat[day + 1][0] > Stock_env.trainingDat[day][0]:
                shares_held += (availible_cash / Stock_env.trainingDat[day][0])
                availible_cash = 0
            elif Stock_env.trainingDat[day + 1][0] < Stock_env.trainingDat[day][0]:
                available_cash += (shares_held * Stock_env.trainingDat[day][0])
                shares_held = 0
        total_money = available_cash + \
            (shares_held * Stock_env.trainingDat[Stock_env.random_chunk_start + 40][0])
        return total_money


    STARTING_CASH = 1000

    def eval_train(network):
        available_cash = Stock_env.STARTING_CASH
        shares_held = 0
        # Pick random 40 day block
        start = Stock_env.random_chunk_start
        for day in range(start, start + 40):
            action = network.feedforward(
                Stock_env.trainingDat[day].tolist())[0]
            if day == start:
                continue
            if action > 0.03:
                buy = available_cash * action
                # Divide by closing price
                shares_held += (buy / Stock_env.trainingDat[day][0])
                available_cash -= buy
            elif action < -0.03:
                shares_sold = shares_held * -action
                available_cash += (shares_sold * Stock_env.trainingDat[day][0])
                shares_held = shares_held * (1+action)
        total_money = available_cash + \
            (shares_held * Stock_env.trainingDat[start + 40][0])
        return total_money

    def eval_test(network):
        total_money = 0
        for test_case in range(1):
            available_cash = Stock_env.STARTING_CASH
            shares_held = 0
            # Pick random 40 day block
            start = random.randint(len(Stock_env.testingDat) - 40)
            for day in range(start, start + 40):
                action = network.feedforward(
                    Stock_env.testingDat[day].tolist())[0]
                if day == start:
                    continue
                if action > 0.03:
                    buy = available_cash * action
                    # Divide by closing price
                    shares_held += buy / Stock_env.testingDat[day][0]
                    available_cash -= buy
                elif action < -0.03:
                    shares_sold = shares_held * -action
                    available_cash += (shares_sold * Stock_env.testingDat[day][0])
                    shares_held = shares_held * (1+action)
            total_money += available_cash + \
                (shares_held * Stock_env.testingDat[start + 40][0])
        return total_money


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
