from .environment import Environment
from Net import Network
import Net
import numpy as np
from numpy import random
import pandas as pd
from sklearn import preprocessing
import os

CHUNK = 40


def loadStockData(filename):
    df = pd.read_csv(filename)[['open', 'high', 'low', 'close', 'volume']]
    df = df.iloc[::-1]
    data = df.values
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    training = data[:-(CHUNK+1)]
    testing = data[-(CHUNK+1):]
    return training, testing, scaler


def loadAllData():
    trainingDat, testingDat, scaler = [], [], []
    for fileName in os.listdir("D:/stock_data"):
        stockData = loadStockData("D:/stock_data/" + fileName)
        trainingDat.append(stockData[0])
        testingDat.append(stockData[1])
        scaler.append(stockData[2])
    print("DATA LOADED")
    return trainingDat, testingDat


STARTING_CASH = 1000


class Stock_env(Environment):
    trainingDat, testingDat = loadAllData()

    random_chunk_start = -1
    random_stock = -1

    def setRandomStart():
        Stock_env.random_stock = random.randint(len(Stock_env.trainingDat))
        Stock_env.random_chunk_start = random.randint(
            len(Stock_env.trainingDat[Stock_env.random_stock]) - CHUNK)

    def theoretical_max_profit():
        available_cash = STARTING_CASH
        shares_held = 0
        for day in range(Stock_env.random_chunk_start, Stock_env.random_chunk_start + CHUNK):
            if Stock_env.trainingDat[Stock_env.random_stock][day + 1][3] > Stock_env.trainingDat[Stock_env.random_stock][day][3]:
                shares_held += (availible_cash /
                                Stock_env.trainingDat[Stock_env.random_stock][day][3])
                availible_cash = 0
            elif Stock_env.trainingDat[Stock_env.random_stock][day + 1][3] < Stock_env.trainingDat[Stock_env.random_stock][day][3]:
                available_cash += (shares_held *
                                   Stock_env.trainingDat[Stock_env.random_stock][day][3])
                shares_held = 0
        total_money = available_cash + \
            (shares_held *
             Stock_env.trainingDat[Stock_env.random_stock][Stock_env.random_chunk_start + CHUNK][3])
        return total_money

    STARTING_CASH = 1000

    def eval_train(network):
        available_cash = Stock_env.STARTING_CASH
        shares_held = 0
        # Pick random 40 day block
        start = Stock_env.random_chunk_start
        for day in range(start, start + CHUNK):
            action = network.feedforward(
                Stock_env.trainingDat[Stock_env.random_stock][day].tolist())[0]
            if day == start:
                continue
            if action > 0.03:
                buy = available_cash * action
                # Divide by closing price
                shares_held += (buy /
                                Stock_env.trainingDat[Stock_env.random_stock][day][3])
                available_cash -= buy
            elif action < -0.03:
                shares_sold = shares_held * -action
                available_cash += (shares_sold *
                                   Stock_env.trainingDat[Stock_env.random_stock][day][3])
                shares_held = shares_held * (1+action)
        total_money = available_cash + \
            (shares_held *
             Stock_env.trainingDat[Stock_env.random_stock][start + CHUNK][3])
        return total_money

    def eval_test(network):
        total_money = 0
        for test_case in range(1):
            available_cash = Stock_env.STARTING_CASH
            shares_held = 0
            random_stock = random.randint(len(Stock_env.testingDat))
            # Pick random 40 day block
            start = random.randint(
                len(Stock_env.testingDat[random_stock]) - CHUNK)
            for day in range(start, start + CHUNK):
                action = network.feedforward(
                    Stock_env.testingDat[random_stock][day].tolist())[0]
                if day == start:
                    continue
                if action > 0.03:
                    buy = available_cash * action
                    # Divide by closing price
                    shares_held += buy / \
                        Stock_env.testingDat[random_stock][day][3]
                    available_cash -= buy
                elif action < -0.03:
                    shares_sold = shares_held * -action
                    available_cash += (shares_sold *
                                       Stock_env.testingDat[random_stock][day][3])
                    shares_held = shares_held * (1+action)
            total_money += available_cash + \
                (shares_held *
                 Stock_env.testingDat[random_stock][start + CHUNK][3])
        return total_money
