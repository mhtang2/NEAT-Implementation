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
    if len(df) < CHUNK:
        print(f"{filename} has too little data")
        return None
    df = df.iloc[::-1]
    df['close_unscaled'] = df['close']
    scaler = preprocessing.MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
        df[['open', 'high', 'low', 'close', 'volume']])
    return df.values


def loadAllData():
    trainingDat, testingDat = [], []
    dataFiles = os.listdir("D:/stock_data")
    testNum = random.choice(range(len(dataFiles)),
                            size=len(dataFiles)//10, replace=False)
    for i, fileName in enumerate(os.listdir("D:/stock_data")):
        stockData = loadStockData("D:/stock_data/" + fileName)
        if stockData is None:
            continue
        if i in testNum:
            testingDat.append(stockData)
        else:
            trainingDat.append(stockData)
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

    def perfect_bot():
        available_cash = STARTING_CASH
        shares_held = 0
        for day in range(Stock_env.random_chunk_start, Stock_env.random_chunk_start + CHUNK-1):
            if Stock_env.trainingDat[Stock_env.random_stock][day + 1][5] > Stock_env.trainingDat[Stock_env.random_stock][day][5]:
                shares_held += (available_cash /
                                Stock_env.trainingDat[Stock_env.random_stock][day][5])
                available_cash = 0
            elif Stock_env.trainingDat[Stock_env.random_stock][day + 1][5] < Stock_env.trainingDat[Stock_env.random_stock][day][5]:
                available_cash += (shares_held *
                                   Stock_env.trainingDat[Stock_env.random_stock][day][5])
                shares_held = 0
        total_money = available_cash +\
            (shares_held *
             Stock_env.trainingDat[Stock_env.random_stock][Stock_env.random_chunk_start + CHUNK][5])
        return total_money

    def momentum_bot():
        available_cash = STARTING_CASH
        shares_held = 0
        for day in range(Stock_env.random_chunk_start + 1, Stock_env.random_chunk_start + CHUNK):
            if Stock_env.trainingDat[Stock_env.random_stock][day][5] > Stock_env.trainingDat[Stock_env.random_stock][day-1][5]:
                shares_held += (available_cash /
                                Stock_env.trainingDat[Stock_env.random_stock][day][5])
                available_cash = 0
            elif Stock_env.trainingDat[Stock_env.random_stock][day][5] < Stock_env.trainingDat[Stock_env.random_stock][day-1][5]:
                available_cash += (shares_held *
                                   Stock_env.trainingDat[Stock_env.random_stock][day][5])
                shares_held = 0
        total_money = available_cash +\
            (shares_held *
             Stock_env.trainingDat[Stock_env.random_stock][Stock_env.random_chunk_start + CHUNK][5])
        return total_money

    STARTING_CASH = 1000

    def eval_train(network):
        available_cash = Stock_env.STARTING_CASH
        shares_held = 0
        # Pick random 40 day block
        start = Stock_env.random_chunk_start
        for day in range(start, start + CHUNK):
            action = network.feedforward(
                Stock_env.trainingDat[Stock_env.random_stock][day][:5].tolist())[0]
            if day == start:
                continue
            if action > 0.03:
                buy = available_cash * action
                # Divide by closing price
                shares_held += (buy /
                                Stock_env.trainingDat[Stock_env.random_stock][day][5])
                available_cash -= buy
            elif action < -0.03:
                shares_sold = shares_held * -action
                available_cash += (shares_sold *
                                   Stock_env.trainingDat[Stock_env.random_stock][day][5])
                shares_held = shares_held * (1+action)
        total_money = available_cash + \
            (shares_held *
             Stock_env.trainingDat[Stock_env.random_stock][start + CHUNK][5])
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
                    Stock_env.testingDat[random_stock][day][:5].tolist())[0]
                if day == start:
                    continue
                if action > 0.03:
                    buy = available_cash * action
                    # Divide by closing price
                    shares_held += buy / \
                        Stock_env.testingDat[random_stock][day][5]
                    available_cash -= buy
                elif action < -0.03:
                    shares_sold = shares_held * -action
                    available_cash += (shares_sold *
                                       Stock_env.testingDat[random_stock][day][5])
                    shares_held = shares_held * (1+action)
            total_money += available_cash + \
                (shares_held *
                 Stock_env.testingDat[random_stock][start + CHUNK][5])
        return total_money
