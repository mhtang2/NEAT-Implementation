import requests
import pandas as pd
import numpy as np
import io
from datetime import date, timedelta, datetime, timezone
import time
import pytz
import json
from os import path


def retrieveTDAuthToken():
    """Get auth token using refresh token"""
    print("Retrieving TD Auth token")
    r = requests.post("https://api.tdameritrade.com/v1/oauth2/token", data={
        "grant_type": "refresh_token",
        "refresh_token": TDRefreshKey,
        "client_id": TDConsumerKey
    })
    if r.status_code != 200:
        print("Error getting TD auth code")
        return
    global TDAuthToken, TDAuthExpire
    TDAuthToken = json.loads(r.text)["access_token"]
    TDAuthExpire = datetime.now() + timedelta(minutes=29)


with open("apikeys.key") as APIfp:
    obj = json.load(APIfp)
    financialKey = obj['financialAPI']
    TDConsumerKey = obj["TDConsumer"]
    TDRefreshKey = obj["TDRefresh"]
    TDAuthToken, TDAuthExpire = ["", time.time()]
    retrieveTDAuthToken()


def getSymbolsTD(arr: list, directory="data", freqType="minute", periodType="day", freq=1, endDate=None, daysBack=0, save=True,
                 disableDelay=False, datedFileName=False):
    if endDate is None:
        # Calculate end time as now, or previous close if it is past close
        prevclose = datetime.now(tz=pytz.timezone("America/New_York"))
        if prevclose.hour < 4:
            prevclose -= timedelta(days=1)
        endDateStamp = 1000 * int(min(time.time(), prevclose.timestamp()))
        endDate = datetime.fromtimestamp(endDateStamp / 1e3)
    else:
        endDateStamp = 1000 * int(endDate.timestamp())
    if daysBack > 0 and endDate.weekday() < daysBack:
        daysBack += 2
    startDateStamp = endDateStamp - (daysBack * 86400000)

    res = {}
    count = 0
    errors = 0
    for sym in arr:
        count += 1
        if not disableDelay:
            print(f"Downloading  {sym}", end=" ")
        fileName = f"{directory}/{sym}{endDate.strftime('%m-%d-%Y') if datedFileName else ''}.csv"
        if save and path.exists(fileName):
            print("File exists")
            continue
        if not disableDelay:
            print("")
        nextRun = time.time() + 0.51
        r = requests.get(f"https://api.tdameritrade.com/v1/marketdata/{sym}/pricehistory",
                         headers={"Authorization": f"Bearer {TDAuthToken}"},
                         params={
                             "periodType": periodType,
                             "frequencyType": freqType,
                             "frequency": freq,
                            # In place of end start
                             "period": 5
                            #  "needExtendedHoursData": "true",
                            #  "endDate": endDateStamp,
                            #  "startDate": startDateStamp,
                         })
        if r.status_code == 200:
            try:
                obj = json.loads(r.text)
                df = pd.DataFrame(obj["candles"])
                df["datetime"] = pd.to_datetime(df['datetime'], unit='ms').dt.tz_localize(
                    'UTC').dt.tz_convert('America/New_York')
                if save:
                    df.to_csv(fileName, index=False)
                else:
                    res[sym] = df
            except:
                print("Download Error ", sym)
                errors += 1
        else:
            print(f"Download Error {sym} \n {r.text}")
            errors += 1
        sleepDur = nextRun - time.time()
        if sleepDur > 0 and not disableDelay:
            time.sleep(sleepDur)
    print(f"Total #errors {errors}")
    if not save:
        return res

if __name__=='__main__':
    df = pd.read_csv("fortune500.csv")
    syms = df['Symbol'].to_numpy()
    print(len(syms))
    getSymbolsTD(syms,directory="D:/stock_data",freqType="daily",periodType="year")
