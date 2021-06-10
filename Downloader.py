import requests
import pandas as pd
import numpy as np
import io
from utils import Data
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


def getSymbolsTD(arr: list, directory="data/TDtest", freqType="minute", periodType="day", freq=1, endDate=None, daysBack=0, save=True,
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
                             "needExtendedHoursData": "true",
                             "endDate": endDateStamp,
                             "startDate": startDateStamp,
                         })
        if r.status_code == 200:
            try:
                obj = json.loads(r.text)
                df = pd.DataFrame(obj["candles"])
                df["datetime"] = pd.to_datetime(df['datetime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
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


def getSymbolsHourlyFinancial(arr: np.ndarray, directory="data/hourly"):
    """Uses financial"""
    i = 0
    l = len(arr)
    for sym in arr:
        print(f"Downloading {sym} {i}/{l}", flush=True)
        i += 1
        link = f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{sym}"
        r = requests.get(link)
        try:
            jsonDat = json.loads(r.text)
            jsonDat.reverse()
            df = pd.DataFrame(jsonDat)
            df.to_csv(directory + f"/{sym}.csv", index=False)
        except (KeyError, json.decoder.JSONDecodeError):
            print(sym, r.text)


def getSymbolsMinutelyFinancial(arr: np.ndarray, directory="data/minutely"):
    """Uses financial"""
    i = 0
    l = len(arr)
    for sym in arr:
        print(f"Downloading {sym} {i}/{l}", flush=True)
        i += 1
        link = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{sym}"
        r = requests.get(link)
        try:
            jsonDat = json.loads(r.text)
            jsonDat.reverse()
            df = pd.DataFrame(jsonDat)
            df.to_csv(f"{directory}/{sym}.csv", index=False)
        except KeyError:
            print(sym, r.text)


def getSymbols5minutelyFinancial(arr: np.ndarray, directory="data/quarterhourly"):
    """Uses financial"""
    i = 0
    l = len(arr)
    for sym in arr:
        print(f"Downloading {sym} {i}/{l}", flush=True)
        i += 1
        link = f"https://financialmodelingprep.com/api/v3/historical-chart/15min/{sym}"
        r = requests.get(link)
        try:
            jsonDat = json.loads(r.text)
            jsonDat.reverse()
            df = pd.DataFrame(jsonDat)
            df.to_csv(f"{directory}/{sym}.csv", index=False)
        except KeyError:
            print(sym, r.text)


def getSymbolsNASDAQ(arr: np.ndarray, days=5 * 365.25, directory="data"):
    """Download csv for each symbol"""
    dateEnd = date.today().strftime("%Y-%m-%d")
    # handle crossing monday
    if date.today().weekday() - days < 0:
        days += 2
    dateBegin = (date.today() - timedelta(days)).strftime("%Y-%m-%d")
    i = 0
    l = len(arr)
    for sym in arr:
        i += 1
        link = f"https://www.nasdaq.com/api/v1/historical/{sym}/stocks/{dateBegin}/{dateEnd}"
        headers = {'user-agent': 'Mozilla/5.0'}
        r = requests.get(link, headers=headers)
        file = open(directory + f"/{sym}.csv", 'w')
        file.write(r.text.replace("$", ""))
        print(f"Downloading {sym} {i}/{l}", flush=True)


def getSymbolsDaily(arr: np.ndarray, days=365, directory='data'):
    """Download csv for each symbol"""
    dateEnd = date.today().strftime("%Y-%m-%d")
    # handle crossing monday
    if date.today().weekday() - days < 0:
        days += 2
    dateBegin = (date.today() - timedelta(days)).strftime("%Y-%m-%d")
    i = 0
    l = len(arr) - 1
    for sym in arr:
        print(f"Downloading {sym} {i}/{l}")
        link = f"https://financialmodelingprep.com/api/v3/historical-price-full/{sym}?from={dateBegin}&to={dateEnd}"
        r = requests.get(link)
        dat = json.loads(r.text)
        try:
            df = pd.DataFrame(dat["historical"]).filter(["date", "open", "high", "low", "close", "volume"])
            df.to_csv(f"{directory}/{sym}.csv", index=False)
        except KeyError:
            print(sym, r.text)
        i += 1


def downloadNASDAQListing():
    """Download full listing with opening prices using NASDAQ API"""
    link = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
    headers = {'user-agent': 'Mozilla/5.0'}
    r = requests.get(link, headers=headers)
    dat = r.text.replace("$", "")
    df = pd.read_csv(io.StringIO(dat), usecols=np.arange(7), sep=",", engine="python", quotechar='"', error_bad_lines=False)
    df.columns = ["symbol", "name", "price", "marketcap", "ipo", "sector", "industry"]
    df["symbol"] = df["symbol"].apply(lambda x: x.strip())
    df.to_csv("data/nasdaqListing.csv", index=False)
    print(df.to_string())


def downloadListing():
    a = requests.get(f"https://financialmodelingprep.com/api/v3/company/stock/list?apikey={financialKey}")
    obj = json.loads(a.text)
    df = pd.DataFrame(obj["symbolsList"]).filter(["symbol", "price", "exchange"])
    df = df[(df.exchange.str.contains("NASDAQ|Nasdaq|NYSE|New York Stock Exchange", na=False))].sort_values("price")
    df.to_csv("data/Listing.csv", index=False)
    print(df.size)


def downloadCurrent():
    symbols = Data.listingsUnderLimit(5)
    link = f"https://financialmodelingprep.com/api/v3/quote/{','.join(map(str, symbols))}?apikey={financialKey}"
    a = requests.get(
        link)
    companies = json.loads(a.text)
    print(f"Downloaded curent {datetime.fromtimestamp(companies[0]['timestamp'])}")
    companiesDict = {}
    for entry in companies:
        symbol = entry.pop("symbol")
        companiesDict[symbol] = entry
    with open("data/currentPricesDict.json", "w") as f:
        json.dump(companiesDict, f, indent=2)


def downloadMicrosHourly():
    syms = Data.listingsUnderLimit(5)
    getSymbolsHourlyFinancial(syms)


def downloadMicrosMinutely():
    syms = Data.listingsUnderLimit(5)
    getSymbolsMinutelyFinancial(syms)


def downloadMicros5minutely():
    syms = Data.listingsUnderLimit(5)
    getSymbols5minutelyFinancial(syms)


def downloadMicros90Day():
    syms = Data.listingsUnderLimit(5)
    getSymbolsNASDAQ(syms, 90, "data/90day")


def downloadMicros5Y():
    syms = Data.listingsUnderLimit(5)
    getSymbolsDaily(syms, days=int(5 * 365.25), directory="data/5year")
