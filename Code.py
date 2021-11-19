# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:33:46 2021

@author: schue
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv("TäglicheDaten.csv", sep=";")
df["Gesamtschneehöhe"] = df["Gesamtschneehöhe"].replace('-', 0)
df["Gesamtschneehöhe"] = pd.to_numeric(df["Gesamtschneehöhe"])
df["Lufttemperatur Tagesminimum"] = df["Lufttemperatur Tagesminimum"].replace('-', 0)
df["Lufttemperatur Tagesminimum"] = pd.to_numeric(df["Lufttemperatur Tagesminimum"])
df["SchneenächsterTag"]= df['Gesamtschneehöhe'].shift(-1) - df['Gesamtschneehöhe']
# df = df.fillna(value= "-")

#Umformatierung Date zu Datum
df["Datum"] = pd.to_datetime(df["date"], format='%Y%m%d')
dfDez = df[pd.to_datetime(df['Datum']).dt.month == 12]
dfSnowDezperYearSum = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).sum()
dfSnowDezperYearCount = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).agg(pos=lambda ts: (ts > 0).sum()) 
dfSnow = df.loc[df['SchneenächsterTag'] > 0]
dfSnow = dfSnow.replace("-", np.nan)


# dfDez2017 = dfDez[pd.to_datetime(df['Datum']).dt.year == 2017]
# dfDez1981 = dfDez[pd.to_datetime(df['Datum']).dt.year == 1981]

# print(dfDez1981)

# dfSnowDezperYearSum.plot()
# dfSnowDezperYearCount.plot()
# dfDez2017["Gesamtschneehöhe"].hist()
# dfDez2017["Lufttemperatur Tagesmittel"].hist()
# dfDez1981["Gesamtschneehöhe"].hist()
# dfDez1981["Lufttemperatur Tagesmittel"].hist()
# dfSnow.plot.scatter(x='SchneenächsterTag',y='Niederschlag', c = "Lufttemperatur Tagesmittel", cmap="viridis")
# dfSnow.plot.scatter(x='SchneenächsterTag',y='Niederschlag', c = "Lufttemperatur Tagesminimum")
# dfSnow["Lufttemperatur Tagesmittel"].hist(bins = 40)
# dfSnow["Lufttemperatur Tagesminimum"].hist(bins = 10)

Ten2010 = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year >= 2010) & (pd.to_datetime(dfDez['Datum']).dt.year <= 2019)]
# Ten2010 = Ten2010.groupby([pd.to_datetime(Ten2010['Datum']).dt.day]).agg({"Lufttemperatur Tagesminimum":"mean", "Niederschlag": "mean"})
Ten2010.hist("Niederschlag")
meanTempTen2010 = np.mean(Ten2010["Lufttemperatur Tagesminimum"])
stdTempTen2010 = np.std(Ten2010["Lufttemperatur Tagesminimum"])
Ten2010norm = norm(meanTempTen2010, stdTempTen2010)
print("Wahrscheinlichkeit Min. Temperatur unter 2.0 Grad während 10 Jahren ab 2010:", round(Ten2010norm.cdf(2.0), 4) * 100)

Ten1870 = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year >= 1870) & (pd.to_datetime(dfDez['Datum']).dt.year <= 1879)]
# Ten1870 = Ten1870.groupby([pd.to_datetime(Ten1870['Datum']).dt.day]).agg({"Lufttemperatur Tagesminimum":"mean", "Niederschlag": "mean"})
Ten1870.hist("Niederschlag")
meanTempTen1870 = np.mean(Ten1870["Lufttemperatur Tagesminimum"])
stdTempTen1870 = np.std(Ten1870["Lufttemperatur Tagesminimum"])
Ten1870norm = norm(meanTempTen1870, stdTempTen1870)
print("Wahrscheinlichkeit Min. Temperatur unter 2.0 Grad während 10 Jahren ab 1870:", round(Ten1870norm.cdf(2.0), 4) * 100)