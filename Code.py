# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:33:46 2021

@author: schue
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\schue\Documents\Github\FHNW-WER-Wetter\TäglicheDaten.csv", sep=";")
df["Gesamtschneehöhe"] = df["Gesamtschneehöhe"].replace('-', 0)
df["Gesamtschneehöhe"] = pd.to_numeric(df["Gesamtschneehöhe"])
df["SchneenächsterTag"]= df['Gesamtschneehöhe'].shift(-1) - df['Gesamtschneehöhe']
# df = df.fillna(value= "-")

#Umformatierung Date zu Datum
df["Datum"] = pd.to_datetime(df["date"], format='%Y%m%d')
dfDez = df[pd.to_datetime(df['Datum']).dt.month == 12]
dfSnowDezperYearSum = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).sum()
dfSnowDezperYearCount = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).agg(pos=lambda ts: (ts > 0).sum()) 
dfSnow = df.loc[df['SchneenächsterTag'] > 0]
dfSnow = dfSnow.replace("-", np.nan)


dfDez2017 = dfDez[pd.to_datetime(df['Datum']).dt.year == 2017]
dfDez1981 = dfDez[pd.to_datetime(df['Datum']).dt.year == 1981]

print(dfDez1981)

dfSnowDezperYearSum.plot()
# dfSnowDezperYearCount.plot()
# dfDez2017["Gesamtschneehöhe"].hist()
# dfDez1981["Gesamtschneehöhe"].hist()
# dfSnow.plot.scatter(x='SchneenächsterTag',y='Niederschlag', c = "Lufttemperatur Tagesmittel", cmap="viridis")
# dfSnow.plot.scatter(x='SchneenächsterTag',y='Niederschlag', c = "Lufttemperatur Tagesminimum")
# dfSnow["Lufttemperatur Tagesmittel"].hist(bins = 40)
# dfSnow["Lufttemperatur Tagesminimum"].hist(bins = 10)
