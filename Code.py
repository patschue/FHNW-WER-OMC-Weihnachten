# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:33:46 2021

@author: schue
"""

import pandas as pd
import numpy as np

# df = pd.read_excel(r"C:\Users\schue\Documents\Github\FHNW-WER-Wetter\TäglicheDaten.xlsx")
df = pd.read_csv(r"C:\Users\schue\Documents\Github\FHNW-WER-Wetter\TäglicheDaten.csv", sep=";")
df["Gesamtschneehöhe"] = df["Gesamtschneehöhe"].replace('-', 0)
df["Gesamtschneehöhe"] = pd.to_numeric(df["Gesamtschneehöhe"])
# df = df.reset_index()

df["Datum"] = pd.to_datetime(df["date"], format='%Y%m%d')
dfDez = df[pd.to_datetime(df['Datum']).dt.month == 12]
dfSnowDezperYearSum = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).sum()
dfSnowDezperYearCount = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).agg(pos=lambda ts: (ts > 0).sum()) 

dfDez2017 = dfDez[pd.to_datetime(df['Datum']).dt.year == 2017]
dfDez1981 = dfDez[pd.to_datetime(df['Datum']).dt.year == 1981]

# dfSnowDezperYearSum.plot()
# dfSnowDezperYearCount.plot()
# dfDez2017["Gesamtschneehöhe"].hist()
dfDez1981["Gesamtschneehöhe"].hist()