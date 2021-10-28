# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:33:46 2021

@author: schue
"""

import pandas as pd

df = pd.read_excel(r"C:\Users\schue\Documents\Github\FHNW-WER-Wetter\TäglicheDaten.xlsx")

df["Datum"] = pd.to_datetime(df["date"], format='%Y%m%d').dt.date
# df = df.set_index("Datum")

# GB = df.groupby([(df.index.dt.year),(df.index.dt.month )]).sum()
# b.groupby(by=[b.index.month, b.index.year])
# max_temp = df.groupby([df['Datum'].dt.year, df['Datum'].dt.month]).agg({'count'})