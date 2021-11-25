# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:33:46 2021

@author: schue
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import Distribution as d
import matplotlib.pyplot as plt
from numpy.random.mtrand import exponential
from statistics import mean


##Global Definitions
samplingTemperature = 2.0
TimePeriod = 10

dfPropabilitiesSnowPerDecade = pd.DataFrame({'TimePeriod': [],'Year': [],'NProbabilitySnow': [],'YearlyProbabilitySnow': []})


# Lesen von CSV daten
df = pd.read_csv("TäglicheDaten.csv", sep=";")

#Korrekturen von Leeren werten mit 0
df["Gesamtschneehöhe"] = df["Gesamtschneehöhe"].replace('-', 0)
df["Lufttemperatur Tagesminimum"] = df["Lufttemperatur Tagesminimum"].replace('-', 0)

#Schneehöhe numerisch machen
df["Gesamtschneehöhe"] = pd.to_numeric(df["Gesamtschneehöhe"])
df["Lufttemperatur Tagesminimum"] = pd.to_numeric(df["Lufttemperatur Tagesminimum"])

#Umformatierung zu Datum
df["Datum"] = pd.to_datetime(df["date"], format='%Y%m%d')

#Berechnung Schneehöhe gestern - Schneehöhe heute = Differenz eines Tages
df["SchneeTagesDifferenz"]= df['Gesamtschneehöhe'].shift(-1) - df['Gesamtschneehöhe']

#Selektion von Dezember Tagen
dfDez = df[pd.to_datetime(df['Datum']).dt.month == 12]

#DfDez mit weniger Spalten
dfDez = dfDez[['Datum','Gesamtschneehöhe','SchneeTagesDifferenz','Niederschlag','Lufttemperatur Tagesmittel','Lufttemperatur Tagesminimum','Lufttemperatur Tagesmaximum']]


def LoopTimePeriod():
    for timeperiod in range(1900,2020,TimePeriod):
       
        #Define TimePeriod Start and End Year
        startYear = timeperiod
        endYear = timeperiod +TimePeriod
        print(str(startYear) + " - " + str(endYear))
        
        for year in range(startYear,endYear):
            print(year)
            #dfDezYear = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year >= year) & (pd.to_datetime(dfDez['Datum']).dt.year <= year)]
            dfDezYear = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year == year)]
            print(dfDezYear)
            
            #Get Year of each TimePeriod 
            dfDezYearNoPrecipation= dfDezYear[(dfDezYear['Niederschlag'] == 0)]
            #Remove Precipation < 2 weil 2mm quasi kein Niederschlag ist (to be discussed)
            dfDezYearWithPrecipation=  dfDezYear[(dfDezYear['Niederschlag'] > 2.0)]
            
            # Probability for Precipation in Year December (PrecipationDays / All Days)
            pYear = dfDezYearWithPrecipation.shape[0] / (dfDezYearWithPrecipation.shape[0] + dfDezYearNoPrecipation.shape[0])
            
            ### Mean Temperatur per Year and Probability of changing temp
            meanTempYear = np.mean(dfDezYear["Lufttemperatur Tagesmittel"])
            stdTempYear = np.std(dfDezYear["Lufttemperatur Tagesmittel"])
            Yearnorm = norm(meanTempYear, stdTempYear)
            
            TempYearbeneathSamplingTemp = round(Yearnorm.cdf(samplingTemperature), 4)
            YearProbabilitySnow =(pYear * TempYearbeneathSamplingTemp) /TempYearbeneathSamplingTemp
                       
            dfPropabilitiesSnowPerDecade.loc[len(dfPropabilitiesSnowPerDecade.index)] = [timeperiod,year, len(dfDezYear),YearProbabilitySnow]
    
    a = dfPropabilitiesSnowPerDecade[(dfPropabilitiesSnowPerDecade['TimePeriod'] == 1900)]
    b = dfPropabilitiesSnowPerDecade[(dfPropabilitiesSnowPerDecade['TimePeriod'] == 2000)]

    return stats.ttest_ind(a['YearlyProbabilitySnow'], b['YearlyProbabilitySnow'],nan_policy="omit")

TTestResults= LoopTimePeriod()


    
    
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
linregress_x = dfDez["Lufttemperatur Tagesminimum"]
# linregress_x = df["Lufttemperatur Tagesminimum"]
linregress_y = dfDez['Niederschlag']
# linregress_y = df['Niederschlag']
slope, intercept, r_value, p_value, std_err = stats.linregress(linregress_x, linregress_y)
print("Die Korrelation beträgt:", round(r_value, 3), " und R^2 beträgt:", round(r_value ** 2, 3))
print("Die Variablen haben also keinen statistischen Zusammenhang.")
plt.plot(linregress_x, linregress_y, 'o', label='original data')
plt.plot(linregress_x, intercept + slope*linregress_x, 'r', label='fitted line')
plt.legend()
plt.show()
    
    
    

dfSnowDezperYearSum = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).sum()
dfSnowDezperYearCount = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).agg(pos=lambda ts: (ts > 0).sum()) 
dfSnow = df.loc[df['SchneeTagesDifferenz'] > 0]
dfSnow = dfSnow.replace("-", np.nan)
dfSnowDez = dfSnow[pd.to_datetime(dfSnow['Datum']).dt.month == 12]

# Loop for each decade
# Funktion kann wieder aufgelöst werden, ich habe sie erstellt, damit die Rechnungen nicht jedesmal laufen
def LoopDecade():
    for decade in range(2000,2020,10):
        
        #Define Decade Start and End Year
        startYear = decade
        endYear = decade +9
        print(str(startYear) + " - " + str(endYear))
        
        
        dfDezDecade = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year >= startYear) & (pd.to_datetime(dfDez['Datum']).dt.year <= endYear)]
    
        #Get Subsets of each Decade 
        dfDezDecadeNoPrecipation= dfDezDecade[(dfDezDecade['Niederschlag'] == 0)]
        #Remove Precipation < 2 weil 2mm quasi kein Niederschlag ist (to be discussed)
        dfDezDecadeWithPrecipation=  dfDezDecade[(dfDezDecade['Niederschlag'] > 2.0)]
    
    
    
    
        ### Probability that Precipation occurs ##
    
        # Probability for Precipation in decades December (PrecipationDays / All Days)
        p = dfDezDecadeWithPrecipation.shape[0] / (dfDezDecadeWithPrecipation.shape[0] + dfDezDecadeNoPrecipation.shape[0])
        
        #Generieren von 31 Tagen mit der Bernoulli Probability (simulierter Dezember)
        bernoullidaysperdecade = stats.bernoulli.rvs(p, size=31)
        
        #selektion von tagen mit precipation
        DaysPrecipationperYear = bernoullidaysperdecade[bernoullidaysperdecade==1].shape[0]
        
        
        ### Simulation of Precipation precipitation quantity ###
        
        #Plot the decade
        dfDezDecadeWithPrecipation.hist("Niederschlag")
        plt.suptitle('Niederschlag ' + str(startYear) + "-" +str(endYear))
        #Fit Distribution
        print("Niederschlag Verteilung"+ str(startYear) + "-" +str(endYear))
        distributionYearPrecipation = d.FindmostfittingDistribution(dfDezDecadeWithPrecipation['Niederschlag'])
        
    
      
        #Sample generation for quantity of Precipation on days with Precipation (PrecipationDaysDecade / 10 (years) = mean Precipation days for a year in the decade)
        explambda = mean(dfDezDecadeWithPrecipation['Niederschlag'])
        sample = exponential(explambda, DaysPrecipationperYear)
        print(sample)
        
        
        
        
        ### Snow height and Differences
        dfDezDecade.hist("Gesamtschneehöhe")
        plt.suptitle('Gesamtschneehöhe ' + str(startYear) + "-" +str(endYear))
        dfDezDecade.hist("SchneeTagesDifferenz")
        plt.suptitle('SchneeTagesDifferenz ' + str(startYear) + "-" +str(endYear))
        
        
        ### Fitting 
        print("Temperatur Verteilung"+ str(startYear) + "-" +str(endYear))
        distributionDecadeTemperature = d.FindmostfittingDistribution(dfDezDecade['Lufttemperatur Tagesmittel'])
        
        
        
        ### Mean Temperatur per Decade and Probability of changing temp
        meanTempDecade = np.mean(dfDezDecade["Lufttemperatur Tagesminimum"])
        stdTempDecade = np.std(dfDezDecade["Lufttemperatur Tagesminimum"])
        TenDecadenorm = norm(meanTempDecade, stdTempDecade)
        
        ProbabilitySnow = round(TenDecadenorm.cdf(samplingTemperature), 4)
        print("Wahrscheinlichkeit Min. Temperatur unter "+ str(samplingTemperature) +" Grad während 10 Jahren ab "+ str(startYear) +":", ProbabilitySnow)
        
        print(DaysPrecipationperYear)
        print((p*ProbabilitySnow) /ProbabilitySnow)
        #print(((DaysPrecipationperYear/31)*ProbabilitySnow) /ProbabilitySnow)
        
        
        
        
        ### BodenTemperatur Voraussetzung Lin Regression



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
