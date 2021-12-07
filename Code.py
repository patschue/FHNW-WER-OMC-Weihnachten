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
from statsmodels.stats.weightstats import ztest as ztest


##Global Definitions
samplingTemperature = 2.0
TimePeriod = 10

dfPropabilitiesSnowPerDecade = pd.DataFrame({'TimePeriod': [],'Year': [],'NProbabilitySnow': [],'YearlyProbabilitySnow': []})
TTestResults =pd.DataFrame({'TimePeriod': [],'TStatistic': [],'PValue': []})

# Lesen von CSV daten
# df = pd.read_csv("TäglicheDaten.csv", sep=";")
# df = pd.read_csv("Meiringen.csv", sep=";")
# df = pd.read_csv("Säntis.csv", sep=";")
# df = pd.read_csv("StBernard.csv", sep=";")
df = pd.read_csv("Samedan.csv", sep=";")

#Korrekturen von Leeren werten mit 0
df["Gesamtschneehöhe"] = df["Gesamtschneehöhe"].replace('-', 0)
df["Lufttemperatur Tagesminimum"] = df["Lufttemperatur Tagesminimum"].replace('-', 0)

#Schneehöhe numerisch machen
df["Gesamtschneehöhe"] = pd.to_numeric(df["Gesamtschneehöhe"])
df["Lufttemperatur Tagesminimum"] = pd.to_numeric(df["Lufttemperatur Tagesminimum"])
df["Lufttemperatur Tagesmittel"] = pd.to_numeric(df["Lufttemperatur Tagesmittel"])


#Umformatierung zu Datum
df["Datum"] = pd.to_datetime(df["date"], format='%Y%m%d')

#Berechnung Schneehöhe gestern - Schneehöhe heute = Differenz eines Tages
df["SchneeTagesDifferenz"]= df['Gesamtschneehöhe'].shift(-1) - df['Gesamtschneehöhe']
df["SchneeVortag"]= df['SchneeTagesDifferenz'].shift(-1)
df["SchneeVorVortag"]= df["SchneeVortag"].shift(-1)

#Selektion von Dezember Tagen
dfDez = df[pd.to_datetime(df['Datum']).dt.month == 12]
dfDezforDays = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year >= 1886) & (pd.to_datetime(dfDez['Datum']).dt.year <= 1900) | (pd.to_datetime(dfDez['Datum']).dt.year >= 1931) & (pd.to_datetime(dfDez['Datum']).dt.year <= 2020)]

#DfDez mit weniger Spalten
dfDez = dfDez[['Datum','Gesamtschneehöhe','SchneeTagesDifferenz', "SchneeVortag", "SchneeVorVortag",'Niederschlag','Lufttemperatur Tagesmittel','Lufttemperatur Tagesminimum','Lufttemperatur Tagesmaximum']]

dfSnowDezperYearSum = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).sum()
dfSnowDezperYearCount = dfDez['Gesamtschneehöhe'].groupby(dfDez['Datum'].dt.year).agg(pos=lambda ts: (ts > 0).sum()) 
dfSnow = df.loc[df['SchneeTagesDifferenz'] > 0]
dfSnow = dfSnow.replace("-", np.nan)
dfSnowDez = dfSnow[pd.to_datetime(dfSnow['Datum']).dt.month == 12]
dfTest = dfSnowDez[(pd.to_datetime(dfSnowDez['Datum']).dt.year >= 1886) & (pd.to_datetime(dfSnowDez['Datum']).dt.year <= 1900) | (pd.to_datetime(dfSnowDez['Datum']).dt.year >= 1931) & (pd.to_datetime(dfSnowDez['Datum']).dt.year <= 2020)]
dfVortagSnow = dfDez.loc[dfDez['SchneeVortag'] > 0]
dfVorVortagSnow = dfDez.loc[dfDez['SchneeVorVortag'] > 0]


def Analyse_Vortage():
    dfSnowDez["Lufttemperatur Tagesmittel"].hist(bins = 40)
    print("Vortage ohne Niederschlag:", len(dfVortagSnow.loc[dfVortagSnow['Niederschlag'] == 0]), ", Anzahl Schneetage:", len(dfSnowDez), ", Anteil:", round(len(dfVortagSnow.loc[dfVortagSnow['Niederschlag'] == 0]) / len(dfVortagSnow), 2))
    print("Vortage Schnee Durchschnitt Temperatur:", round(dfVortagSnow["Lufttemperatur Tagesmittel"].mean(), 2), "Dezember Durchschnitt Temperatur:", round(dfDezforDays["Lufttemperatur Tagesmittel"].mean(), 2))
    print("Vorvortage ohne Niederschlag:", len(dfVorVortagSnow.loc[dfVorVortagSnow['Niederschlag'] == 0]), ", Anzahl Schneetage:", len(dfSnowDez), ", Anteil:", round(len(dfVorVortagSnow.loc[dfVorVortagSnow['Niederschlag'] == 0]) / len(dfVorVortagSnow), 2))
    print("Vorvortage Schnee Durchschnitt Temperatur:", round(dfVorVortagSnow["Lufttemperatur Tagesmittel"].mean(), 2), "Dezember Durchschnitt Temperatur:", round(dfDezforDays["Lufttemperatur Tagesmittel"].mean(), 2))
    print("Tage mit Temperatur für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]))
    print("Tage mit Temperatur für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]))
    print("Tage mit Temperatur x + 2 für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]))
    print("Tage mit Temperatur x + 2 für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]))    
    # Test = dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]
    print("Tage mit Niederschlag:", len(dfDezforDays[dfDezforDays["Niederschlag"] > 2.0]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Niederschlag"] > 2.0])) 
    print("Tage mit Niederschlag:", len(dfDezforDays[dfDezforDays["Niederschlag"] > 2.0]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Niederschlag"] > 2.0])) 
    print("Tage mit Niederschlag und Temperatur für Schnee:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]))
    print("Tage mit Niederschlag und Temperatur für Schnee:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
    print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[(dfDezforDays["Niederschlag"] > 2.0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]))
    # CDF Temperatur Vortag
    meanTempDecade = np.mean(dfVortagSnow["Lufttemperatur Tagesmittel"])
    stdTempDecade = np.std(dfVortagSnow["Lufttemperatur Tagesmittel"])
    VerteilungTempVortag = norm(meanTempDecade, stdTempDecade)
    VerteilungTempVortag = round(VerteilungTempVortag.cdf(3.5), 4)
    print("An", round(VerteilungTempVortag*100, 2), "% der Tage vor dem Schneefall war die Temperatur unter 3.5 Grad.")
    print("Diese Temperatur wurde an", round(len(dfDez.loc[dfDez['Lufttemperatur Tagesmittel'] <= 2])  / len(dfDez['Lufttemperatur Tagesmittel']), 2) * 100, "% der beobachteten Tage im Dezember unterschritten.")
    # CDF Temperatur Vorvortag
    meanTempDecade = np.mean(dfVorVortagSnow["Lufttemperatur Tagesmittel"])
    stdTempDecade = np.std(dfVorVortagSnow["Lufttemperatur Tagesmittel"])
    VerteilungTempVorvortag = norm(meanTempDecade, stdTempDecade)
    VerteilungTempVorvortag = round(VerteilungTempVorvortag.cdf(4), 4)
    print("An", round(VerteilungTempVorvortag*100, 2), "% des zweiten Tages vor dem Schneefall war die Temperatur unter 4.0 Grad.")
    print("Diese Temperatur wurde an", round(len(dfDez.loc[dfDez['Lufttemperatur Tagesmittel'] <= 3.5])  / len(dfDez['Lufttemperatur Tagesmittel']), 2) * 100, "% der beobachteten Tage im Dezember unterschritten.")
    # Regressionsanalyse Temperatur Vortag zu Temperatur Dezember
    print("Regressionsanalyse nicht möglich, da für Temperatur vor Schneefall im Dezember und Temperatur im Dezember ungleich viele Beobachtungen vorhanden sind.")
    # Hypothesentest Temperatur Vortag
    # HypothesentestTempVortag = stats.ttest_ind(dfDezforDays["Lufttemperatur Tagesmittel"], dfVortagSnow["Lufttemperatur Tagesmittel"],nan_policy="omit")
    # print("Anhand eines extrem kleinen p-Werts von:", HypothesentestTempVortag[1], "stellen wir fest, dass die Temperatur am Tag vor Schneefall unterschiedlich von den restlichen Temperaturen im Dezember ist.")
    # Alternativer Hypothesentest Temperatur Vortag
    result = ztest(dfDezforDays["Lufttemperatur Tagesmittel"], dfVortagSnow["Lufttemperatur Tagesmittel"], value=0)
    print(result)
    # Hypothesentest Temperatur Vorvortag
    # HypothesentestTempVortag = stats.ttest_ind(dfDezforDays["Lufttemperatur Tagesmittel"], dfVorVortagSnow["Lufttemperatur Tagesmittel"],nan_policy="omit")
    # print("Anhand eines extrem kleinen p-Werts von:", HypothesentestTempVortag[1], "stellen wir fest, dass die Temperatur zwei Tage vor Schneefall unterschiedlich von den restlichen Temperaturen im Dezember ist.")
    # Alternativer Hypothesentest Temperatur Vorvortag
    result = ztest(dfDezforDays["Lufttemperatur Tagesmittel"], dfVorVortagSnow["Lufttemperatur Tagesmittel"], value=0)
    print(result)
    # To-do: Überlegung, ob T-Test korrekt ist. P-Wert ist extrem klein, wohl wegen der grossen Anzahl Beobachtungen
    # return Test

# Analyse_Vortage()


def LoopTimePeriod():

    
    for timeperiod in range(2010,1900,-TimePeriod):
       
        #Define TimePeriod Start and End Year
        startYear = timeperiod
        endYear = timeperiod +TimePeriod
        print(str(startYear) + " - " + str(endYear))
        
        for year in range(endYear,startYear,-1):
            # print(year)
            #dfDezYear = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year >= year) & (pd.to_datetime(dfDez['Datum']).dt.year <= year)]
            dfDezYear = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year == year)]
            # print(dfDezYear)
            
            #Get Year of each TimePeriod 
            dfDezYearNoPrecipation= dfDezYear[(dfDezYear['Niederschlag'] == 0)]
            #Remove Precipation < 2 weil 2mm quasi kein Niederschlag ist (to be discussed)
            dfDezYearWithPrecipation=  dfDezYear[(dfDezYear['Niederschlag'] > 2.0)]
            
            # Probability for Precipation in Year December (PrecipationDays / All Days)
            pPrecipationYear = dfDezYearWithPrecipation.shape[0] / (dfDezYearWithPrecipation.shape[0] + dfDezYearNoPrecipation.shape[0])
            
            ### Mean Temperatur per Year and Probability of changing temp
            meanTempYear = np.mean(dfDezYear["Lufttemperatur Tagesmittel"])
            stdTempYear = np.std(dfDezYear["Lufttemperatur Tagesmittel"])
            Yearnorm = norm(meanTempYear, stdTempYear)
            
            TempYearbeneathSamplingTemp = round(Yearnorm.cdf(samplingTemperature), 4)
            # Wahrscheinlichkeit von Schnee gegeben Temperatur genug kalt
            YearProbabilitySnow =(pPrecipationYear * TempYearbeneathSamplingTemp) /TempYearbeneathSamplingTemp
                       
            dfPropabilitiesSnowPerDecade.loc[len(dfPropabilitiesSnowPerDecade.index)] = [timeperiod,year, len(dfDezYear),YearProbabilitySnow]
            
        PropDecadeOld = dfPropabilitiesSnowPerDecade[(dfPropabilitiesSnowPerDecade['TimePeriod'] == timeperiod)]
        PropDecadeNow = dfPropabilitiesSnowPerDecade[(dfPropabilitiesSnowPerDecade['TimePeriod'] == 2010)]
        
        TTest= stats.ttest_ind(PropDecadeOld['YearlyProbabilitySnow'], PropDecadeNow['YearlyProbabilitySnow'],nan_policy="omit")
        TTestResults.loc[len(TTestResults.index)] = [timeperiod,TTest[0],TTest[1]]
        TTestResults["PValue"].plot()
            

    
    #a = dfPropabilitiesSnowPerDecade[(dfPropabilitiesSnowPerDecade['TimePeriod'] == 1900)]
    

    return TTestResults

TTestResults= LoopTimePeriod()


    
def Regressionsanalyse():    
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
    
# Regressionsanalyse()   
    



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
        # Wahrscheinlichkeit für Schnee, gegeben Niederschlag
        print("Wahrscheinlichkeit für Schnee, gegeben Niederschlag:", (p*ProbabilitySnow) /p)
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
