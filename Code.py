# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest as ztest


# Global Definitions
samplingTemperature = 2
TimePeriod = 10
HypothesisDecade = 2010


#Regionen CSV Daten
# regionsCsvData = [["Basel.csv", ";",1860]]
regionsCsvData = [["Basel.csv", ";",1860], ["Meiringen.csv", ";",1958], ["SilsMaria.csv", ";",1900], ["StBernard.csv", ";",1860], ["Säntis.csv", ";",1900]] #Meiringen war 1860
Vortage = pd.DataFrame(index=['Tage mit Temperatur für Schnee', 'Tage mit Schneefall', 'Temperatur Wahrscheinlichkeit Folgetag', 'Temperatur Wahrscheinlichkeit zweiter Folgetag', 'Niederschlag Wahrscheinlichkeit Folgetag', 'Niederschlag Wahrscheinlichkeit zweiter Folgetag', 'P-Wert Vergleich Temp Vortag'])


def CleanDF(regionDataFrame):

    # Korrekturen von Leeren werten mit 0
    regionDataFrame["Gesamtschneehöhe"] = regionDataFrame["Gesamtschneehöhe"].replace('-', 0)
    regionDataFrame["Lufttemperatur Tagesmittel"] = regionDataFrame["Lufttemperatur Tagesmittel"].replace(
    "−", "-")
    regionDataFrame["Lufttemperatur Tagesmittel"] = pd.to_numeric(
    regionDataFrame["Lufttemperatur Tagesmittel"], errors='coerce')
    regionDataFrame["Niederschlag"] = pd.to_numeric(regionDataFrame["Niederschlag"], errors='coerce')

    #Schneehöhe numerisch machen
    regionDataFrame["Gesamtschneehöhe"] = pd.to_numeric(regionDataFrame["Gesamtschneehöhe"])


    # Umformatierung zu Datum
    regionDataFrame["Datum"] = pd.to_datetime(regionDataFrame["date"], format='%Y%m%d')

    # Berechnung Schneehöhe gestern - Schneehöhe heute = Differenz eines Tages
    regionDataFrame["SchneeTagesDifferenz"] = regionDataFrame['Gesamtschneehöhe'].shift(
    -1) - regionDataFrame['Gesamtschneehöhe']
    regionDataFrame["SchneeVortag"] = regionDataFrame['SchneeTagesDifferenz'].shift(-1)
    regionDataFrame["SchneeVorVortag"] = regionDataFrame["SchneeVortag"].shift(-1)

    # Selektion von Dezember Tagen
    dfDez = regionDataFrame[pd.to_datetime(regionDataFrame['Datum']).dt.month == 12]


    # DfDez mit weniger Spalten
    dfDez = dfDez[['Datum', 'Gesamtschneehöhe', 'SchneeTagesDifferenz', "SchneeVortag", "SchneeVorVortag",
               'Niederschlag', 'Lufttemperatur Tagesmittel']]
    
    return dfDez


def LoopTimePeriod(Region):

    for timeperiod in range(HypothesisDecade, Region[2], -TimePeriod):

        # Define TimePeriod Start and End Year
        startYear = timeperiod
        endYear = timeperiod + TimePeriod
        print(str(startYear) + " - " + str(endYear))

        for year in range(endYear, startYear, -1):
            dfDezYear = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year == year)]
            # Get Dataframe of Precipation Days
            dfDezYearWithPrecipation = dfDezYear[(dfDezYear['Niederschlag'] > 0)]
            
            # Probabilities
            try:
                PPrecipationYearly =  len(dfDezYearWithPrecipation) / len(dfDezYear)
                # Mean Temperatur per Year and Probability of changing temp
                meanTempYear = np.mean(dfDezYear["Lufttemperatur Tagesmittel"])
                stdTempYear = np.std(dfDezYear["Lufttemperatur Tagesmittel"])
                Yearnorm = norm(meanTempYear, stdTempYear)
                TempYearbeneathSamplingTemp = round(Yearnorm.cdf(samplingTemperature), 4)

                # Wahrscheinlichkeit von Schnee gegeben Temperatur genug kalt
                YearProbabilitySnow = PPrecipationYearly * TempYearbeneathSamplingTemp

                #print(YearProbabilitySnow)
                dfPropabilitiesSnowPerDecade.loc[len(dfPropabilitiesSnowPerDecade.index)] = [timeperiod, year, len(
                    dfDezYear), YearProbabilitySnow, TempYearbeneathSamplingTemp, PPrecipationYearly]
            except ZeroDivisionError:
                print("0 Division")
                
        PropDecadeOld = dfPropabilitiesSnowPerDecade[(
            dfPropabilitiesSnowPerDecade['TimePeriod'] == timeperiod)]
        PropDecadeNow = dfPropabilitiesSnowPerDecade[(
            dfPropabilitiesSnowPerDecade['TimePeriod'] == HypothesisDecade)]

        TTest = stats.ttest_ind(
            PropDecadeOld['YearlyProbabilitySnow'], PropDecadeNow['YearlyProbabilitySnow'], nan_policy="omit")
        TTestResults.loc[len(TTestResults.index)] = [Region[0], timeperiod, TTest[0], TTest[1]]

    return TTestResults




def PlotTTestResults(TTestResults,Region):
    
    TTestResults = TTestResults.sort_values(
    by='TimePeriod', ascending=True).reset_index(drop=True)
    TTestResults["RollingMean"] = TTestResults["PValue"].rolling(3).mean()
    # define subplots
    fig, ax = plt.subplots()

    # add first line to plot
    ax.plot(TTestResults.TimePeriod, TTestResults.PValue)

    # add x-axis label
    ax.set_xlabel('Zeitabschnitte unterteilt in' f" {TimePeriod} Jahre", fontsize=14)

    # add y-axis label
    ax.set_ylabel('P Wert der Null Hypothese', fontsize=16)

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()
    ax.get_shared_y_axes().join(ax, ax2)

    # add second line to plot
    ax2.plot(TTestResults.TimePeriod, TTestResults.RollingMean, color="red",label='rollierender Durchschnitt')
    
    
    text = '\n'.join((f"Region: {Region[0]} ;",f"Temperatur für Schneefall: {samplingTemperature} ;",f"ZeitAbschnitte: {TimePeriod}"))
    ax.text(1, 0.8,text, ha='left', va='center', transform=plt.gcf().transFigure)


def Regressionsanalyse(TTestResults):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    linregress_x = TTestResults["TimePeriod"]
    # linregress_x = df["Lufttemperatur Tagesminimum"]
    linregress_y = TTestResults["PValue"]

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        linregress_x, linregress_y)
    #plt.plot(linregress_x, linregress_y, 'o', label='original data')
    plt.plot(linregress_x, intercept + slope *
              linregress_x,color="orange", label='P Value Regression')
    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.6))
    plt.show()


def Analyse_Vortage(Region):
    if Region[0] == "StBernard.csv":
        Vortage[Region[0]] = np.nan
    else:
        # Cleanup für Analyse Vortage
        dfSnow = df.loc[df['SchneeTagesDifferenz'] > 0]
        dfSnow = dfSnow.replace("-", np.nan)
        dfSnowDez = dfSnow[pd.to_datetime(dfSnow['Datum']).dt.month == 12]
        dfVortagSnow = dfDez.loc[dfDez['SchneeVortag'] > 0]
        dfVorVortagSnow = dfDez.loc[dfDez['SchneeVorVortag'] > 0]
        dfDezforDays = dfDez[(pd.to_datetime(dfDez['Datum']).dt.year >= 1886) & (pd.to_datetime(dfDez['Datum']).dt.year <= 1900) | (pd.to_datetime(dfDez['Datum']).dt.year >= 1931) & (pd.to_datetime(dfDez['Datum']).dt.year <= 2020)]
        # Next Step
        print("Tage mit Temperatur für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]))
        print("Tage mit Temperatur für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]))
        TageTemperatur = len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature])
        TageSchneefall = len(dfDezforDays[dfDezforDays["SchneeVortag"] > 0])
        WahrscheinlichkeitTempSchneeFolgetag = round(len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]), 4)
        WahrscheinlichkeitTempSchneezweiterFolgetag = round(len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature]), 4)
        print("Tage mit Temperatur x + 2 für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]))
        print("Tage mit Temperatur x + 2 für Schnee:", len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]), "und an", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature + 2]))    
        print("Tage mit Niederschlag:", len(dfDezforDays[dfDezforDays["Niederschlag"] > 0]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Niederschlag"] > 0])) 
        print("Tage mit Niederschlag:", len(dfDezforDays[dfDezforDays["Niederschlag"] > 0]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Niederschlag"] > 0])) 
        WahrscheinlichkeitNiedersSchneeFolgetag = round(len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Niederschlag"] > 0]), 4)
        WahrscheinlichkeitNiedersSchneezweiterFolgetag = round(len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[dfDezforDays["Niederschlag"] > 0]), 4)
        print("Tage mit Niederschlag und Temperatur für Schnee:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]), "folgenden Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVortag"] > 0)]) / len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]))
        print("Tage mit Niederschlag und Temperatur für Schnee:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]), "und an", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]), "übernächsten Tagen hat es geschneit.")
        print("Wahrscheinlichkeit:", len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature) & (dfDezforDays["SchneeVorVortag"] > 0)]) / len(dfDezforDays[(dfDezforDays["Niederschlag"] > 0) & (dfDezforDays["Lufttemperatur Tagesmittel"] <= samplingTemperature)]))
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
        result = ztest(dfDezforDays["Lufttemperatur Tagesmittel"], dfVortagSnow["Lufttemperatur Tagesmittel"], value=0)
        print(result)
        Vortage[Region[0]] = [TageTemperatur, TageSchneefall, WahrscheinlichkeitTempSchneeFolgetag, WahrscheinlichkeitTempSchneezweiterFolgetag, WahrscheinlichkeitNiedersSchneeFolgetag, WahrscheinlichkeitNiedersSchneezweiterFolgetag, result[1]]
        # Hypothesentest Temperatur Vorvortag
        result = ztest(dfDezforDays["Lufttemperatur Tagesmittel"], dfVorVortagSnow["Lufttemperatur Tagesmittel"], value=0)
        print(result)



#Main Loop for Regions
for Region in regionsCsvData:
        df = pd.read_csv(Region[0],sep=Region[1])
        print(f"started Region: {Region[0]}")
        dfDez= CleanDF(df)
        
        #Reset Dataframes
        dfPropabilitiesSnowPerDecade = pd.DataFrame({'TimePeriod': [], 'Year': [], 'NProbabilitySnow': [], 'YearlyProbabilitySnow': [
        ], 'TempYearbeneathSamplingTemp': [], 'PPrecipationYearly': []})
        TTestResults = pd.DataFrame({'Region': [],'TimePeriod': [], 'TStatistic': [], 'PValue': []})
        
        TTestResults = LoopTimePeriod(Region)
        PlotTTestResults(TTestResults,Region)
        Regressionsanalyse(TTestResults)
        Analyse_Vortage(Region)


