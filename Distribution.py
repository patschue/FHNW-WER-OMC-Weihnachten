# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:01:16 2021

@author: Tom
"""

from scipy import stats

def FindmostfittingDistribution(MatrixColumn):
	list_of_dists = ['expon', 'logistic', 'norm']
	results = []
	for i in list_of_dists:
		dist = getattr(stats, i)
		param = dist.fit(MatrixColumn)
		a = stats.kstest(MatrixColumn, i, args=param)
		results.append((i,a[0],a[1]))
    
	results.sort(key=lambda x:float(x[2]), reverse=True)
	for j in results:
    		print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))
	return results